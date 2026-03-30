"""
train.py
--------
Unified training script for the CityNetwork benchmark.

Supports:
  - Standard GNN methods  : gcn, sage, cheb, sgformer, nagphormer, graphgps, gt
  - HierarchialRW (this work)     : hierarchial_rw   (anonymized random-walk Transformer encoder
                                     + linear classification head)

For GNN methods the pipeline is:
    CityNetwork → NeighborLoader → model(x, edge_index) → cross-entropy

For HierarchialRW the pipeline is:
    CityNetwork → SparseTensor adj → get_random_walk_batch(nodes)
               → Transformer encoder → linear head → cross-entropy

Usage:
    # GNN baseline
    python train.py --method gcn   --dataset paris --device 0

    # HierarchialRW
    python train.py --method hierarchial_rw --dataset paris --device 0 \
        --hierarchial_rw_walk_length 8 --hierarchial_rw_num_walks 16 --hierarchial_rw_hidden_dim 128
"""

import os
import csv
import math
import time
import datetime
import dataclasses
from copy import deepcopy
from itertools import chain
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_sparse import SparseTensor
from torch_geometric.utils import (
    to_undirected, coalesce, remove_self_loops, degree
)
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from einops import rearrange
from torch_cluster import random_walk as cluster_random_walk
from muon import SingleDeviceMuonWithAuxAdam

import wandb
import numpy as np
import argparse

from citynetworks import CityNetwork
from torch_geometric.datasets import Planetoid
from benchmark.configs import parse_method, parser_add_main_args
from benchmark.utils import eval_acc, count_parameters, plot_logging_info


# ===========================================================================
# HierarchialRW Model Components
# ===========================================================================

@dataclasses.dataclass(frozen=True)
class MupConfig:
    init_std: float = 0.01
    mup_width_multiplier: float = 2.0


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._init()

    def _init(self):
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: self.dim // 2].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        seq = torch.arange(self.max_seq_len, dtype=theta.dtype, device=theta.device)
        idx = torch.einsum("i,j->ij", seq, theta).float()
        self.register_buffer("cache", torch.stack([idx.cos(), idx.sin()], dim=-1), persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None):
        sl = x.size(1)
        rc = self.cache[:sl] if input_pos is None else self.cache[input_pos]
        xs = x.float().reshape(*x.shape[:-1], -1, 2)
        rc = rc.view(-1, xs.size(1), 1, xs.size(3), 2)
        out = torch.stack([
            xs[..., 0] * rc[..., 0] - xs[..., 1] * rc[..., 1],
            xs[..., 1] * rc[..., 0] + xs[..., 0] * rc[..., 1],
        ], -1).flatten(3)
        return out.type_as(x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_heads, seq_len, n_layer,
                 attn_dp=0.0, ffn_dp=0.0, resid_dp=0.0, drop_path=0.0,
                 cfg=MupConfig()):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.resid_dp   = resid_dp
        self.ffn_dp     = ffn_dp
        self.drop_path  = drop_path

        self.qkv  = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.attn_norm  = nn.Parameter(torch.ones(hidden_dim))
        self.ffn_norm   = nn.Parameter(torch.ones(hidden_dim))
        self.rope = RotaryPositionalEmbeddings(hidden_dim // num_heads, max_seq_len=seq_len)

        s = cfg.init_std / math.sqrt(cfg.mup_width_multiplier)
        sr = cfg.init_std / math.sqrt(2 * n_layer * cfg.mup_width_multiplier)
        for w, std in [(self.up.weight, sr), (self.gate.weight, sr), (self.down.weight, s),
                       (self.qkv.weight, s), (self.o.weight, s)]:
            nn.init.normal_(w, 0.0, std)

    def _stoch_depth(self, x):
        if self.drop_path <= 0.0 or not self.training:
            return x
        kp = 1.0 - self.drop_path
        mask = torch.floor(kp + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1),
                                           dtype=x.dtype, device=x.device))
        return x.div(kp) * mask

    def forward(self, x, offset=None):
        # Attention
        a = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.attn_norm
        q, k, v = self.qkv(a).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'n t (h d)->n t h d', h=self.num_heads) for t in (q, k, v)]
        q, k = self.rope(q, input_pos=offset), self.rope(k, input_pos=offset)
        q, k, v = [rearrange(t, 'n t h d->n h t d') for t in (q, k, v)]
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=1.0 / k.shape[-1])
        o = rearrange(o, 'n h t d->n t (h d)', h=self.num_heads)
        o = self.o(o)
        if self.resid_dp > 0:
            o = F.dropout(o, p=self.resid_dp, training=self.training)
        x = x + self._stoch_depth(o)
        # FFN (SwiGLU)
        f = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.ffn_norm
        f = self.down(F.silu(self.up(f)) * self.gate(f))
        if self.ffn_dp > 0:
            f = F.dropout(f, p=self.ffn_dp, training=self.training)
        return x + self._stoch_depth(f)


class HierarchialRWEncoder(nn.Module):
    """
    Anonymized random-walk Transformer encoder.
    Produces an L2-normalised embedding for each input source node.
    """
    def __init__(self, emb_dim, num_layers, hidden_dim, intermediate_dim, num_heads,
                 num_walks, seq_len, attn_dp=0.0, ffn_dp=0.0, resid_dp=0.0,
                 drop_path=0.0, cfg=MupConfig()):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, intermediate_dim, num_heads,
                             num_walks * seq_len, n_layer=num_layers,
                             attn_dp=attn_dp, ffn_dp=ffn_dp, resid_dp=resid_dp,
                             drop_path=drop_path, cfg=cfg)
            for _ in range(num_layers)
        ])
        self.emb         = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))
        nn.init.normal_(self.emb.weight, 0.0, cfg.init_std)

    def forward(self, x, anon_indices):
        _, ctx_len, _ = x.shape
        x = self.emb(x)
        for depth, idx in enumerate(reversed(anon_indices)):
            for layer in self.layers:
                x = layer(x, idx)
            if depth < len(anon_indices) - 1:
                x = x[:, -1, :]
                x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
                x = rearrange(x, '(n t) z->n t z', t=ctx_len)
        x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
        return F.normalize(x[:, -1, :], dim=-1)


class HierarchialRWNodeClassifier(nn.Module):
    """
    HierarchialRW encoder + linear classification head.
    Drop-in model for the benchmark — wraps the walk encoder so
    the training loop can call model.encode(x, adj, nodes) + model.classify(emb).
    """
    def __init__(self, emb_dim, num_classes, num_layers, hidden_dim, num_heads,
                 num_walks, walk_length, recurrent_steps=1,
                 attn_dp=0.0, ffn_dp=0.0, resid_dp=0.0, drop_path=0.0,
                 mup_init_std=0.01, mup_width_mult=2.0):
        super().__init__()
        self.hidden_dim       = hidden_dim
        self.num_walks        = num_walks
        self.walk_length      = walk_length
        self.recurrent_steps  = recurrent_steps

        intermediate_dim = hidden_dim * 4
        cfg = MupConfig(init_std=mup_init_std, mup_width_multiplier=mup_width_mult)

        self.encoder = HierarchialRWEncoder(
            emb_dim=emb_dim, num_layers=num_layers, hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim, num_heads=num_heads,
            num_walks=num_walks, seq_len=walk_length,
            attn_dp=attn_dp, ffn_dp=ffn_dp, resid_dp=resid_dp,
            drop_path=drop_path, cfg=cfg
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def encode(self, adj: SparseTensor, X: torch.Tensor, nodes: torch.Tensor,
               p: float = 1.0, q: float = 1.0) -> torch.Tensor:
        """Walk-encode a batch of nodes; returns (len(nodes), hidden_dim)."""
        feats, anon_idx = get_random_walk_batch(
            adj, X, nodes,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            recurrent_steps=self.recurrent_steps,
            p=p, q=q,
        )
        return self.encoder(feats, anon_idx)

    def forward(self, adj: SparseTensor, X: torch.Tensor, nodes: torch.Tensor,
                p: float = 1.0, q: float = 1.0) -> torch.Tensor:
        """Full forward: encode nodes → classify."""
        emb = self.encode(adj, X, nodes, p, q)
        return self.head(emb)


# ===========================================================================
# Walk Sampling & Anonymization (same as hierarchial_rw_e2e.py)
# ===========================================================================

@torch.no_grad()
def anonymize_rws(rws: torch.Tensor, rev_walks: bool = True) -> torch.Tensor:
    if rev_walks:
        rws = torch.flip(rws, dims=[-1])
    s, _  = torch.sort(rws, dim=-1)
    su    = torch.searchsorted(s, rws)
    c     = torch.full_like(s, fill_value=s.shape[-1])
    rw_i  = torch.arange(rws.shape[-1], device=rws.device)[None, :].expand_as(s)
    first = c.scatter_reduce_(-1, su, rw_i, reduce="amin")
    ret   = first.gather(-1, su)
    if rev_walks:
        ret = torch.flip(ret, dims=[-1])
    return ret


@torch.no_grad()
def get_random_walk_batch(
    adj: SparseTensor, x: torch.Tensor, start_nodes: torch.Tensor,
    walk_length: int, num_walks: int, recurrent_steps: int = 1,
    p: float = 1.0, q: float = 1.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    row, col, _ = adj.coo()
    row, col    = row.to(x.device), col.to(x.device)
    current     = start_nodes
    rws_list    = []

    for _ in range(recurrent_steps):
        src   = current.repeat_interleave(num_walks)
        walks = cluster_random_walk(row, col, src, walk_length - 1,
                                    p=p, q=q, num_nodes=adj.size(0))
        rws   = torch.flip(walks.view(current.size(0), num_walks, walk_length).flatten(1, 2),
                           dims=[-1])
        rws_list.append(rws)
        if recurrent_steps > 1:
            current = rws.reshape(-1)

    anon = [anonymize_rws(r, rev_walks=True) for r in rws_list]
    return x[rws_list[-1]], anon


# ===========================================================================
# Standard GNN train / eval loops  (NeighborLoader-based)
# ===========================================================================

def train_gnn(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    all_out, all_lbl = [], []
    for graph in loader:
        graph = graph.to(device)
        optimizer.zero_grad()
        out   = model(graph.x, graph.edge_index)[:graph.batch_size]
        lbl   = graph.y[:graph.batch_size]
        loss  = F.cross_entropy(out, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * graph.num_nodes
        all_out.append(out)
        all_lbl.append(lbl)
    acc, f1 = eval_acc(torch.cat(all_out), torch.cat(all_lbl))
    return acc, f1, total_loss / len(loader.dataset)


@torch.no_grad()
def eval_gnn(model, loader, device):
    model.eval()
    all_out, all_lbl = [], []
    for graph in loader:
        graph = graph.to(device)
        out   = model(graph.x, graph.edge_index)[:graph.batch_size]
        all_out.append(out)
        all_lbl.append(graph.y[:graph.batch_size])
    return eval_acc(torch.cat(all_out), torch.cat(all_lbl))


# ===========================================================================
# HierarchialRW train / eval loops  (walk-batch-based)
# ===========================================================================

def train_hierarchial_rw(model, node_idx, adj, X, optimizer, device,
                batch_size, p, q, accumulation_steps=1):
    """
    One epoch of HierarchialRW node classification training.
    Samples batches of node indices, encodes them via random walks, then
    computes cross-entropy loss against their ground-truth labels.
    """
    model.train()
    perm       = torch.randperm(node_idx.size(0), device=device)
    node_idx   = node_idx[perm]
    labels_all = adj._storage._value  # not used here; labels come from data.y

    # We carry the full label tensor from outside
    # (passed as `data.y` via the closure in main)
    raise RuntimeError("use train_hierarchial_rw_epoch() — see main()")


def train_hierarchial_rw_epoch(model, node_idx, y, adj, X, optimizer,
                      batch_size, device, p=1.0, q=1.0, accumulation_steps=1):
    model.train()
    perm     = torch.randperm(node_idx.size(0), device=device)
    shuffled = node_idx[perm]

    total_loss = 0.0
    all_out, all_lbl = [], []
    optimizer.zero_grad()

    batches = torch.split(shuffled, batch_size)
    for i, batch_nodes in enumerate(batches):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(adj, X, batch_nodes, p=p, q=q)
            labels = y[batch_nodes]
            loss   = F.cross_entropy(logits, labels) / accumulation_steps

        loss.backward()
        total_loss += loss.item() * accumulation_steps

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(batches):
            optimizer.step()
            optimizer.zero_grad()

        all_out.append(logits.detach().float())
        all_lbl.append(labels)

    acc, f1 = eval_acc(torch.cat(all_out), torch.cat(all_lbl))
    return acc, f1, total_loss / len(batches)


@torch.no_grad()
def eval_hierarchial_rw(model, node_idx, y, adj, X, batch_size, device, p=1.0, q=1.0):
    model.eval()
    all_out, all_lbl = [], []
    for batch_nodes in torch.split(node_idx, batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(adj, X, batch_nodes, p=p, q=q)
        all_out.append(logits.float().cpu())
        all_lbl.append(y[batch_nodes].cpu())
    return eval_acc(torch.cat(all_out), torch.cat(all_lbl))


# ===========================================================================
# Argument Parser  (extends existing parser_add_main_args)
# ===========================================================================

def parser_add_hierarchial_rw_args(parser):
    """HierarchialRW-specific arguments, prefixed with --hierarchial_rw_."""
    g = parser.add_argument_group('HierarchialRW')
    g.add_argument('--hierarchial_rw_walk_length',    type=int,   default=8)
    g.add_argument('--hierarchial_rw_num_walks',      type=int,   default=16)
    g.add_argument('--hierarchial_rw_recurrent_steps',type=int,   default=1)
    g.add_argument('--hierarchial_rw_hidden_dim',     type=int,   default=128)
    g.add_argument('--hierarchial_rw_num_layers',     type=int,   default=1)
    g.add_argument('--hierarchial_rw_num_heads',      type=int,   default=16)
    g.add_argument('--hierarchial_rw_attn_dropout',   type=float, default=0.1)
    g.add_argument('--hierarchial_rw_ffn_dropout',    type=float, default=0.1)
    g.add_argument('--hierarchial_rw_resid_dropout',  type=float, default=0.1)
    g.add_argument('--hierarchial_rw_drop_path',      type=float, default=0.05)
    g.add_argument('--hierarchial_rw_mup_init_std',   type=float, default=0.01)
    g.add_argument('--hierarchial_rw_mup_width_mult', type=float, default=2.0)
    g.add_argument('--hierarchial_rw_node2vec_p',     type=float, default=1.0)
    g.add_argument('--hierarchial_rw_node2vec_q',     type=float, default=1.0)
    g.add_argument('--hierarchial_rw_muon_min_lr',    type=float, default=1e-4)
    g.add_argument('--hierarchial_rw_muon_max_lr',    type=float, default=1e-3)
    g.add_argument('--hierarchial_rw_adam_lr',        type=float, default=5e-4)
    g.add_argument('--hierarchial_rw_batch_size',     type=int,   default=512,
                   help='Nodes per walk-encoding batch (separate from --batch_size)')
    return parser


# ===========================================================================
# HierarchialRW LR Schedule
# ===========================================================================

def trapezoidal_lr(step, max_lr, min_lr, warmup, cool, total):
    if step <= warmup:
        return (step / warmup) * (max_lr - min_lr) + min_lr
    if step <= total - cool:
        return max_lr
    return max(min_lr, ((total - step) / max(cool, 1)) * (max_lr - min_lr) + min_lr)


# ===========================================================================
# Results CSV Helper (crash-recoverable incremental saving)
# ===========================================================================

def save_result_row(results_dir, exp_name, dataset, method, seed, run,
                    best_val_acc, best_val_f1, test_acc, test_f1,
                    extra: dict = None):
    os.makedirs(os.path.join(results_dir, exp_name), exist_ok=True)
    fpath  = os.path.join(results_dir, exp_name, f"{dataset}-{method}.csv")
    fields = ['dataset', 'method', 'seed', 'run',
              'best_val_acc', 'best_val_f1', 'test_acc', 'test_f1']
    row    = dict(dataset=dataset, method=method, seed=seed, run=run,
                  best_val_acc=round(best_val_acc, 6),
                  best_val_f1=round(best_val_f1, 6),
                  test_acc=round(test_acc, 6),
                  test_f1=round(test_f1, 6))
    if extra:
        fields += list(extra.keys())
        row.update(extra)
    write_header = not os.path.exists(fpath)
    with open(fpath, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ===========================================================================
# Per-seed main  (handles both GNN and HierarchialRW)
# ===========================================================================

def main(args, logging_dict):
    seed_everything(args.seed)
    device     = torch.device(f"cuda:{args.device}")
    is_hierarchial_rw   = (args.method.lower() == 'hierarchial_rw')

    # ── W&B init ─────────────────────────────────────────────────────────────
    run = wandb.init(
        project=args.wandb_project,
        entity=getattr(args, 'wandb_entity', None) or None,
        group=f"{args.experiment_name}/{args.dataset}/{args.method}",
        name=f"seed{args.seed}",
        config=vars(args),
        reinit=True,
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=f'data/{args.dataset}', name=args.dataset)
        data    = dataset[0]
    else:
        dataset = CityNetwork(root=f'data/{args.dataset}', name=args.dataset)
        data    = dataset[0]

    data = data.to(device)
    num_features = data.x.size(1)
    num_classes  = dataset.num_classes

    wandb.config.update({
        'num_nodes':    data.num_nodes,
        'num_edges':    data.edge_index.size(1),
        'num_features': num_features,
        'num_classes':  num_classes,
    }, allow_val_change=True)

    # ── Build adjacency (needed by HierarchialRW; harmless for GNNs) ────────────────
    ei = data.edge_index
    if data.is_directed():
        ei = to_undirected(ei)
    ei, _ = coalesce(ei, None, num_nodes=data.num_nodes)
    ei, _ = remove_self_loops(ei)
    ew    = torch.ones(ei.size(1), device=device)
    adj   = SparseTensor.from_edge_index(ei, ew, [data.num_nodes, data.num_nodes])

    # ── Masks ─────────────────────────────────────────────────────────────────
    train_mask = data.train_mask
    val_mask   = data.val_mask
    test_mask  = data.test_mask
    # Handle per-split mask shape (N, num_splits) → take first split
    if train_mask.dim() == 2:
        train_mask = train_mask[:, 0]
        val_mask   = val_mask[:, 0]
        test_mask  = test_mask[:, 0]

    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    val_idx   = val_mask.nonzero(as_tuple=False).view(-1)
    test_idx  = test_mask.nonzero(as_tuple=False).view(-1)
    y         = data.y.view(-1)

    # ── Model & Optimizer ────────────────────────────────────────────────────
    if is_hierarchial_rw:
        model = HierarchialRWNodeClassifier(
            emb_dim=num_features,
            num_classes=num_classes,
            num_layers=args.hierarchial_rw_num_layers,
            hidden_dim=args.hierarchial_rw_hidden_dim,
            num_heads=args.hierarchial_rw_num_heads,
            num_walks=args.hierarchial_rw_num_walks,
            walk_length=args.hierarchial_rw_walk_length,
            recurrent_steps=args.hierarchial_rw_recurrent_steps,
            attn_dp=args.hierarchial_rw_attn_dropout,
            ffn_dp=args.hierarchial_rw_ffn_dropout,
            resid_dp=args.hierarchial_rw_resid_dropout,
            drop_path=args.hierarchial_rw_drop_path,
            mup_init_std=args.hierarchial_rw_mup_init_std,
            mup_width_mult=args.hierarchial_rw_mup_width_mult,
        ).to(device)

        # Muon for matrix weights, Adam for scalars + head
        w_matrix = [p for p in model.encoder.parameters() if p.ndim >= 2 and p.requires_grad]
        w_scalar = [p for p in model.encoder.parameters() if p.ndim <  2 and p.requires_grad]
        if args.hierarchial_rw_recurrent_steps <= 1:
            w_scalar = [p for p in w_scalar if p is not model.encoder.norm_weight]
        optimizer = SingleDeviceMuonWithAuxAdam([
            dict(params=w_matrix,              use_muon=True,  lr=0.04, weight_decay=0.01),
            dict(params=w_scalar,              use_muon=False, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.0),
            dict(params=model.head.parameters(),use_muon=False, lr=args.hierarchial_rw_adam_lr,
                 betas=(0.9, 0.95), weight_decay=args.weight_decay),
        ])

        # Gradient accumulation for high-recurrence runs
        acc_steps = max(1, 256 // args.hierarchial_rw_batch_size) if args.hierarchial_rw_recurrent_steps > 1 else 1

        # LR schedule state
        total_steps  = math.ceil(train_idx.size(0) / args.hierarchial_rw_batch_size) * args.epochs
        warmup_steps = total_steps // 10
        cool_steps   = total_steps - warmup_steps
        global_step  = 0

        # Node features in bfloat16 for autocast
        X = data.x.bfloat16()

        train_loader = None  # not used by HierarchialRW
        val_loader   = None
        test_loader  = None
    else:
        # Standard GNN
        model = parse_method(args, dataset, data, device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        train_loader = NeighborLoader(
            data, num_neighbors=[-1] * args.num_layers,
            batch_size=args.batch_size, input_nodes=train_mask,
        )
        val_loader = NeighborLoader(
            data, num_neighbors=[-1] * args.num_layers,
            batch_size=args.batch_size, input_nodes=val_mask,
        )
        test_loader = NeighborLoader(
            data, num_neighbors=[-1] * args.num_layers,
            batch_size=args.batch_size, input_nodes=test_mask,
        )
        acc_steps   = 1
        global_step = 0
        total_steps = args.epochs * math.ceil(train_idx.size(0) / args.batch_size)
        warmup_steps = cool_steps = 0

    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"  Dataset : {args.dataset}  |  Method : {args.method}")
    print(f"  Nodes   : {data.num_nodes:,}  |  Edges : {data.edge_index.size(1):,}")
    print(f"  Params  : {n_params:,}")
    print(f"{'='*60}\n")
    wandb.config.update({'num_params': n_params}, allow_val_change=True)

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    best_val_f1    = 0.0
    best_test_acc  = 0.0
    best_test_f1   = 0.0
    best_model_sd  = None

    for epoch in range(args.epochs):
        t0 = time.time()

        if is_hierarchial_rw:
            # Update Muon LR
            for pg in optimizer.param_groups:
                if pg.get('use_muon', False):
                    pg['lr'] = trapezoidal_lr(
                        global_step, args.hierarchial_rw_muon_max_lr, args.hierarchial_rw_muon_min_lr,
                        warmup_steps, cool_steps, total_steps
                    )

            train_acc, train_f1, train_loss = train_hierarchial_rw_epoch(
                model=model,
                node_idx=train_idx,
                y=y,
                adj=adj,
                X=X,
                optimizer=optimizer,
                batch_size=args.hierarchial_rw_batch_size,
                device=device,
                p=args.hierarchial_rw_node2vec_p,
                q=args.hierarchial_rw_node2vec_q,
                accumulation_steps=acc_steps,
            )
            global_step += math.ceil(train_idx.size(0) / args.hierarchial_rw_batch_size)

            val_acc,  val_f1  = eval_hierarchial_rw(model, val_idx,  y, adj, X,
                                            args.hierarchial_rw_batch_size, device,
                                            args.hierarchial_rw_node2vec_p, args.hierarchial_rw_node2vec_q)
            test_acc, test_f1 = eval_hierarchial_rw(model, test_idx, y, adj, X,
                                            args.hierarchial_rw_batch_size, device,
                                            args.hierarchial_rw_node2vec_p, args.hierarchial_rw_node2vec_q)
        else:
            train_acc, train_f1, train_loss = train_gnn(model, train_loader, optimizer, device)
            val_acc,   val_f1               = eval_gnn(model, val_loader,   device)
            test_acc,  test_f1              = eval_gnn(model, test_loader,   device)

        epoch_time = time.time() - t0

        # W&B logging
        wandb.log({
            'epoch':      epoch + 1,
            'train/loss': train_loss,
            'train/acc':  train_acc,
            'train/f1':   train_f1,
            'val/acc':    val_acc,
            'val/f1':     val_f1,
            'test/acc':   test_acc,
            'test/f1':    test_f1,
            'epoch_time': epoch_time,
        })

        # Best checkpoint
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_f1   = val_f1
            best_test_acc = test_acc
            best_test_f1  = test_f1
            best_model_sd = deepcopy(model.state_dict())
            wandb.run.summary['best_val_acc']  = best_val_acc
            wandb.run.summary['best_test_acc'] = best_test_acc

        # Progress display
        if (epoch + 1) % args.display_step == 0 or epoch == 0:
            msg = (f"E{epoch+1:>5} | loss {train_loss:.4f} | "
                   f"trn {train_acc*100:.2f}% | "
                   f"val {val_acc*100:.2f}% | "
                   f"tst {test_acc*100:.2f}% | "
                   f"{epoch_time:.1f}s")
            print(f"{msg:<100}", end="\r", flush=True)

    print()  # newline after \r progress

    # Accumulate into logging_dict for cross-seed aggregation
    logging_dict['train_acc'].append(best_val_acc)     # kept for compat
    logging_dict['val_acc'].append(best_val_acc)
    logging_dict['test_acc'].append(best_test_acc)
    logging_dict['val_f1'].append(best_val_f1)
    logging_dict['test_f1'].append(best_test_f1)

    # Save model checkpoint
    if args.save_model and best_model_sd is not None:
        ckpt_dir = os.path.join('results', args.experiment_name, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir,
                                 f"{args.dataset}-{args.method}-seed{args.seed}.pt")
        torch.save({'state_dict': best_model_sd, 'args': vars(args)}, ckpt_path)
        artifact = wandb.Artifact(
            f"{args.dataset}-{args.method}-seed{args.seed}", type='model'
        )
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact)

    # Incremental result CSV (crash-recoverable)
    save_result_row(
        results_dir='results',
        exp_name=args.experiment_name,
        dataset=args.dataset,
        method=args.method,
        seed=args.seed,
        run=0,
        best_val_acc=best_val_acc,
        best_val_f1=best_val_f1,
        test_acc=best_test_acc,
        test_f1=best_test_f1,
        extra={'num_params': n_params},
    )

    wandb.finish()
    return best_val_acc, best_val_f1, best_test_acc, best_test_f1


# ===========================================================================
# Entry Point  (multi-seed runner)
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CityNetwork Benchmark — GNNs + HierarchialRW'
    )
    parser = parser_add_main_args(parser)
    parser = parser_add_hierarchial_rw_args(parser)

    # W&B flags
    parser.add_argument('--wandb_project', type=str, default='gnn-benchmark')
    parser.add_argument('--wandb_entity',  type=str, default='')

    args = parser.parse_args()

    logging_dict = {
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'val_f1':    [], 'test_f1': [],
    }

    seeds = list(range(args.seed, args.seed + args.runs))

    for seed in seeds:
        args.seed = seed
        best_val_acc, best_val_f1, best_test_acc, best_test_f1 = main(args, logging_dict)
        print(f"  Seed {seed} done — val {best_val_acc*100:.2f}%  test {best_test_acc*100:.2f}%")

    # ── Aggregate across seeds ───────────────────────────────────────────────
    def mean_std(lst):
        a = np.array(lst)
        return a.mean(), a.std()

    mv, sv = mean_std(logging_dict['val_acc'])
    mt, st = mean_std(logging_dict['test_acc'])

    sep = '─' * 52
    print(f"\n{sep}")
    print(f"  Results over {args.runs} seeds  |  {args.dataset} / {args.method}")
    print(sep)
    print(f"  {'Split':<10} {'Mean':>10}  {'Std':>10}")
    print(sep)
    print(f"  {'Val':<10} {mv*100:>9.2f}%  {sv*100:>9.2f}%")
    print(f"  {'Test':<10} {mt*100:>9.2f}%  {st*100:>9.2f}%")
    print(f"{sep}\n")

    # Aggregated W&B run
    agg = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=f"{args.experiment_name}/{args.dataset}/{args.method}",
        name='aggregated',
        reinit=True,
    )
    wandb.log({
        'agg/mean_val_acc':  mv, 'agg/std_val_acc':  sv,
        'agg/mean_test_acc': mt, 'agg/std_test_acc': st,
    })
    wandb.finish()

    # Aggregate CSV row
    save_result_row(
        results_dir='results',
        exp_name=args.experiment_name,
        dataset=args.dataset,
        method=args.method,
        seed=-1,  # -1 flags this as the aggregate row
        run=-1,
        best_val_acc=mv,
        best_val_f1=mean_std(logging_dict['val_f1'])[0],
        test_acc=mt,
        test_f1=mean_std(logging_dict['test_f1'])[0],
        extra={
            'std_val_acc':  sv,
            'std_test_acc': st,
            'num_seeds':    args.runs,
        },
    )
    print(f"Results saved → results/{args.experiment_name}/{args.dataset}-{args.method}.csv")
