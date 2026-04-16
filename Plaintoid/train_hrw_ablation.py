"""
Ablation: Walk Aggregation Strategy
====================================
--walk_agg flatten  (default / current):
    Walks are flattened to [N, num_walks * walk_length, D] and fed as one
    long sequence to the Transformer.

--walk_agg concat:
    Each walk is processed independently through the Transformer →
    produces num_walks vectors of size hidden_dim → these are concatenated
    along the feature dim → projected back to hidden_dim.
    Shape flow: [N, num_walks, walk_length, D] → [N*num_walks, walk_length, D]
                → Transformer → [N*num_walks, hidden_dim]
                → reshape [N, num_walks * hidden_dim]
                → Linear(num_walks * hidden_dim, hidden_dim) → [N, hidden_dim]
"""

import os
os.environ["WANDB_MODE"] = "online"
import wandb
import numpy as np
import argparse
import json
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (
    train_test_split_edges,
    negative_sampling,
    to_undirected,
)
import torch_geometric
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from einops import rearrange
from torch.utils.data import DataLoader
import tqdm.auto as tqdm
import math
import pdb
import pickle
import dataclasses
from muon import SingleDeviceMuonWithAuxAdam, MuonWithAuxAdam
import pandas as pd
import time
from itertools import chain
import torch_geometric.transforms as T
from typing import Tuple, List, Dict
from typing import Optional
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
from torch_geometric.utils import (
    to_undirected,
    coalesce,
    remove_self_loops,
    remove_isolated_nodes,
    from_networkx,
    to_networkx,
    degree,
)
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from torch_cluster import random_walk as cluster_random_walk

from torch.utils.data import DataLoader
from torch_geometric.data import Data
import torch
from torch_geometric.utils import (
    to_undirected,
    coalesce,
    remove_self_loops,
    remove_isolated_nodes,
)
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
import random
from typing import Tuple

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def str_to_bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('true', '1', 't', 'yes', 'y'):
        return True
    elif val.lower() in ('false', '0', 'f', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {val}')


def downsample_edges(edge_index, ratio=0.5, seed=42):
    torch.manual_seed(seed)
    num_edges = edge_index.size(1)
    sample_size = int(num_edges * ratio)
    perm = torch.randperm(num_edges)[:sample_size]
    return edge_index[:, perm]


def sample_negative_edges(pos_edge_index, num_nodes, num_neg_samples, device):
    return negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method='sparse'
    ).to(device)


@torch.no_grad()
def sample_walks_for_nodes(adj, node_indices, walk_length, num_walks, p=1.0, q=1.0):
    row, col, _ = adj.coo()
    row = row.to(node_indices.device)
    col = col.to(node_indices.device)
    sources_repeated = node_indices.repeat_interleave(num_walks)
    walks = cluster_random_walk(
        row, col, sources_repeated, walk_length - 1, p=p, q=q,
        num_nodes=adj.size(0)
    )
    num_nodes = node_indices.size(0)
    walks = walks.view(num_nodes, num_walks, walk_length)
    walks = torch.flip(walks, dims=[-1])   # reverse: target -> source
    return walks   # [N, num_walks, walk_length]


@torch.no_grad()
def anonymize_walks(walks: torch.Tensor, rev_walks: bool = True) -> torch.Tensor:
    """walks: [*, walk_len]"""
    if rev_walks:
        walks = torch.flip(walks, dims=[-1])
    s, si = torch.sort(walks, dim=-1)
    su = torch.searchsorted(s, walks)
    c = torch.full_like(s, fill_value=s.shape[-1])
    rw_i = torch.arange(walks.shape[-1], device=walks.device)[None, :].expand_as(s)
    first = c.scatter_reduce_(-1, su, rw_i, reduce="amin")
    ret = first.gather(-1, su)
    if rev_walks:
        ret = torch.flip(ret, dims=[-1])
    return ret


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

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
        self.rope_init()

    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096):
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None):
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack([
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ], -1).flatten(3)
        return x_out.type_as(x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_heads, seq_len, n_layer,
                 attn_dropout_p=0.0, ffn_dropout_p=0.0, resid_dropout_p=0.0,
                 drop_path_p=0.0, config=MupConfig()):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_dropout_p = attn_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.drop_path_p = drop_path_p
        self.up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.input_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.attn_norm_weight  = nn.Parameter(torch.ones(hidden_dim))
        self.qkv  = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RotaryPositionalEmbeddings(dim=(hidden_dim // num_heads), max_seq_len=seq_len)

        s = config.init_std
        m = config.mup_width_multiplier
        nn.init.normal_(self.up.weight,   std=s / math.sqrt(2 * n_layer * m))
        nn.init.normal_(self.gate.weight, std=s / math.sqrt(2 * n_layer * m))
        nn.init.normal_(self.down.weight, std=s / math.sqrt(m))
        nn.init.normal_(self.qkv.weight,  std=s / math.sqrt(m))
        nn.init.normal_(self.o.weight,    std=s / math.sqrt(m))

    def _drop_path(self, x):
        if self.drop_path_p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_path_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep) * torch.floor(keep + torch.rand(shape, dtype=x.dtype, device=x.device))

    def forward(self, x, offset=None):
        ax = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.attn_norm_weight
        q, k, v = self.qkv(ax).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'n t (h d) -> n t h d', h=self.num_heads) for t in (q, k, v)]
        q = self.rope(q, input_pos=offset)
        k = self.rope(k, input_pos=offset)
        q, k, v = [rearrange(t, 'n t h d -> n h t d') for t in (q, k, v)]
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=1.0 / k.shape[-1])
        o = rearrange(o, 'n h t d -> n t (h d)', h=self.num_heads)
        attn_out = self.o(o)
        if self.resid_dropout_p > 0:
            attn_out = F.dropout(attn_out, p=self.resid_dropout_p, training=self.training)
        x = x + self._drop_path(attn_out)

        fx = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.input_norm_weight
        ffn_out = self.down(F.silu(self.up(fx)) * self.gate(fx))
        if self.ffn_dropout_p > 0:
            ffn_out = F.dropout(ffn_out, p=self.ffn_dropout_p, training=self.training)
        return x + self._drop_path(ffn_out)


class Transformer(nn.Module):
    def __init__(self, emb_dim, num_layers, hidden_dim, intermediate_dim, num_heads, seq_len,
                 attn_dropout_p=0.0, ffn_dropout_p=0.0, resid_dropout_p=0.0,
                 drop_path_p=0.0, config=MupConfig()):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim, intermediate_dim, num_heads, seq_len, n_layer=num_layers,
                attn_dropout_p=attn_dropout_p, ffn_dropout_p=ffn_dropout_p,
                resid_dropout_p=resid_dropout_p, drop_path_p=drop_path_p, config=config
            ) for _ in range(num_layers)
        ])
        self.emb = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))
        nn.init.normal_(self.emb.weight, std=config.init_std)

    def forward(self, x, anon_indices, *, apply_emb=True):
        """x: [B, seq_len, emb_dim]  →  [B, hidden_dim]"""
        if apply_emb:
            x = self.emb(x)
        for layer in self.layers:
            x = layer(x, anon_indices)
        x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
        return F.normalize(x[:, -1, :], dim=-1)


# ---------------------------------------------------------------------------
# DPCWUE — two walk aggregation strategies
# ---------------------------------------------------------------------------

class DPCWUE(nn.Module):
    """
    walk_agg = 'flatten':
        Flatten [N, W, L] -> [N, W*L] and process as one long sequence.
        (original behaviour)

    walk_agg = 'concat':
        Process each walk independently [N*W, L] → Transformer → [N*W, D],
        then reshape to [N, W*D] and project back to [N, D] via a linear.
        This gives each walk its own representation before combining.
    """

    def __init__(
        self,
        num_nodes: int,
        emb_dim: int,
        num_layers: int,
        hidden_dim: int,
        intermediate_dim: int,
        num_heads: int,
        num_walks: int,
        walk_length: int,
        recurrent_steps: int,
        walk_agg: str = 'flatten',        # 'flatten' | 'concat'
        attn_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        drop_path_p: float = 0.0,
        config: MupConfig = MupConfig(),
    ):
        super().__init__()
        assert walk_agg in ('flatten', 'concat'), f"Unknown walk_agg: {walk_agg}"
        self.num_nodes       = num_nodes
        self.hidden_dim      = hidden_dim
        self.num_walks       = num_walks
        self.walk_length     = walk_length
        self.recurrent_steps = recurrent_steps
        self.walk_agg        = walk_agg

        if walk_agg == 'flatten':
            # One transformer, seq_len = num_walks * walk_length
            seq_len = num_walks * walk_length
        else:
            # One transformer per-walk, seq_len = walk_length
            seq_len = walk_length

        self.transformer = Transformer(
            emb_dim=hidden_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            attn_dropout_p=attn_dropout_p,
            ffn_dropout_p=ffn_dropout_p,
            resid_dropout_p=resid_dropout_p,
            drop_path_p=drop_path_p,
            config=config,
        )
        self.transformer = torch.compile(self.transformer, dynamic=True)

        if walk_agg == 'concat':
            # Project concatenated per-walk representations back to hidden_dim
            self.walk_proj = nn.Linear(num_walks * hidden_dim, hidden_dim, bias=False)
            nn.init.normal_(self.walk_proj.weight, std=config.init_std)
        else:
            self.walk_proj = None

        self.input_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        nn.init.normal_(self.input_proj.weight, std=config.init_std)

    # ------------------------------------------------------------------
    def _encode_walks_flatten(self, walk_features, walks):
        """
        walk_features : [N, W, L, D]
        walks         : [N, W, L]   (node indices, for anonymization)
        → [N, hidden_dim]
        """
        N, W, L, D = walk_features.shape
        # Flatten walk and length dims: [N, W*L, D]
        x = walk_features.reshape(N, W * L, D)
        # Anonymize on flattened walks [N, W*L]
        anon = anonymize_walks(walks.reshape(N, W * L), rev_walks=True)
        return self.transformer(x, anon, apply_emb=False)   # [N, D]

    def _encode_walks_concat(self, walk_features, walks):
        """
        walk_features : [N, W, L, D]
        walks         : [N, W, L]
        → [N, hidden_dim]
        """
        N, W, L, D = walk_features.shape
        # Process each walk independently: [N*W, L, D]
        x = walk_features.reshape(N * W, L, D)
        # Anonymize per-walk: [N*W, L]
        anon = anonymize_walks(walks.reshape(N * W, L), rev_walks=True)
        per_walk = self.transformer(x, anon, apply_emb=False)   # [N*W, D]
        # Concatenate walks: [N, W*D] → project → [N, D]
        concat = per_walk.reshape(N, W * self.hidden_dim)
        return F.normalize(self.walk_proj(concat), dim=-1)

    # ------------------------------------------------------------------
    def forward(self, node_features, adj, batch_nodes, p=1.0, q=1.0,
                track_level_sizes=False):
        device = node_features.device

        level_nodes    = [batch_nodes]
        walks_per_step = []

        for _ in range(self.recurrent_steps):
            # walks shape: [N, W, L]
            walks = sample_walks_for_nodes(
                adj, level_nodes[-1], self.walk_length, self.num_walks, p, q
            )
            walks_per_step.append(walks)
            next_nodes = torch.unique(walks.reshape(-1))
            level_nodes.append(next_nodes)

        if track_level_sizes:
            self.last_level_sizes = [n.numel() for n in level_nodes]

        h_all = self.input_proj(node_features)   # [num_nodes, D]

        for step in reversed(range(self.recurrent_steps)):
            walks = walks_per_step[step]          # [|level_nodes[step]|, W, L]
            walk_features = h_all[walks]           # [N, W, L, D]

            if self.walk_agg == 'flatten':
                h_level = self._encode_walks_flatten(walk_features, walks)
            else:
                h_level = self._encode_walks_concat(walk_features, walks)

            h_all[level_nodes[step]] = h_level.to(h_all.dtype)

        return h_all[batch_nodes]


# ---------------------------------------------------------------------------
# Link predictor + losses
# ---------------------------------------------------------------------------

class LinkPredictorMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, num_layers=3, dropout=0):
        super().__init__()
        self.lins = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_dim, out_dim))
        else:
            self.lins.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, h1, h2):
        x = h1 * h2
        for lin in self.lins[:-1]:
            x = F.relu(F.dropout(lin(x), p=self.dropout, training=self.training))
        return torch.sigmoid(self.lins[-1](x))


def binary_cross_entropy_loss(pos_scores, neg_scores):
    return -torch.log(pos_scores + 1e-15).mean() - torch.log(1 - neg_scores + 1e-15).mean()


def trapezoidal_lr_schedule(global_batch_idx, max_lr, min_lr, warmup, cool, total_batches):
    if global_batch_idx <= warmup:
        return (global_batch_idx / warmup) * (max_lr - min_lr) + min_lr
    elif global_batch_idx <= (total_batches - cool):
        return max_lr
    else:
        return ((total_batches - global_batch_idx) / cool) * (max_lr - min_lr) + min_lr


# ---------------------------------------------------------------------------
# Evaluation helpers  (unchanged from original)
# ---------------------------------------------------------------------------

def get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, k_list):
    result = {}
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    for K in k_list:
        result[f'Hits@{K}'] = result_hit_val[f'Hits@{K}']
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred,
                                   neg_val_pred.repeat(pos_val_pred.size(0), 1))
    result['MRR'] = result_mrr_val['MRR']
    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int),
                           torch.zeros(neg_val_pred.size(0), dtype=int)])
    result_auc = evaluate_auc(val_pred, val_true)
    result['AUC'] = result_auc['AUC']
    result['AP']  = result_auc['AP']
    return result


@torch.no_grad()
def test_edge_dp(model, link_predictor, adj, node_features, config, edge_index, batch_size):
    input_data = edge_index.t()
    all_scores = []
    for perm in DataLoader(range(input_data.size(0)), batch_size=batch_size):
        batch_edge_index = input_data[perm].t()
        nodes = batch_edge_index.unique()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings = model(node_features, adj, nodes,
                               p=config['node2vec_p'], q=config['node2vec_q'])
            node_to_idx = {n.item(): i for i, n in enumerate(nodes)}
            u = embeddings[[node_to_idx[n.item()] for n in batch_edge_index[0]]]
            v = embeddings[[node_to_idx[n.item()] for n in batch_edge_index[1]]]
            scores = link_predictor(u, v) if config['use_mlp'] else torch.sigmoid((u * v).sum(-1))
        all_scores.append(scores.cpu())
    return torch.cat(all_scores, dim=0).float()


@torch.no_grad()
def evaluate_link_prediction_dp(model, link_predictor, edge_index, neg_edge_index,
                                  adj, node_features, config,
                                  evaluator_hit, evaluator_mrr, device,
                                  eval_batch_size=512):
    model.eval()
    if config['use_mlp']:
        link_predictor.eval()
    pos_scores = test_edge_dp(model, link_predictor, adj, node_features, config,
                               edge_index, batch_size=eval_batch_size)
    neg_scores = test_edge_dp(model, link_predictor, adj, node_features, config,
                               neg_edge_index, batch_size=eval_batch_size)
    return get_metric_score(
        evaluator_hit, evaluator_mrr,
        torch.flatten(pos_scores), torch.flatten(neg_scores),
        config.get('hits_k', [1, 10, 50, 100])
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_config():
    parser = argparse.ArgumentParser(description="CWUE Walk-Aggregation Ablation")

    # ── Ablation flag ─────────────────────────────────────────────────────
    parser.add_argument('--walk_agg', type=str, default='flatten',
                        choices=['flatten', 'concat'],
                        help='"flatten": all walks as one long sequence (default). '
                             '"concat": each walk encoded independently then concat-projected.')

    # ── Run ───────────────────────────────────────────────────────────────
    g = parser.add_argument_group('Run')
    g.add_argument('--seed',              type=int,   default=2025)
    g.add_argument('--num_epochs',        type=int,   default=3000)
    g.add_argument('--global_batch_size', type=int,   default=256)
    g.add_argument('--patience',          type=int,   default=100)
    g.add_argument('--train_edge_downsample_ratio', type=float, default=1.0)
    g.add_argument('--eval_metric',       type=str,   default='MRR')

    # ── Data ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group('Data')
    g.add_argument('--data_root',       type=str,        default='data/Cora')
    g.add_argument('--data_name',       type=str,        default='Cora')
    g.add_argument('--val_split_ratio', type=float,      default=0.15)
    g.add_argument('--test_split_ratio',type=float,      default=0.05)
    g.add_argument('--use_fixed_splits',type=str_to_bool,default=False)
    g.add_argument('--split_dir',       type=str,        default='data/Cora/fixed_splits')
    g.add_argument('--use_laplacian_pe',type=str_to_bool,default=False)
    g.add_argument('--laplacian_pe_path',type=str,       default='data/Cora/laplacian_pe.pt')

    # ── Model ─────────────────────────────────────────────────────────────
    g = parser.add_argument_group('Model')
    g.add_argument('--num_layers',              type=int,   default=1)
    g.add_argument('--hidden_dim',              type=int,   default=128)
    g.add_argument('--intermediate_dim_multiplier', type=int, default=4)
    g.add_argument('--num_heads',               type=int,   default=16)
    g.add_argument('--recurrent_steps',         type=int,   default=3)
    g.add_argument('--mup_init_std',            type=float, default=0.01)
    g.add_argument('--mup_width_multiplier',    type=float, default=2.0)

    # ── Walk ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group('Walk')
    g.add_argument('--walk_length', type=int,   default=8)
    g.add_argument('--num_walks',   type=int,   default=16)
    g.add_argument('--node2vec_p',  type=float, default=1.0)
    g.add_argument('--node2vec_q',  type=float, default=1.0)

    # ── Optimizer ─────────────────────────────────────────────────────────
    g = parser.add_argument_group('Optimizer')
    g.add_argument('--muon_min_lr',    type=float, default=1e-4)
    g.add_argument('--muon_max_lr',    type=float, default=1e-3)
    g.add_argument('--adam_max_lr',    type=float, default=1e-4)
    g.add_argument('--adam_min_lr',    type=float, default=0.0)
    g.add_argument('--grad_clip_norm', type=float, default=0.1)

    # ── MLP ───────────────────────────────────────────────────────────────
    g = parser.add_argument_group('MLP')
    g.add_argument('--use_mlp',       type=str_to_bool, default=True)
    g.add_argument('--mlp_num_layers',type=int,         default=3)
    g.add_argument('--mlp_lr',        type=float,       default=1e-3)
    g.add_argument('--mlp_dropout',   type=float,       default=0.1)

    # ── Regularization ────────────────────────────────────────────────────
    g = parser.add_argument_group('Regularization')
    g.add_argument('--attn_dropout',  type=float, default=0.1)
    g.add_argument('--ffn_dropout',   type=float, default=0.1)
    g.add_argument('--resid_dropout', type=float, default=0.1)
    g.add_argument('--drop_path',     type=float, default=0.05)

    # ── Misc ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group('Misc')
    g.add_argument('--neg_sample_ratio', type=int,   default=1)
    g.add_argument('--hits_k', type=int, nargs='+',  default=[1, 10, 50, 100])
    g.add_argument('--wb_entity',  type=str, default='cshao676')
    g.add_argument('--wb_project', type=str, default='walk_agg_ablation')

    args = parser.parse_args()
    config = vars(args)
    config['intermediate_dim'] = config['hidden_dim'] * config['intermediate_dim_multiplier']
    del config['intermediate_dim_multiplier']
    return config


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_experiment(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  walk_agg = {config['walk_agg'].upper()}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    if config['data_name'] in ['Cora', 'PubMed', 'CiteSeer']:
        dataset = Planetoid(root=config['data_root'], name=config['data_name'])
        data = dataset[0].to(device)
    else:
        raise ValueError(f"Dataset {config['data_name']} not supported in this script.")

    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    if config['use_laplacian_pe']:
        lap_pe = torch.load(config['laplacian_pe_path'], map_location=device, weights_only=False)
        if lap_pe.size(0) != data.x.size(0):
            lap_pe = lap_pe[:data.x.size(0)]
        data.x = torch.cat([data.x, lap_pe], dim=1)

    config['emb_dim'] = data.x.size(1)

    transform = T.RandomLinkSplit(
        num_val=config['val_split_ratio'],
        num_test=config['test_split_ratio'],
        is_undirected=True,
        add_negative_train_samples=False
    )
    train_data, val_data, test_data = transform(data)
    full_train_pos_edge_index = train_data.edge_index.to(device)
    val_pos_edge_index  = val_data.edge_label_index[:,  val_data.edge_label  == 1].to(device)
    val_neg_edge_index  = val_data.edge_label_index[:,  val_data.edge_label  == 0].to(device)
    test_pos_edge_index = test_data.edge_label_index[:, test_data.edge_label == 1].to(device)
    test_neg_edge_index = test_data.edge_label_index[:, test_data.edge_label == 0].to(device)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    nodenum = data.x.size(0)
    adj = SparseTensor.from_edge_index(
        full_train_pos_edge_index,
        torch.ones(full_train_pos_edge_index.size(1), device=device),
        [nodenum, nodenum]
    )

    # ── W&B ───────────────────────────────────────────────────────────────
    run_name = (
        f"walk_agg={config['walk_agg']}"
        f"_W={config['num_walks']}_L={config['walk_length']}"
        f"_D={config['hidden_dim']}_R={config['recurrent_steps']}"
        f"_seed={config['seed']}"
    )
    wandb.init(
        entity=config['wb_entity'],
        project=config['wb_project'],
        group=f"{config['data_name']}_walk_agg_ablation",
        name=run_name,
        config=config,
        reinit=True,
    )
    run_id = wandb.run.id

    # ── Model ─────────────────────────────────────────────────────────────
    torch.manual_seed(config['seed'])
    mup_cfg = MupConfig(init_std=config['mup_init_std'],
                        mup_width_multiplier=config['mup_width_multiplier'])

    model = DPCWUE(
        num_nodes=data.num_nodes,
        emb_dim=config['emb_dim'],
        num_layers=config['num_layers'],
        hidden_dim=config['hidden_dim'],
        intermediate_dim=config['intermediate_dim'],
        num_heads=config['num_heads'],
        num_walks=config['num_walks'],
        walk_length=config['walk_length'],
        recurrent_steps=config['recurrent_steps'],
        walk_agg=config['walk_agg'],          # ← ablation flag
        attn_dropout_p=config['attn_dropout'],
        ffn_dropout_p=config['ffn_dropout'],
        resid_dropout_p=config['resid_dropout'],
        drop_path_p=config['drop_path'],
        config=mup_cfg,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters [{config['walk_agg']}]: {num_params:,}")
    wandb.log({'model/num_params': num_params}, step=0)

    link_predictor = None
    if config['use_mlp']:
        link_predictor = LinkPredictorMLP(
            in_dim=config['hidden_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['mlp_num_layers'],
            dropout=config['mlp_dropout'],
        ).to(device)

    # ── DataLoader ────────────────────────────────────────────────────────
    pos_edge_dataset = TensorDataset(full_train_pos_edge_index.t())
    train_loader = DataLoader(pos_edge_dataset, batch_size=config['global_batch_size'], shuffle=True)
    batches_per_epoch = math.ceil(full_train_pos_edge_index.size(1) / config['global_batch_size'])
    total_batches     = batches_per_epoch * config['num_epochs']
    warmup = total_batches // 10
    cool   = total_batches - warmup

    X = data.x.to(device).bfloat16()

    # ── Optimizer ─────────────────────────────────────────────────────────
    hidden_weights      = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    hidden_gains_biases = [p for p in model.parameters() if p.ndim <  2 and p.requires_grad]
    param_groups = [
        dict(params=hidden_weights,      use_muon=True,  lr=0.04,  weight_decay=0.01),
        dict(params=hidden_gains_biases, use_muon=False, lr=5e-5,  betas=(0.9,0.95), weight_decay=0.0),
    ]
    if config['use_mlp']:
        param_groups.append(
            dict(params=link_predictor.parameters(), use_muon=False,
                 lr=config['mlp_lr'], betas=(0.9, 0.95), weight_decay=0.0)
        )
    opt = SingleDeviceMuonWithAuxAdam(param_groups)

    # ── Training ──────────────────────────────────────────────────────────
    best_val_metric        = 0.0
    best_val_epoch         = 0
    epochs_without_improve = 0
    best_val_results       = {}
    best_test_results      = {}
    pbar = tqdm.tqdm(total=total_batches, desc=f"[{config['walk_agg']}]")

    for epoch in range(config['num_epochs']):
        model.train()
        if config['use_mlp']:
            link_predictor.train()

        for batch_idx, (batch_data,) in enumerate(train_loader):
            global_step = epoch * batches_per_epoch + batch_idx
            muon_lr = trapezoidal_lr_schedule(
                global_step, config['muon_max_lr'], config['muon_min_lr'],
                warmup, cool, total_batches
            )
            for pg in opt.param_groups:
                if pg.get('use_muon', False):
                    pg['lr'] = muon_lr

            batch_pos_edges = batch_data.t().to(device)
            local_bs = batch_pos_edges.size(1)
            batch_neg_edges = sample_negative_edges(
                full_train_pos_edge_index, train_data.num_nodes,
                int(local_bs * config['neg_sample_ratio']), device
            )

            all_nodes = torch.cat([
                batch_pos_edges[0], batch_pos_edges[1],
                batch_neg_edges[0], batch_neg_edges[1]
            ]).unique()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings = model(X, adj, all_nodes,
                                   p=config['node2vec_p'], q=config['node2vec_q'],
                                   track_level_sizes=(batch_idx == 0 and epoch == 0))
                n2i = {n.item(): i for i, n in enumerate(all_nodes)}
                pu = embeddings[[n2i[u.item()] for u in batch_pos_edges[0]]]
                pv = embeddings[[n2i[v.item()] for v in batch_pos_edges[1]]]
                nu = embeddings[[n2i[u.item()] for u in batch_neg_edges[0]]]
                nv = embeddings[[n2i[v.item()] for v in batch_neg_edges[1]]]

                if config['use_mlp']:
                    pos_scores = link_predictor(pu, pv)
                    neg_scores = link_predictor(nu, nv).view(-1, config['neg_sample_ratio'])
                else:
                    pos_scores = torch.sigmoid((pu * pv).sum(-1))
                    neg_scores = torch.sigmoid((nu * nv).sum(-1)).view(-1, config['neg_sample_ratio'])

                loss = binary_cross_entropy_loss(pos_scores, neg_scores)

            loss.backward()
            all_params = chain(model.parameters(), link_predictor.parameters()) \
                         if config['use_mlp'] else model.parameters()
            torch.nn.utils.clip_grad_norm_(all_params, config['grad_clip_norm'])
            opt.step()
            opt.zero_grad(set_to_none=True)

            wandb.log({'train/loss': loss.item(), 'train/lr': muon_lr}, step=global_step)
            pbar.set_description(f"[{config['walk_agg']}] loss={loss.item():.4f}")
            pbar.update(1)

        # ── Evaluation every 5 epochs ──────────────────────────────────────
        if (epoch == 0) or ((epoch + 1) % 5 == 0) or (epoch == config['num_epochs'] - 1):
            model.eval()
            val_results = evaluate_link_prediction_dp(
                model, link_predictor, val_pos_edge_index, val_neg_edge_index,
                adj, X, config, evaluator_hit, evaluator_mrr, device,
                eval_batch_size=config['global_batch_size']
            )
            test_results = evaluate_link_prediction_dp(
                model, link_predictor, test_pos_edge_index, test_neg_edge_index,
                adj, X, config, evaluator_hit, evaluator_mrr, device,
                eval_batch_size=config['global_batch_size']
            )
            global_step = (epoch + 1) * batches_per_epoch
            wandb.log({
                **{f'val/{k}':  v for k, v in val_results.items()},
                **{f'test/{k}': v for k, v in test_results.items()},
                'epoch': epoch + 1,
            }, step=global_step)

            vm = val_results[config['eval_metric']]
            print(f"\r  Epoch {epoch+1:4d}  val/{config['eval_metric']}={vm:.4f}", flush=True)

            if vm > best_val_metric:
                best_val_metric        = vm
                best_val_epoch         = epoch + 1
                best_val_results       = val_results.copy()
                best_test_results      = test_results.copy()
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= config['patience']:
                print(f"  Early stop at epoch {epoch+1}.")
                break

    pbar.close()

    print(f"\n  Best val {config['eval_metric']} = {best_val_metric:.4f}  (epoch {best_val_epoch})")
    wandb.summary[f'best_val_{config["eval_metric"]}'] = best_val_metric
    wandb.summary['best_val_epoch'] = best_val_epoch
    for k, v in best_test_results.items():
        wandb.summary[f'best_test_{k}'] = v

    wandb.finish()
    return {
        'walk_agg': config['walk_agg'],
        'best_val_epoch': best_val_epoch,
        'best_val': best_val_results,
        'best_test': best_test_results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config = get_config()

    # Run both strategies back-to-back with the same hyper-params
    all_results = {}
    for agg in ['flatten', 'concat']:
        cfg = dict(config)
        cfg['walk_agg'] = agg
        result = run_experiment(cfg)
        all_results[agg] = result

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ABLATION SUMMARY")
    print("="*60)
    metric = config['eval_metric']
    for agg, res in all_results.items():
        val_m  = res['best_val'].get(metric, float('nan'))
        test_m = res['best_test'].get(metric, float('nan'))
        print(f"  [{agg:>7s}]  val/{metric}={val_m:.4f}   test/{metric}={test_m:.4f}"
              f"   (epoch {res['best_val_epoch']})")
    print("="*60)


if __name__ == "__main__":
    main()