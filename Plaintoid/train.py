"""
heart_e2e.py
------------
Full end-to-end standalone training script for HeART (Heterogeneous Anonymized
Random-walk Transformer) for link prediction on graphs.

No external heart_model.py required — everything is self-contained.

Outputs
-------
  <data_name>/checkpoints/best_model_<run_id>.pth
  <data_name>/profiling/profiling_<run_id>.csv          (per-epoch memory + timing)
  <data_name>/profiling/profiling_summary.csv           (one row per run, appended)
  <data_name>/Transformer_<data_name>_LinkPred_<run_id>.csv
  <data_name>/experiment_results_link_prediction_updated_final.csv

Usage
-----
  python heart_e2e.py --data_name Cora --data_root data/Cora
  python heart_e2e.py --data_name CiteSeer --data_root data/CiteSeer
  python heart_e2e.py --data_name PubMed --data_root data/PubMed
"""

# ===========================================================================
# Imports
# ===========================================================================

import os
os.environ["WANDB_MODE"] = "offline"

import wandb
import psutil
import argparse
import pickle
import math
import time
import datetime
import dataclasses
from itertools import chain
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import tqdm.auto as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (
    to_undirected, coalesce, remove_self_loops,
    negative_sampling, degree
)
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data
from ogb.linkproppred import Evaluator
from einops import rearrange
from torch_cluster import random_walk as cluster_random_walk
from muon import SingleDeviceMuonWithAuxAdam

from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage,
])


# ===========================================================================
# μP Configuration
# ===========================================================================

@dataclasses.dataclass(frozen=True)
class MupConfig:
    init_std: float = 0.01
    mup_width_multiplier: float = 2.0


# ===========================================================================
# Rotary Positional Embeddings (RoPE)
# ===========================================================================

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


# ===========================================================================
# Transformer Layer
# ===========================================================================

class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        seq_len,
        n_layer,
        attn_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        drop_path_p: float = 0.0,
        config=MupConfig(),
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim      = hidden_dim
        self.num_heads       = num_heads
        self.attn_dropout_p  = attn_dropout_p
        self.ffn_dropout_p   = ffn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.drop_path_p     = drop_path_p

        # Attention
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o   = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # FFN (SwiGLU)
        self.up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        # Learnable RMSNorm scales
        self.attn_norm_weight  = nn.Parameter(torch.ones(hidden_dim))
        self.input_norm_weight = nn.Parameter(torch.ones(hidden_dim))

        # RoPE
        self.rope = RotaryPositionalEmbeddings(
            dim=(hidden_dim // num_heads), max_seq_len=seq_len
        )

        # μP initialization
        std_base  = config.init_std / math.sqrt(config.mup_width_multiplier)
        std_resid = config.init_std / math.sqrt(2 * n_layer * config.mup_width_multiplier)
        nn.init.normal_(self.up.weight,   mean=0.0, std=std_resid)
        nn.init.normal_(self.gate.weight, mean=0.0, std=std_resid)
        nn.init.normal_(self.down.weight, mean=0.0, std=std_base)
        nn.init.normal_(self.qkv.weight,  mean=0.0, std=std_base)
        nn.init.normal_(self.o.weight,    mean=0.0, std=std_base)

    def forward(self, x, offset=None):
        # Self-attention sublayer
        attnx = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.attn_norm_weight
        qkv   = self.qkv(attnx)
        q, k, v = qkv.chunk(3, dim=-1)

        q, k, v = [rearrange(t, 'n t (h d) -> n t h d', h=self.num_heads) for t in (q, k, v)]
        q = self.rope(q, input_pos=offset)
        k = self.rope(k, input_pos=offset)
        q, k, v = [rearrange(t, 'n t h d -> n h t d', h=self.num_heads) for t in (q, k, v)]

        o_walks = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=1.0 / k.shape[-1]
        )
        o_walks  = rearrange(o_walks, 'n h t d -> n t (h d)', h=self.num_heads)
        attn_out = self.o(o_walks)
        if self.resid_dropout_p > 0:
            attn_out = F.dropout(attn_out, p=self.resid_dropout_p, training=self.training)
        x = x + self._drop_path(attn_out)

        # FFN sublayer (SwiGLU)
        ffnx    = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.input_norm_weight
        ffn_out = self.down(F.silu(self.up(ffnx)) * self.gate(ffnx))
        if self.ffn_dropout_p > 0:
            ffn_out = F.dropout(ffn_out, p=self.ffn_dropout_p, training=self.training)
        x = x + self._drop_path(ffn_out)
        return x

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        """Stochastic depth (drop path) regularization."""
        if self.drop_path_p <= 0.0 or not self.training:
            return x
        keep_prob    = 1.0 - self.drop_path_p
        shape        = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * torch.floor(random_tensor)


# ===========================================================================
# Transformer Encoder
# ===========================================================================

class Transformer(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_heads,
        num_walks,
        seq_len,
        attn_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        drop_path_p: float = 0.0,
        config: MupConfig = MupConfig(),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.mup_cfg    = config

        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim, intermediate_dim, num_heads,
                num_walks * seq_len, n_layer=num_layers,
                attn_dropout_p=attn_dropout_p,
                ffn_dropout_p=ffn_dropout_p,
                resid_dropout_p=resid_dropout_p,
                drop_path_p=drop_path_p,
                config=config,
            )
            for _ in range(num_layers)
        ])

        self.emb         = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))

        nn.init.normal_(self.emb.weight, mean=0.0, std=config.init_std)

    def forward(self, x, anon_indices, source_nodes=None):
        """
        Args:
            x            : (B, ctx_len, emb_dim)  walk node features
            anon_indices : list of anonymized position tensors, one per recurrent step
        Returns:
            node_emb : (B, hidden_dim)  L2-normalised source node embeddings
        """
        batch_size, ctx_len, _ = x.shape
        x = self.emb(x)

        for depth, idx in enumerate(reversed(anon_indices)):
            for layer in self.layers:
                x = layer(x, idx)
            if depth < len(anon_indices) - 1:
                x = x[:, -1, :]
                x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
                x = rearrange(x, '(n t) z -> n t z', t=ctx_len)

        x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
        return F.normalize(x[:, -1, :], dim=-1)


# ===========================================================================
# MLP Link Predictor
# ===========================================================================

class LinkPredictorMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, num_layers=3, dropout=0.0):
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
        """Score a node pair via element-wise product + MLP."""
        x = h1 * h2
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(self.lins[-1](x))


# ===========================================================================
# Loss
# ===========================================================================

def binary_cross_entropy_loss(pos_scores, neg_scores):
    """
    BCE loss over positive and negative edge scores.
      pos_scores : (B,)
      neg_scores : (B, K)
    """
    pos_loss = -torch.log(pos_scores + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_scores + 1e-15).mean()
    return pos_loss + neg_loss


# ===========================================================================
# Random Walk Sampling & Anonymization
# ===========================================================================

@torch.no_grad()
def anonymize_rws(rws: torch.Tensor, rev_walks: bool = True) -> torch.Tensor:
    """
    Replace absolute node IDs with relative first-occurrence indices.
    Makes representations structure-only, ID-agnostic.

    Args:
        rws       : (N, walk_length)
        rev_walks : walks arrive in Target->Source order; flip before anonymizing
    """
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
    adj: SparseTensor,
    x: torch.Tensor,
    start_nodes: torch.Tensor,
    walk_length: int,
    num_walks: int,
    recurrent_steps: int = 1,
    p: float = 1.0,
    q: float = 1.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Sample Node2Vec walks from start_nodes, anonymize them, fetch node features.

    With recurrent_steps > 1, walk nodes from step k become sources for step k+1,
    expanding the receptive field hierarchically.

    Returns:
        batch_features    : (B, num_walks * walk_length, emb_dim)
        anon_indices_list : list of anonymized index tensors, one per recurrent step
    """
    row, col, _ = adj.coo()
    row = row.to(x.device)
    col = col.to(x.device)

    current_sources = start_nodes
    rws_list = []

    for _ in range(recurrent_steps):
        num_sources      = current_sources.size(0)
        sources_repeated = current_sources.repeat_interleave(num_walks)

        walks = cluster_random_walk(
            row, col, sources_repeated,
            walk_length - 1,
            p=p, q=q,
            num_nodes=adj.size(0)
        )

        walks = walks.view(num_sources, num_walks, walk_length)
        rws   = walks.flatten(1, 2)
        rws   = torch.flip(rws, dims=[-1])        # Source->Target to Target->Source
        rws_list.append(rws)

        if recurrent_steps > 1:
            current_sources = rws.reshape(-1)

    anon_indices_list = [anonymize_rws(rws, rev_walks=True) for rws in rws_list]
    batch_features    = x[rws_list[-1]]

    return batch_features, anon_indices_list


# ===========================================================================
# Profiling Helpers
# ===========================================================================

def get_cpu_ram_gib() -> float:
    """Current process RSS in GiB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def get_gpu_stats(device) -> dict:
    """GPU memory stats in GiB. Returns zeros if no GPU."""
    if not torch.cuda.is_available():
        return dict(vram_allocated_gib=0.0, vram_reserved_gib=0.0,
                    vram_peak_gib=0.0, vram_total_gib=0.0, vram_utilization_pct=0.0)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved  = torch.cuda.memory_reserved(device)  / (1024 ** 3)
    peak      = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    total     = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return dict(
        vram_allocated_gib    = round(allocated, 4),
        vram_reserved_gib     = round(reserved,  4),
        vram_peak_gib         = round(peak,      4),
        vram_total_gib        = round(total,      2),
        vram_utilization_pct  = round(allocated / total * 100, 2) if total > 0 else 0.0,
    )


def save_profiling_csv(epoch_rows: list, summary_row: dict, output_dir: str, run_id: str):
    """Write per-epoch profiling log and append to the run summary CSV."""
    os.makedirs(output_dir, exist_ok=True)

    per_epoch_path = os.path.join(output_dir, f"profiling_{run_id}.csv")
    pd.DataFrame(epoch_rows).to_csv(per_epoch_path, index=False)
    print(f"[Profiling] Per-epoch log  → {per_epoch_path}")

    summary_path = os.path.join(output_dir, "profiling_summary.csv")
    summary_df   = pd.DataFrame([summary_row])
    if os.path.exists(summary_path):
        summary_df.to_csv(summary_path, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(summary_path, mode='w', header=True, index=False)
    print(f"[Profiling] Run summary    → {summary_path}")


# ===========================================================================
# Argument Parsing
# ===========================================================================

def str_to_bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('true', '1', 't', 'yes', 'y'):
        return True
    if val.lower() in ('false', '0', 'f', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean expected, got: {val}')


def get_config():
    parser = argparse.ArgumentParser(description="HeART End-to-End Link Prediction")

    g = parser.add_argument_group('Run')
    g.add_argument('--seed',                       type=int,         default=2025)
    g.add_argument('--num_epochs',                 type=int,         default=300)
    g.add_argument('--global_batch_size',          type=int,         default=256)
    g.add_argument('--patience',                   type=int,         default=4)
    g.add_argument('--train_edge_downsample_ratio',type=float,       default=1.0)
    g.add_argument('--eval_metric',                type=str,         default='MRR')

    g = parser.add_argument_group('Data')
    g.add_argument('--data_root',                  type=str,         default='data/Cora')
    g.add_argument('--data_name',                  type=str,         default='Cora')
    g.add_argument('--val_split_ratio',            type=float,       default=0.15)
    g.add_argument('--test_split_ratio',           type=float,       default=0.05)
    g.add_argument('--use_fixed_splits',           type=str_to_bool, default=False)
    g.add_argument('--split_dir',                  type=str,         default='data/Cora/fixed_splits')
    g.add_argument('--use_laplacian_pe',           type=str_to_bool, default=False)
    g.add_argument('--laplacian_pe_path',          type=str,         default='data/Cora/laplacian_pe.pt')
    g.add_argument('--use_deepwalk_embeds',        type=str_to_bool, default=False)
    g.add_argument('--deepwalk_pkl_path',          type=str,         default=None)

    g = parser.add_argument_group('Model')
    g.add_argument('--num_layers',                 type=int,         default=1)
    g.add_argument('--hidden_dim',                 type=int,         default=128)
    g.add_argument('--intermediate_dim_multiplier',type=int,         default=4)
    g.add_argument('--num_heads',                  type=int,         default=16)
    g.add_argument('--recurrent_steps',            type=int,         default=1)
    g.add_argument('--mup_init_std',               type=float,       default=0.01)
    g.add_argument('--mup_width_multiplier',       type=float,       default=2.0)

    g = parser.add_argument_group('Random Walk')
    g.add_argument('--walk_length',                type=int,         default=8)
    g.add_argument('--num_walks',                  type=int,         default=16)
    g.add_argument('--node2vec_p',                 type=float,       default=1.0)
    g.add_argument('--node2vec_q',                 type=float,       default=1.0)

    g = parser.add_argument_group('Optimizer')
    g.add_argument('--muon_min_lr',                type=float,       default=1e-4)
    g.add_argument('--muon_max_lr',                type=float,       default=1e-3)
    g.add_argument('--adam_max_lr',                type=float,       default=1e-4)
    g.add_argument('--adam_min_lr',                type=float,       default=0.0)
    g.add_argument('--grad_clip_norm',             type=float,       default=0.1)

    g = parser.add_argument_group('MLP Predictor')
    g.add_argument('--use_mlp',                    type=str_to_bool, default=True)
    g.add_argument('--mlp_num_layers',             type=int,         default=3)
    g.add_argument('--mlp_lr',                     type=float,       default=1e-3)
    g.add_argument('--mlp_dropout',                type=float,       default=0.1)

    g = parser.add_argument_group('Regularization')
    g.add_argument('--attn_dropout',               type=float,       default=0.1)
    g.add_argument('--ffn_dropout',                type=float,       default=0.1)
    g.add_argument('--resid_dropout',              type=float,       default=0.1)
    g.add_argument('--drop_path',                  type=float,       default=0.05)

    g = parser.add_argument_group('Misc')
    g.add_argument('--neg_sample_ratio',           type=int,         default=1)
    g.add_argument('--hits_k',                     type=int, nargs='+', default=[1, 10, 50, 100])

    g = parser.add_argument_group('W&B')
    g.add_argument('--wb_entity',  type=str, default='graph-diffusion-model-link-prediction')
    g.add_argument('--wb_project', type=str, default='ani-cwue-link-prediction-final')

    args   = parser.parse_args()
    config = vars(args)
    config['intermediate_dim'] = config['hidden_dim'] * config['intermediate_dim_multiplier']
    del config['intermediate_dim_multiplier']
    return config


# ===========================================================================
# Data Utilities
# ===========================================================================

def load_graph_arxiv23(data_root) -> Data:
    return torch.load(data_root + 'arxiv_2023/graph.pt', weights_only=False)


def downsample_edges(edge_index, ratio=0.5, seed=42):
    torch.manual_seed(seed)
    perm = torch.randperm(edge_index.size(1))[: int(edge_index.size(1) * ratio)]
    return edge_index[:, perm]


def sample_negative_edges(pos_edge_index, num_nodes, num_neg_samples, device):
    return negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method='sparse'
    ).to(device)


# ===========================================================================
# LR Schedule
# ===========================================================================

def trapezoidal_lr_schedule(step, max_lr, min_lr, warmup, cool, total):
    if step <= warmup:
        return (step / warmup) * (max_lr - min_lr) + min_lr
    if step <= total - cool:
        return max_lr
    return ((total - step) / cool) * (max_lr - min_lr) + min_lr


# ===========================================================================
# Evaluation
# ===========================================================================

def get_metric_score(evaluator_hit, evaluator_mrr, pos_pred, neg_pred, k_list):
    result = {}
    hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
    for K in k_list:
        result[f'Hits@{K}'] = hits[f'Hits@{K}']
    result['MRR'] = evaluate_mrr(
        evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1)
    )['MRR']
    pred = torch.cat([pos_pred, neg_pred])
    true = torch.cat([torch.ones(pos_pred.size(0), dtype=int),
                      torch.zeros(neg_pred.size(0), dtype=int)])
    auc  = evaluate_auc(pred, true)
    result['AUC'] = auc['AUC']
    result['AP']  = auc['AP']
    return result


@torch.no_grad()
def score_edges(model, link_predictor, adj, X, config, edge_index, batch_size):
    """Score every edge pair in edge_index in batches; return flat score tensor."""
    input_data = edge_index.t()
    all_scores = []

    for perm in DataLoader(range(input_data.size(0)), batch_size=batch_size):
        batch_ei = input_data[perm].t()
        nodes    = batch_ei.unique()

        batch, anon_indices = get_random_walk_batch(
            adj, X, nodes,
            walk_length=config['walk_length'],
            num_walks=config['num_walks'],
            recurrent_steps=config['recurrent_steps'],
            p=config['node2vec_p'],
            q=config['node2vec_q'],
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb          = model(batch, anon_indices)
            n2i          = {n.item(): i for i, n in enumerate(nodes)}
            u            = emb[[n2i[n.item()] for n in batch_ei[0]]]
            v            = emb[[n2i[n.item()] for n in batch_ei[1]]]
            scores       = (link_predictor(u, v) if config['use_mlp']
                            else torch.sigmoid((u * v).sum(dim=-1)))
        all_scores.append(scores.cpu())

    return torch.cat(all_scores, dim=0).float()


@torch.no_grad()
def run_evaluation(model, link_predictor, pos_ei, neg_ei, adj, X, config,
                   evaluator_hit, evaluator_mrr, device):
    model.eval()
    if config['use_mlp']:
        link_predictor.eval()
    bs        = config['global_batch_size']
    pos_scores = score_edges(model, link_predictor, adj, X, config, pos_ei, bs)
    neg_scores = score_edges(model, link_predictor, adj, X, config, neg_ei, bs)
    return get_metric_score(
        evaluator_hit, evaluator_mrr,
        torch.flatten(pos_scores), torch.flatten(neg_scores),
        config.get('hits_k', [1, 10, 50, 100])
    )


@torch.no_grad()
def evaluate_and_log(
    model, link_predictor, adj, X, config,
    evaluator_hit, evaluator_mrr, device,
    train_pos_ei, train_neg_ei,
    val_pos_ei,   val_neg_ei,
    test_pos_ei,  test_neg_ei,
    epoch, best_val_metric, best_val_metrics, best_test_metrics,
    best_val_epoch, epochs_no_improve, BEST_MODEL_PATH,
):
    """Evaluate train/val/test, log to W&B, save best checkpoint, check early stop."""
    model.eval()
    if config['use_mlp']:
        link_predictor.eval()

    train_res = run_evaluation(model, link_predictor, train_pos_ei, train_neg_ei,
                               adj, X, config, evaluator_hit, evaluator_mrr, device)
    val_res   = run_evaluation(model, link_predictor, val_pos_ei,   val_neg_ei,
                               adj, X, config, evaluator_hit, evaluator_mrr, device)
    test_res  = run_evaluation(model, link_predictor, test_pos_ei,  test_neg_ei,
                               adj, X, config, evaluator_hit, evaluator_mrr, device)

    wandb.log({
        **{f'train_{k}': v for k, v in train_res.items()},
        **{f'val_{k}':   v for k, v in val_res.items()},
        **{f'test_{k}':  v for k, v in test_res.items()},
        'epoch': epoch + 1,
    })

    fmt = lambda d: ', '.join(f'{k}: {v:.4f}' for k, v in d.items())
    print(f"Epoch {epoch+1:>4}:  Train [{fmt(train_res)}]")
    print(f"            Val   [{fmt(val_res)}]")
    print(f"            Test  [{fmt(test_res)}]")

    for k, v in val_res.items():
        best_val_metrics[k] = max(v, best_val_metrics.get(k, 0.0))
    for k, v in test_res.items():
        best_test_metrics[k] = max(v, best_test_metrics.get(k, 0.0))

    cur_val     = val_res[config['eval_metric']]
    early_stop  = False

    if cur_val > best_val_metric:
        best_val_metric = cur_val
        best_val_epoch  = epoch + 1
        save_dict       = {'model_state_dict': model.state_dict(), 'config': config}
        if config['use_mlp']:
            save_dict['link_predictor_state_dict'] = link_predictor.state_dict()
        torch.save(save_dict, BEST_MODEL_PATH)
        print(f"  ✅ New best ({config['eval_metric']}: {cur_val:.4f}) → {BEST_MODEL_PATH}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= config['patience']:
        print(f"  Early stopping at epoch {epoch+1} "
              f"({config['patience']} evals without improvement).")
        early_stop = True

    return (best_val_metric, best_val_metrics, best_test_metrics,
            best_val_epoch, epochs_no_improve, early_stop)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # 0. Device & Config
    # -----------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    config = get_config()

    # -----------------------------------------------------------------------
    # 0a. Hardware snapshot (for profiling CSV)
    # -----------------------------------------------------------------------
    cpu_ram_start_gib = get_cpu_ram_gib()
    hw_info = {
        "device":             str(device),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores":  psutil.cpu_count(logical=True),
        "cpu_ram_total_gib":  round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_name":           torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "vram_total_gib":     round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
                              if torch.cuda.is_available() else 0.0,
    }
    print(f"\n{'='*55}\n  Hardware\n{'='*55}")
    for k, v in hw_info.items():
        print(f"  {k:<28}: {v}")
    print(f"{'='*55}\n")

    # -----------------------------------------------------------------------
    # 1. Data Loading
    # -----------------------------------------------------------------------
    if config['data_name'] in ['Cora', 'PubMed', 'CiteSeer']:
        dataset = Planetoid(root=config['data_root'], name=config['data_name'])
        data    = dataset[0].to(device)
    elif config['data_name'].startswith('TAPE'):
        data = load_graph_arxiv23(config['data_root']).to(device)
    else:
        raise ValueError(f"Unknown dataset: {config['data_name']}")

    # Optional: DeepWalk embeddings
    if config['use_deepwalk_embeds']:
        print(f"Loading DeepWalk embeddings: {config['deepwalk_pkl_path']}")
        if not os.path.exists(config['deepwalk_pkl_path']):
            raise FileNotFoundError(config['deepwalk_pkl_path'])
        with open(config['deepwalk_pkl_path'], 'rb') as f:
            dw = pickle.load(f)['data'].to(device)
        data.x = torch.cat([data.x, dw], dim=1)
        print(f"  DeepWalk concatenated → feature dim: {data.x.shape[1]}")

    # -----------------------------------------------------------------------
    # 2. Graph Preprocessing
    # -----------------------------------------------------------------------
    if data.is_directed():
        print("Directed graph detected → converting to undirected.")
        data.edge_index = to_undirected(data.edge_index)

    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
    print(f"\nGraph — {config['data_name']}:")
    print(f"  Nodes: {data.num_nodes}  Edges: {data.edge_index.size(1)}  "
          f"Avg deg: {deg.mean():.2f}  Isolated: {(deg == 0).sum().item()}")

    if config['data_name'] in ['TAPE', 'CiteSeer']:
        data = T.RemoveIsolatedNodes()(data)
        print(f"  Nodes after isolated removal: {data.num_nodes}")

    # Optional: Laplacian PE
    if config['use_laplacian_pe']:
        print(f"Loading Laplacian PEs: {config['laplacian_pe_path']}")
        if not os.path.exists(config['laplacian_pe_path']):
            raise FileNotFoundError(config['laplacian_pe_path'])
        lap_pe = torch.load(config['laplacian_pe_path'], map_location=device, weights_only=False)
        if lap_pe.size(0) != data.x.size(0):
            print(f"  PE size mismatch — truncating to {data.x.size(0)} nodes.")
            lap_pe = lap_pe[:data.x.size(0)]
        data.x = torch.cat([data.x, lap_pe], dim=1)
        print(f"  Laplacian PE concatenated → feature dim: {data.x.shape[1]}")

    config['emb_dim'] = data.x.size(1)
    print(f"  emb_dim: {config['emb_dim']}\n")

    # -----------------------------------------------------------------------
    # 3. Train / Val / Test Splits
    # -----------------------------------------------------------------------
    if config['use_fixed_splits']:
        split_path = os.path.join(config['split_dir'],
                                  f"{config['data_name']}_fixed_split.pt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(split_path)
        print(f"Loading fixed splits: {split_path}")
        sd = torch.load(split_path, map_location=device, weights_only=False)

        train_data = Data(x=data.x,
                          edge_index=sd['train']['edge_index'].to(device),
                          edge_label_index=sd['train']['edge_label_index'].to(device),
                          edge_label=sd['train']['edge_label'].to(device))
        val_data   = Data(x=data.x,
                          edge_index=sd['val']['edge_index'].to(device),
                          edge_label_index=sd['val']['edge_label_index'].to(device),
                          edge_label=sd['val']['edge_label'].to(device))
        test_data  = Data(x=data.x,
                          edge_index=sd['test']['edge_index'].to(device),
                          edge_label_index=sd['test']['edge_label_index'].to(device),
                          edge_label=sd['test']['edge_label'].to(device))

        full_train_pos_ei = train_data.edge_index
    else:
        print("Generating random splits...")
        transform = T.RandomLinkSplit(
            num_val=config['val_split_ratio'],
            num_test=config['test_split_ratio'],
            is_undirected=True,
            add_negative_train_samples=False,
        )
        train_data, val_data, test_data = transform(data)
        full_train_pos_ei = train_data.edge_index.to(device)

    val_pos_ei  = val_data.edge_label_index[:,  val_data.edge_label  == 1].to(device)
    val_neg_ei  = val_data.edge_label_index[:,  val_data.edge_label  == 0].to(device)
    test_pos_ei = test_data.edge_label_index[:, test_data.edge_label == 1].to(device)
    test_neg_ei = test_data.edge_label_index[:, test_data.edge_label == 0].to(device)

    print(f"Train edges: {full_train_pos_ei.size(1)}  "
          f"Val+: {val_pos_ei.size(1)} Val-: {val_neg_ei.size(1)}  "
          f"Test+: {test_pos_ei.size(1)} Test-: {test_neg_ei.size(1)}")

    # -----------------------------------------------------------------------
    # 4. Adjacency Matrix & Evaluators
    # -----------------------------------------------------------------------
    nodenum     = data.x.size(0)
    edge_weight = torch.ones(full_train_pos_ei.size(1), device=full_train_pos_ei.device)
    adj         = SparseTensor.from_edge_index(full_train_pos_ei, edge_weight,
                                               [nodenum, nodenum])

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    # -----------------------------------------------------------------------
    # 5. W&B Init
    # -----------------------------------------------------------------------
    enc_tag = ("deepwalk"  if config['use_deepwalk_embeds'] else
               "laplacian" if config['use_laplacian_pe']    else "none")
    run_name = (
        f"enc-{enc_tag}_rec{config['recurrent_steps']}_"
        f"bs{config['global_batch_size']}_"
        f"muon{config['muon_max_lr']}_adam{config['adam_max_lr']}_"
        f"nw{config['num_walks']}_wl{config['walk_length']}_"
        f"seed{config['seed']}"
    )
    if config['train_edge_downsample_ratio'] < 1.0:
        run_name += f"_dws{config['train_edge_downsample_ratio']}"

    wandb.init(
        entity=config['wb_entity'],
        project=f"{config['data_name']}_rw-cwue_{config['seed']}",
        group=f"{config['data_name']} HeART {'MLP' if config['use_mlp'] else 'DotProduct'}",
        name=run_name,
        config=config,
    )
    run_id = wandb.run.id

    # -----------------------------------------------------------------------
    # 6. Model
    # -----------------------------------------------------------------------
    torch.manual_seed(config['seed'])
    mup_cfg = MupConfig(init_std=config['mup_init_std'],
                        mup_width_multiplier=config['mup_width_multiplier'])

    model = Transformer(
        emb_dim=config['emb_dim'],
        num_layers=config['num_layers'],
        hidden_dim=config['hidden_dim'],
        intermediate_dim=config['intermediate_dim'],
        num_heads=config['num_heads'],
        num_walks=config['num_walks'],
        seq_len=config['walk_length'],
        attn_dropout_p=config['attn_dropout'],
        ffn_dropout_p=config['ffn_dropout'],
        resid_dropout_p=config['resid_dropout'],
        drop_path_p=config['drop_path'],
        config=mup_cfg,
    ).to(device)

    link_predictor = None
    if config['use_mlp']:
        link_predictor = LinkPredictorMLP(
            in_dim=config['hidden_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['mlp_num_layers'],
            dropout=config['mlp_dropout'],
        ).to(device)
        print("Link predictor: MLP")
    else:
        print("Link predictor: Dot product")

    n_params_enc = sum(p.numel() for p in model.parameters())
    n_params_mlp = sum(p.numel() for p in link_predictor.parameters()) if link_predictor else 0
    print(f"Parameters — Encoder: {n_params_enc:,}  MLP: {n_params_mlp:,}  "
          f"Total: {n_params_enc + n_params_mlp:,}")

    # -----------------------------------------------------------------------
    # 7. Checkpoint Path
    # -----------------------------------------------------------------------
    save_dir = os.path.join(config['data_name'], "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(save_dir, f"best_model_{run_id}.pth")

    # -----------------------------------------------------------------------
    # 8. DataLoader & Scheduler Setup
    # -----------------------------------------------------------------------
    gbs                   = config['global_batch_size']
    full_samples          = full_train_pos_ei.size(1)

    if config['train_edge_downsample_ratio'] < 1.0:
        samples_per_epoch = int(full_samples * config['train_edge_downsample_ratio'])
        train_loader      = None        # rebuilt each epoch
    else:
        samples_per_epoch = full_samples
        train_loader      = DataLoader(
            TensorDataset(full_train_pos_ei.t()),
            batch_size=gbs, shuffle=True
        )

    batches_per_epoch = math.ceil(samples_per_epoch / gbs)
    total_batches     = batches_per_epoch * config['num_epochs']
    warmup            = total_batches // 10
    cool              = total_batches - warmup

    print(f"\nBatch size: {gbs}  Batches/epoch: {batches_per_epoch}  "
          f"Total batches: {total_batches}")

    # -----------------------------------------------------------------------
    # 9. Optimizer
    # -----------------------------------------------------------------------
    X = train_data.x.to(device).bfloat16()

    w_matrix = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    w_scalar = [p for p in model.parameters() if p.ndim <  2 and p.requires_grad]
    if config['recurrent_steps'] <= 1:
        w_scalar = [p for p in w_scalar if p is not model.norm_weight]

    param_groups = [
        dict(params=w_matrix, use_muon=True,  lr=0.04, weight_decay=0.01),
        dict(params=w_scalar, use_muon=False, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.0),
    ]
    if config['use_mlp']:
        param_groups.append(
            dict(params=link_predictor.parameters(), use_muon=False,
                 lr=config['mlp_lr'], betas=(0.9, 0.95), weight_decay=0.0)
        )
    opt = SingleDeviceMuonWithAuxAdam(param_groups)

    # Gradient accumulation for high-recurrence runs
    accumulation_steps = (max(1, 256 // gbs)
                          if config['recurrent_steps'] > 1 else 1)
    if accumulation_steps > 1:
        print(f"Gradient accumulation: {accumulation_steps} steps "
              f"(effective batch size: {gbs * accumulation_steps})")

    # -----------------------------------------------------------------------
    # 10. Training State
    # -----------------------------------------------------------------------
    best_val_metric    = 0.0
    best_val_epoch     = 0
    epochs_no_improve  = 0
    eval_every         = 5
    best_val_metrics   = {}
    best_test_metrics  = {}

    # -----------------------------------------------------------------------
    # 11. Profiling State
    # -----------------------------------------------------------------------
    epoch_profile_rows: list = []
    batch_times_all:    list = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        print("Peak vRAM tracker RESET")

    cpu_ram_after_init_gib = get_cpu_ram_gib()
    run_start_time         = time.perf_counter()
    muon_lr                = config['muon_max_lr']

    # -----------------------------------------------------------------------
    # 12. Training Loop
    # -----------------------------------------------------------------------
    pbar         = tqdm.tqdm(total=total_batches, desc="Training")
    running_loss = 0.0

    for epoch in range(config['num_epochs']):

        model.train()
        if config['use_mlp']:
            link_predictor.train()

        epoch_start = time.perf_counter()

        # Per-epoch downsampled loader
        if config['train_edge_downsample_ratio'] < 1.0:
            epoch_pos_ei = downsample_edges(
                full_train_pos_ei,
                ratio=config['train_edge_downsample_ratio'],
                seed=config['seed'] + epoch,
            )
            train_loader = DataLoader(
                TensorDataset(epoch_pos_ei.t()),
                batch_size=gbs, shuffle=True
            )
        else:
            epoch_pos_ei = full_train_pos_ei

        # Subsample train edges for evaluation (HeART style)
        n_sample = min(val_neg_ei.size(1), epoch_pos_ei.size(1))
        torch.manual_seed(config['seed'] + epoch)
        perm              = torch.randperm(epoch_pos_ei.size(1), device=device)[:n_sample]
        eval_train_pos_ei = epoch_pos_ei[:, perm]

        epoch_batch_times = []

        for batch_idx, batch_data in enumerate(train_loader):
            global_step = batch_idx + epoch * batches_per_epoch

            # Update Muon LR
            for pg in opt.param_groups:
                if pg.get('use_muon', False):
                    muon_lr = trapezoidal_lr_schedule(
                        global_step,
                        config['muon_max_lr'], config['muon_min_lr'],
                        warmup, cool, total_batches
                    )
                    pg["lr"] = muon_lr

            # Positive edges
            batch_pos = batch_data[0].t().to(device)

            # Negative sampling
            n_neg     = int(batch_pos.size(1) * config['neg_sample_ratio'])
            batch_neg = sample_negative_edges(
                full_train_pos_ei, train_data.num_nodes, n_neg, device
            )

            # --- timed section ---
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            # Random walk batch
            all_nodes = torch.cat([batch_pos[0], batch_pos[1],
                                   batch_neg[0], batch_neg[1]]).unique()
            feats, anon_idx = get_random_walk_batch(
                adj, X, all_nodes,
                walk_length=config['walk_length'],
                num_walks=config['num_walks'],
                recurrent_steps=config['recurrent_steps'],
                p=config['node2vec_p'],
                q=config['node2vec_q'],
            )

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                emb     = model(feats, anon_idx)
                n2i     = {n.item(): i for i, n in enumerate(all_nodes)}
                pos_u   = emb[[n2i[n.item()] for n in batch_pos[0]]]
                pos_v   = emb[[n2i[n.item()] for n in batch_pos[1]]]
                neg_u   = emb[[n2i[n.item()] for n in batch_neg[0]]]
                neg_v   = emb[[n2i[n.item()] for n in batch_neg[1]]]

                if config['use_mlp']:
                    pos_scores = link_predictor(pos_u, pos_v)
                    neg_scores = link_predictor(neg_u, neg_v).view(
                        -1, int(config['neg_sample_ratio']))
                else:
                    pos_scores = torch.sigmoid((pos_u * pos_v).sum(dim=-1))
                    neg_scores = torch.sigmoid((neg_u * neg_v).sum(dim=-1)).view(
                        -1, int(config['neg_sample_ratio']))

                loss = binary_cross_entropy_loss(pos_scores, neg_scores)

            running_loss += loss.item()
            (loss / accumulation_steps).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or \
               (batch_idx + 1) == len(train_loader):

                # MLP gradient norm
                mlp_grad_norm = 0.0
                if config['use_mlp']:
                    with torch.no_grad():
                        for p in link_predictor.parameters():
                            if p.grad is not None:
                                mlp_grad_norm += p.grad.data.norm(2).item() ** 2
                        mlp_grad_norm = mlp_grad_norm ** 0.5

                # Gradient clip
                clip_params = (
                    chain(model.parameters(), link_predictor.parameters())
                    if config['use_mlp'] else model.parameters()
                )
                torch.nn.utils.clip_grad_norm_(clip_params, config['grad_clip_norm'])

                opt.step()
                opt.zero_grad(set_to_none=True)

                wandb.log(dict(loss=running_loss / accumulation_steps,
                               mlp_grad_norm=mlp_grad_norm,
                               step=global_step, lr=muon_lr))
                running_loss = 0.0

            # --- end timed section ---
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            batch_elapsed = time.perf_counter() - t0
            epoch_batch_times.append(batch_elapsed)
            batch_times_all.append(batch_elapsed)

            # Progress bar
            if torch.cuda.is_available():
                free, total_mem = torch.cuda.mem_get_info(device)
                mem_util = (total_mem - free) / total_mem
            else:
                mem_util = psutil.virtual_memory().percent / 100.0
            pbar.set_description(
                f"E{epoch+1} loss:{loss.item()*accumulation_steps:.4f} "
                f"mem:{mem_util:.2f} {batch_elapsed*1e3:.0f}ms/b"
            )
            pbar.update(1)

        # NaN guard
        if 'loss' in locals() and torch.isnan(loss):
            print("NaN loss — stopping.")
            break

        # --- Per-epoch profiling row ---
        epoch_wall = time.perf_counter() - epoch_start
        avg_bms    = (sum(epoch_batch_times) / len(epoch_batch_times)) * 1000 \
                     if epoch_batch_times else 0.0
        gpu_snap   = get_gpu_stats(device)
        cpu_now    = get_cpu_ram_gib()

        epoch_profile_rows.append({
            "run_id":               run_id,
            "data_name":            config['data_name'],
            "epoch":                epoch + 1,
            "epoch_wall_time_s":    round(epoch_wall, 3),
            "avg_batch_time_ms":    round(avg_bms, 3),
            "num_batches":          len(epoch_batch_times),
            "cpu_ram_process_gib":  round(cpu_now, 4),
            **{f"gpu_{k}": v for k, v in gpu_snap.items()},
        })

        # --- Evaluation ---
        is_eval = (epoch == 0 or
                   (epoch + 1) % eval_every == 0 or
                   epoch == config['num_epochs'] - 1)

        if is_eval:
            (best_val_metric, best_val_metrics, best_test_metrics,
             best_val_epoch, epochs_no_improve, early_stop) = evaluate_and_log(
                model=model, link_predictor=link_predictor,
                adj=adj, X=X, config=config,
                evaluator_hit=evaluator_hit, evaluator_mrr=evaluator_mrr, device=device,
                train_pos_ei=eval_train_pos_ei, train_neg_ei=val_neg_ei,
                val_pos_ei=val_pos_ei,          val_neg_ei=val_neg_ei,
                test_pos_ei=test_pos_ei,        test_neg_ei=test_neg_ei,
                epoch=epoch,
                best_val_metric=best_val_metric,
                best_val_metrics=best_val_metrics,
                best_test_metrics=best_test_metrics,
                best_val_epoch=best_val_epoch,
                epochs_no_improve=epochs_no_improve,
                BEST_MODEL_PATH=BEST_MODEL_PATH,
            )
            if early_stop:
                break

    pbar.close()

    # -----------------------------------------------------------------------
    # 13. Final Profiling Report
    # -----------------------------------------------------------------------
    total_wall_s      = time.perf_counter() - run_start_time
    epochs_trained    = len(epoch_profile_rows)
    avg_epoch_s       = total_wall_s / max(epochs_trained, 1)
    avg_batch_ms      = (sum(batch_times_all) / len(batch_times_all)) * 1000 \
                        if batch_times_all else 0.0
    final_gpu         = get_gpu_stats(device)
    final_cpu_gib     = get_cpu_ram_gib()
    peak_vram_gib     = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                         if torch.cuda.is_available() else 0.0)

    print(f"\n{'='*55}\n  PROFILING REPORT\n{'='*55}")
    print(f"  Total wall time       : {total_wall_s:.1f}s  ({total_wall_s/3600:.3f} hrs)")
    print(f"  Epochs trained        : {epochs_trained}")
    print(f"  Avg time / epoch      : {avg_epoch_s:.2f}s")
    print(f"  Avg time / batch      : {avg_batch_ms:.2f}ms")
    print(f"  GPU vRAM peak         : {peak_vram_gib:.4f} GiB")
    print(f"  GPU vRAM allocated    : {final_gpu['vram_allocated_gib']:.4f} GiB")
    print(f"  GPU vRAM reserved     : {final_gpu['vram_reserved_gib']:.4f} GiB")
    print(f"  CPU RAM (end)         : {final_cpu_gib:.4f} GiB")
    print(f"{'='*55}\n")

    # -----------------------------------------------------------------------
    # 14. Save Profiling CSVs
    # -----------------------------------------------------------------------
    profiling_summary = {
        # Identity
        "run_id":                       run_id,
        "timestamp":                    datetime.datetime.now().isoformat(timespec='seconds'),
        "data_name":                    config['data_name'],
        "seed":                         config['seed'],
        # Hardware
        **{f"hw_{k}": v for k, v in hw_info.items()},
        # Model size
        "params_encoder":               n_params_enc,
        "params_mlp":                   n_params_mlp,
        "params_total":                 n_params_enc + n_params_mlp,
        # Key hyperparams
        "hidden_dim":                   config['hidden_dim'],
        "num_layers":                   config['num_layers'],
        "num_heads":                    config['num_heads'],
        "walk_length":                  config['walk_length'],
        "num_walks":                    config['num_walks'],
        "recurrent_steps":              config['recurrent_steps'],
        "global_batch_size":            gbs,
        "accumulation_steps":           accumulation_steps,
        "effective_batch_size":         gbs * accumulation_steps,
        # Timing
        "epochs_trained":               epochs_trained,
        "total_wall_time_s":            round(total_wall_s, 3),
        "total_wall_time_hrs":          round(total_wall_s / 3600, 5),
        "avg_epoch_wall_time_s":        round(avg_epoch_s, 3),
        "avg_batch_wall_time_ms":       round(avg_batch_ms, 3),
        "total_batches_executed":       len(batch_times_all),
        # GPU memory
        "vram_peak_pytorch_gib":        round(peak_vram_gib, 4),
        "vram_allocated_end_gib":       final_gpu["vram_allocated_gib"],
        "vram_reserved_end_gib":        final_gpu["vram_reserved_gib"],
        "vram_total_gib":               final_gpu["vram_total_gib"],
        "vram_peak_utilization_pct":    round(peak_vram_gib / final_gpu["vram_total_gib"] * 100, 2)
                                        if final_gpu["vram_total_gib"] > 0 else 0.0,
        # CPU memory
        "cpu_ram_at_start_gib":         round(cpu_ram_start_gib, 4),
        "cpu_ram_after_model_init_gib": round(cpu_ram_after_init_gib, 4),
        "cpu_ram_at_end_gib":           round(final_cpu_gib, 4),
        "cpu_ram_delta_gib":            round(final_cpu_gib - cpu_ram_start_gib, 4),
        # Best metrics
        "best_val_epoch":               best_val_epoch,
        **{f"best_val_{k}":  v for k, v in best_val_metrics.items()},
        **{f"best_test_{k}": v for k, v in best_test_metrics.items()},
    }

    profiling_dir = os.path.join(config['data_name'], "profiling")
    save_profiling_csv(
        epoch_rows=epoch_profile_rows,
        summary_row=profiling_summary,
        output_dir=profiling_dir,
        run_id=run_id,
    )

    # -----------------------------------------------------------------------
    # 15. Results CSVs & W&B Final Log
    # -----------------------------------------------------------------------
    wandb.log({
        **{f'best_val_{k}':  v for k, v in best_val_metrics.items()},
        **{f'best_test_{k}': v for k, v in best_test_metrics.items()},
        'best_val_epoch': best_val_epoch,
    })

    print(f"Best Validation {config['eval_metric']}: {best_val_metric:.4f} "
          f"at epoch {best_val_epoch}")
    print(f"Best Val  [{', '.join(f'{k}: {v:.4f}' for k, v in best_val_metrics.items())}]")
    print(f"Best Test [{', '.join(f'{k}: {v:.4f}' for k, v in best_test_metrics.items())}]")

    # Per-run results CSV
    results = {
        'model_data_seed': f"Transformer_{config['data_name']}_LinkPred_{config['seed']}",
        'best_val_epoch':  best_val_epoch,
        **{f'best_val_{k}':  v for k, v in best_val_metrics.items()},
        **{f'best_test_{k}': v for k, v in best_test_metrics.items()},
    }

    # Experiment-level results CSV (appended across runs)
    experiment_results = {
        'seed':             config['seed'],
        'use_mlp':          config['use_mlp'],
        'global_bs':        gbs,
        'adam_max_lr':      config['adam_max_lr'],
        'muon_min_lr':      config['muon_min_lr'],
        'muon_max_lr':      config['muon_max_lr'],
        'mlp_lr':           config['mlp_lr'],
        'walk_length':      config['walk_length'],
        'num_walks':        config['num_walks'],
        'node2vec_p':       config['node2vec_p'],
        'node2vec_q':       config['node2vec_q'],
        'recurrent_steps':  config['recurrent_steps'],
        'mlp_num_layers':   config['mlp_num_layers'],
        'hidden_dim':       config['hidden_dim'],
        'attn_dropout':     config['attn_dropout'],
        'ffn_dropout':      config['ffn_dropout'],
        'resid_dropout':    config['resid_dropout'],
        'mlp_dropout':      config['mlp_dropout'],
        'drop_path':        config['drop_path'],
        'neg_sample_ratio': config['neg_sample_ratio'],
        'patience':         config['patience'],
        'best_val_epoch':   best_val_epoch,
        'grad_clip_norm':   config['grad_clip_norm'],
        **{f'metrics(best_test_{k})': v for k, v in best_test_metrics.items()},
        **{f'metrics(best_val_{k})':  v for k, v in best_val_metrics.items()},
    }

    os.makedirs(config['data_name'], exist_ok=True)
    run_csv = os.path.join(config['data_name'],
                           f"Transformer_{config['data_name']}_LinkPred_{run_id}.csv")
    pd.DataFrame([results]).to_csv(run_csv, index=False)

    summary_csv = os.path.join(config['data_name'],
                               "experiment_results_link_prediction_updated_final.csv")
    exp_df = pd.DataFrame([experiment_results])
    if os.path.exists(summary_csv):
        exp_df.to_csv(summary_csv, mode='a', header=False, index=False)
    else:
        exp_df.to_csv(summary_csv, mode='w', header=True, index=False)

    print(f"\nRun results → {run_csv}")
    print(f"Experiment summary → {summary_csv}")

    wandb.finish()