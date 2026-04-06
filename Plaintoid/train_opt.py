"""
CWUE Link Prediction with Dynamic Programming (DP) Recurrence

This implementation uses a bottom-up DP approach where we compute recurrence
step-by-step for all nodes, keeping memory bounded by O(N * W * L * D) instead
of O((W*L)^R).
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


def str_to_bool(val):
    """Helper function to parse boolean arguments from strings."""
    if isinstance(val, bool):
        return val
    if val.lower() in ('true', '1', 't', 'yes', 'y'):
        return True
    elif val.lower() in ('false', '0', 'f', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {val}')

def downsample_edges(edge_index, ratio=0.5, seed=42):
    torch.manual_seed(seed)
    num_edges = edge_index.size(1)
    sample_size = int(num_edges * ratio)
    perm = torch.randperm(num_edges)[:sample_size]
    return edge_index[:, perm]


def sample_negative_edges(pos_edge_index, num_nodes, num_neg_samples, device):
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method='sparse'
    ).to(device)
    return neg_edge_index


@torch.no_grad()
def sample_walks_for_nodes(
    adj: SparseTensor,
    node_indices: torch.Tensor,
    walk_length: int,
    num_walks: int,
    p: float = 1.0,
    q: float = 1.0
) -> torch.Tensor:
    """
    Sample random walks for given nodes.
    
    Args:
        adj: SparseTensor adjacency
        node_indices: [N] nodes to sample walks from
        walk_length: length of each walk
        num_walks: number of walks per node
        p, q: Node2Vec parameters
        
    Returns:
        walks: [N, num_walks * walk_length] node indices (flattened walks)
               Walks are reversed (target -> source) as required by CWUE
    """
    row, col, _ = adj.coo()
    row = row.to(node_indices.device)
    col = col.to(node_indices.device)
    
    # Repeat nodes for num_walks
    sources_repeated = node_indices.repeat_interleave(num_walks)
    
    # Sample walks
    walks = cluster_random_walk(
        row, col,
        sources_repeated,
        walk_length - 1,
        p=p,
        q=q,
        num_nodes=adj.size(0)
    )
    
    # Reshape to (num_nodes, num_walks, walk_length)
    num_nodes = node_indices.size(0)
    walks = walks.view(num_nodes, num_walks, walk_length)
    
    # Flatten last two dimensions: (num_nodes, num_walks * walk_length)
    walks = walks.flatten(1, 2)
    
    # Reverse walks (target -> source) as required by CWUE
    walks = torch.flip(walks, dims=[-1])
    
    return walks


@torch.no_grad()
def anonymize_walks(walks: torch.Tensor, rev_walks: bool = True) -> torch.Tensor:
    """
    Anonymize walk indices to be position-relative.
    
    Args:
        walks: [N, walk_len] node indices
        
    Returns:
        anon_indices: [N, walk_len] anonymized indices
    """
    # Match recurrence version: anonymize on source->target order, then flip back.
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


# MuP Config
@dataclasses.dataclass(frozen=True)
class MupConfig:
    init_std: float = 0.01
    mup_width_multiplier: float = 2.0


# Rotary Positional Embeddings
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
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_heads, seq_len, n_layer,
                 attn_dropout_p: float = 0.0, ffn_dropout_p: float = 0.0, resid_dropout_p: float = 0.0,
                 drop_path_p: float = 0.0, config=MupConfig()):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_dropout_p = attn_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.drop_path_p = drop_path_p
        self.up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.input_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.attn_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RotaryPositionalEmbeddings(dim=(hidden_dim // num_heads), max_seq_len=seq_len)

        torch.nn.init.normal_(self.up.weight.data, mean=0.0, std=config.init_std / math.sqrt(2 * n_layer * config.mup_width_multiplier))
        torch.nn.init.normal_(self.gate.weight.data, mean=0.0, std=config.init_std / math.sqrt(2 * n_layer * config.mup_width_multiplier))
        torch.nn.init.normal_(self.down.weight.data, mean=0.0, std=config.init_std / math.sqrt(config.mup_width_multiplier))
        torch.nn.init.normal_(self.qkv.weight.data, mean=0.0, std=config.init_std / math.sqrt(config.mup_width_multiplier))
        torch.nn.init.normal_(self.o.weight.data, mean=0.0, std=config.init_std / math.sqrt(config.mup_width_multiplier))

    def forward(self, x, offset=None):
        attnx = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.attn_norm_weight
        qkv = self.qkv(attnx)
        q, k, v = qkv.chunk(3, dim=-1)

        q, k, v = [rearrange(t, 'n t (h d) -> n t h d', h=self.num_heads)
                   for t in (q, k, v)]

        q = self.rope(q, input_pos=offset)
        k = self.rope(k, input_pos=offset)
        q, k, v = [rearrange(t, 'n t h d -> n h t d', h=self.num_heads)
                   for t in (q, k, v)]

        o_walks = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=1.0 / k.shape[-1]
        )
        o_walks = rearrange(o_walks, 'n h t d -> n t (h d)', h=self.num_heads)
        attn_out = self.o(o_walks)
        if self.resid_dropout_p > 0:
            attn_out = F.dropout(attn_out, p=self.resid_dropout_p, training=self.training)
        x = x + self._drop_path(attn_out)

        ffnx = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.input_norm_weight
        ffn_out = self.down(F.silu(self.up(ffnx)) * self.gate(ffnx))
        if self.ffn_dropout_p > 0:
            ffn_out = F.dropout(ffn_out, p=self.ffn_dropout_p, training=self.training)
        x = x + self._drop_path(ffn_out)
        return x

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path_p <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_path_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_tensor


class Transformer(nn.Module):
    """
    Standard Transformer for processing walks.
    This is used within each DP step.
    """
    def __init__(self, emb_dim, num_layers, hidden_dim, intermediate_dim, num_heads, num_walks, seq_len,
                 attn_dropout_p: float = 0.0, ffn_dropout_p: float = 0.0, resid_dropout_p: float = 0.0,
                 drop_path_p: float = 0.0, config: MupConfig = MupConfig()):
        super().__init__()
        self.mup_cfg = config
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim, intermediate_dim, num_heads, num_walks * seq_len, n_layer=num_layers,
                attn_dropout_p=attn_dropout_p, ffn_dropout_p=ffn_dropout_p, resid_dropout_p=resid_dropout_p,
                drop_path_p=drop_path_p, config=config
            ) for layer in range(num_layers)
        ])
        self.emb = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))

        torch.nn.init.normal_(self.emb.weight.data, mean=0.0, std=config.init_std)

    def forward(self, x, anon_indices, *, apply_emb: bool = True):
        """
        Args:
            x: [batch_size, walk_len, emb_dim] - walk features
            anon_indices: [batch_size, walk_len] - anonymized indices for RoPE
        Returns:
            [batch_size, hidden_dim] - representation of last token (target node)
        """
        batch_size, ctx_len, _ = x.shape
        if apply_emb:
            x = self.emb(x)
        for l in self.layers:
            x = l(x, anon_indices)
        x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
        x = F.normalize(x[:, -1, :], dim=-1)
        return x


class DPCWUE(nn.Module):
    """

    Instead of expanding walks exponentially, we compute recurrence step-by-step
    for all nodes, keeping memory bounded by O(N * W * L * D).
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
        attn_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        drop_path_p: float = 0.0,
        config: MupConfig = MupConfig()
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.recurrent_steps = recurrent_steps
        
        # Transformer processes walks at each recurrence step
        self.transformer = Transformer(
            emb_dim=hidden_dim,  # We feed in hidden_dim from previous step
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            num_walks=num_walks,
            seq_len=walk_length,
            attn_dropout_p=attn_dropout_p,
            ffn_dropout_p=ffn_dropout_p,
            resid_dropout_p=resid_dropout_p,
            drop_path_p=drop_path_p,
            config=config
        )

        self.transformer = torch.compile(self.transformer, dynamic=True)
        
        # Initial projection from input features to hidden dim (used once for all nodes)
        self.input_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        torch.nn.init.normal_(self.input_proj.weight.data, mean=0.0, std=config.init_std)

    def forward(
        self,
        node_features: torch.Tensor,
        adj: SparseTensor,
        batch_nodes: torch.Tensor,
        p: float = 1.0,
        q: float = 1.0,
        track_level_sizes: bool = False,
    ) -> torch.Tensor:
        """
        Compute representations for batch_nodes using DP recurrence.
        
        Args:
            node_features: [num_nodes, emb_dim] - input node features
            adj: SparseTensor - graph adjacency
            batch_nodes: [batch_size] - nodes to compute representations for
            p, q: Node2Vec parameters
            
        Returns:
            representations: [batch_size, hidden_dim] - final representations
        """
        device = node_features.device
        
        # Precompute walks for each recurrence level (top-down)
        level_nodes = [batch_nodes]
        walks_per_step = []
        for _ in range(self.recurrent_steps):
            walks = sample_walks_for_nodes(
                adj, level_nodes[-1], self.walk_length, self.num_walks, p, q
            )
            walks_per_step.append(walks)
            next_nodes = torch.unique(walks.view(-1))
            level_nodes.append(next_nodes)

        if track_level_sizes:
            # Keep for lightweight monitoring
            self.last_level_sizes = [n.numel() for n in level_nodes]

        # Initialize h for all nodes once (acts like the original Transformer emb)
        h_all = self.input_proj(node_features)  # [num_nodes, hidden_dim]

        # Bottom-up DP: compute representations from deepest level to batch nodes
        for step in reversed(range(self.recurrent_steps)):
            walks = walks_per_step[step]  # walks from level_nodes[step]
            walk_features = h_all[walks]
            anon_indices = anonymize_walks(walks, rev_walks=True)
            h_level = self.transformer(walk_features, anon_indices, apply_emb=False)
            h_all[level_nodes[step]] = h_level.to(h_all.dtype)

        # Return representations for original batch_nodes in order
        return h_all[batch_nodes]


class LinkPredictorMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, num_layers=3, dropout=0):
        super(LinkPredictorMLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_dim, out_dim))
        else:
            self.lins.append(torch.nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(torch.nn.Linear(hidden_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, h1, h2):
        x = h1 * h2
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def binary_cross_entropy_loss(pos_scores, neg_scores):
    pos_loss = -torch.log(pos_scores + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_scores + 1e-15).mean()
    loss = pos_loss + neg_loss
    return loss


@torch.no_grad()
def test_edge_dp(model, link_predictor, adj, node_features, config, edge_index, batch_size):
    """Test edges using DP CWUE."""
    input_data = edge_index.t()
    all_scores = []
    
    for perm in DataLoader(range(input_data.size(0)), batch_size=batch_size):
        batch_edge_index = input_data[perm].t()
        nodes = batch_edge_index.unique()
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings = model(
                node_features, adj, nodes,
                p=config['node2vec_p'],
                q=config['node2vec_q']
            )
            
            node_to_idx = {n.item(): i for i, n in enumerate(nodes)}
            u = embeddings[[node_to_idx[n.item()] for n in batch_edge_index[0]]]
            v = embeddings[[node_to_idx[n.item()] for n in batch_edge_index[1]]]
            
            if config['use_mlp']:
                scores = link_predictor(u, v)
            else:
                dot_product = (u * v).sum(dim=-1)
                scores = torch.sigmoid(dot_product)
        all_scores.append(scores.cpu())
    
    pred_all = torch.cat(all_scores, dim=0).float()
    return pred_all


@torch.no_grad()
def evaluate_link_prediction_dp(model, link_predictor, edge_index, neg_edge_index,
                                 adj, node_features, config, evaluator_hit, evaluator_mrr,
                                 device, eval_batch_size=512):
    model.eval()
    if config['use_mlp']:
        link_predictor.eval()
    
    pos_scores = test_edge_dp(model, link_predictor, adj, node_features, config,
                               edge_index, batch_size=eval_batch_size)
    neg_scores = test_edge_dp(model, link_predictor, adj, node_features, config,
                               neg_edge_index, batch_size=eval_batch_size)
    
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_scores), torch.flatten(pos_scores)
    k_list = config.get('hits_k', [1, 10, 50, 100])
    
    result = get_metric_score(
        evaluator_hit=evaluator_hit,
        evaluator_mrr=evaluator_mrr,
        pos_val_pred=pos_valid_pred,
        neg_val_pred=neg_valid_pred,
        k_list=k_list
    )
    return result


def get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, k_list: list):
    result = {}
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    for K in k_list:
        result[f'Hits@{K}'] = result_hit_val[f'Hits@{K}']
    
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1))
    result['MRR'] = result_mrr_val['MRR']
    
    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([
        torch.ones(pos_val_pred.size(0), dtype=int),
        torch.zeros(neg_val_pred.size(0), dtype=int)
    ])
    
    result_auc_val = evaluate_auc(val_pred, val_true)
    result['AUC'] = result_auc_val['AUC']
    result['AP'] = result_auc_val['AP']
    
    return result


@torch.no_grad()
def evaluate_and_log_dp(
    model, link_predictor, adj, node_features, config,
    evaluator_hit, evaluator_mrr, device,
    train_pos_edge_index, train_neg_edge_index,
    val_pos_edge_index, val_neg_edge_index,
    test_pos_edge_index, test_neg_edge_index,
    epoch, best_val_eval_metric, best_val_metrics,
    best_test_metrics, best_val_epoch,
    epochs_without_improvement, BEST_MODEL_PATH
):
    model.eval()
    if config['use_mlp']:
        link_predictor.eval()
    
    train_results = evaluate_link_prediction_dp(
        model, link_predictor,
        edge_index=train_pos_edge_index,
        neg_edge_index=train_neg_edge_index,
        adj=adj, node_features=node_features, config=config,
        evaluator_hit=evaluator_hit, evaluator_mrr=evaluator_mrr,
        eval_batch_size=config['global_batch_size'], device=device
    )
    val_results = evaluate_link_prediction_dp(
        model, link_predictor, val_pos_edge_index, val_neg_edge_index,
        adj, node_features, config, evaluator_hit=evaluator_hit,
        evaluator_mrr=evaluator_mrr, eval_batch_size=config['global_batch_size'], device=device
    )
    test_results = evaluate_link_prediction_dp(
        model, link_predictor, test_pos_edge_index, test_neg_edge_index,
        adj, node_features, config, evaluator_hit=evaluator_hit,
        evaluator_mrr=evaluator_mrr, eval_batch_size=config['global_batch_size'], device=device
    )
        
    wandb.log({
        **{f'train/{k}': v for k, v in train_results.items()},
        **{f'val/{k}': v for k, v in val_results.items()},
        **{f'test/{k}': v for k, v in test_results.items()},
        'epoch': epoch + 1,
    })
        
    train_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_results.items()])
    val_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_results.items()])
    test_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_results.items()])
    print(f"\rEpoch {epoch + 1}: Train [{train_metrics_str}] | Val [{val_metrics_str}] | Test [{test_metrics_str}]", end="", flush=True)
    
    val_metric = val_results[config['eval_metric']]
    early_stop_flag = False
    
    if val_metric > best_val_eval_metric:
        best_val_eval_metric = val_metric
        best_val_epoch = epoch + 1
        
        # if wandb.run is not None:
        #     wandb.run.summary['best_val_epoch'] = best_val_epoch
        
        # save_dict = {'model_state_dict': model.state_dict(), 'config': config}
        # if config['use_mlp']:
        #     save_dict['link_predictor_state_dict'] = link_predictor.state_dict()
        # torch.save(save_dict, BEST_MODEL_PATH)
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= config['patience']:
        print(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {config['patience']} epochs)", flush=True)
        early_stop_flag = True
    
    return (
        best_val_eval_metric, best_val_metrics, best_test_metrics,
        best_val_epoch, epochs_without_improvement, early_stop_flag
    )


def trapezoidal_lr_schedule(global_batch_idx, max_lr, min_lr, warmup, cool, total_batches):
    world_size = 1
    if global_batch_idx <= warmup:
        lr = (global_batch_idx / warmup) * (max_lr - min_lr) + min_lr
    elif warmup < global_batch_idx <= (total_batches - cool):
        lr = max_lr
    else:
        lr_scale = ((total_batches - global_batch_idx) / cool)
        lr = lr_scale * (max_lr - min_lr) + min_lr
    return lr * (world_size ** 1)



def get_config():
    parser = argparse.ArgumentParser(description="CWUE Link Prediction with DP Recurrence")
    
    run_group = parser.add_argument_group('Run Configuration')
    run_group.add_argument('--seed', type=int, default=2025, help='Random seed')
    run_group.add_argument('--num_epochs', type=int, default=3000, help='Number of training epochs')
    run_group.add_argument('--global_batch_size', type=int, default=256, help='Batch size for positive edges')
    run_group.add_argument('--patience', type=int, default=100, help='Epochs for early stopping patience')
    run_group.add_argument('--train_edge_downsample_ratio', type=float, default=1.0,
                           help='Ratio to downsample training edges')
    run_group.add_argument('--eval_metric', type=str, default='MRR',
                           help='Evaluation metric for choosing the best model')
    
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data_root', type=str, default="data/Cora", help='Root directory for the dataset')
    data_group.add_argument('--data_name', type=str, default="Cora", help='Dataset Name')
    data_group.add_argument('--val_split_ratio', type=float, default=0.15, help='Split ratio for Validation data')
    data_group.add_argument('--test_split_ratio', type=float, default=0.05, help='Split ratio for Test data')
    data_group.add_argument('--deepwalk_pkl_path', type=str, default=None,
                            help='Path to a .pkl file containing custom DeepWalk embeddings')
    data_group.add_argument('--use_fixed_splits', type=str_to_bool, default=False,
                            help='If True, loads splits from file instead of random sampling')
    data_group.add_argument('--split_dir', type=str, default="data/Cora/fixed_splits",
                            help='Directory containing the fixed_split.pt file')
    data_group.add_argument('--use_laplacian_pe', type=str_to_bool, default=False,
                            help='Whether to concatenate Laplacian PEs to node features')
    data_group.add_argument('--laplacian_pe_path', type=str, default="data/Cora/laplacian_pe.pt",
                            help='Path to the pre-generated Laplacian PE file')
    data_group.add_argument('--laplacian_edge_subsampling_ratio', type=float, default=1.0,
                            help='Edges subsampled for Laplacian PE calculation')
    
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--num_layers', type=int, default=1, help='Number of Transformer layers')
    model_group.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the model')
    model_group.add_argument('--intermediate_dim_multiplier', type=int, default=4,
                             help='Multiplier for intermediate FFN dim')
    model_group.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    model_group.add_argument('--recurrent_steps', type=int, default=2, help='Number of recurrent steps for CWUE')
    model_group.add_argument('--mup_init_std', type=float, default=0.01, help='MuP initialization standard deviation')
    model_group.add_argument('--mup_width_multiplier', type=float, default=2.0, help='MuP width multiplier')
    
    walk_group = parser.add_argument_group('Random Walk')
    walk_group.add_argument('--walk_length', type=int, default=8, help='Length of each random walk')
    walk_group.add_argument('--num_walks', type=int, default=16, help='Number of walks per source node')
    walk_group.add_argument('--node2vec_p', type=float, default=1.0, help='Return parameter p (Node2Vec)')
    walk_group.add_argument('--node2vec_q', type=float, default=1.0, help='In-out parameter q (Node2Vec)')
    
    optim_group = parser.add_argument_group('Optimizer')
    optim_group.add_argument('--muon_min_lr', type=float, default=1e-4, help='Muon min learning rate')
    optim_group.add_argument('--muon_max_lr', type=float, default=1e-3, help='Muon max learning rate')
    optim_group.add_argument('--adam_max_lr', type=float, default=1e-4, help='Adam max learning rate')
    optim_group.add_argument('--adam_min_lr', type=float, default=0.0, help='Adam min learning rate')
    optim_group.add_argument('--grad_clip_norm', type=float, default=0.1, help='Gradient clipping norm')
    
    mlp_group = parser.add_argument_group('MLP Link Predictor')
    mlp_group.add_argument('--use_mlp', type=str_to_bool, default=True, help='Use MLP link predictor')
    mlp_group.add_argument('--mlp_num_layers', type=int, default=3, help='Number of layers in the MLP predictor')
    mlp_group.add_argument('--mlp_lr', type=float, default=1e-3, help='Learning rate for the MLP predictor')
    mlp_group.add_argument('--mlp_dropout', type=float, default=0.1, help='Dropout rate for the MLP predictor')
    
    reg_group = parser.add_argument_group('Regularization')
    reg_group.add_argument('--attn_dropout', type=float, default=0.1, help='Attention dropout')
    reg_group.add_argument('--ffn_dropout', type=float, default=0.1, help='FFN dropout')
    reg_group.add_argument('--resid_dropout', type=float, default=0.1, help='Residual dropout')
    reg_group.add_argument('--drop_path', type=float, default=0.05, help='Stochastic depth drop path rate')
    
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--neg_sample_ratio', type=int, default=1,
                            help='Ratio of negative to positive samples per batch')
    misc_group.add_argument('--hits_k', type=int, nargs='+', default=[1, 10, 50, 100],
                            help='List of K values for Hits@K metrics')
    misc_group.add_argument('--use_deepwalk_embeds', type=str_to_bool, default=False,
                            help='Should use Deepwalk embeddings for node features enhancement')
    
    logging_group = parser.add_argument_group('Weights & Biases Logging')
    logging_group.add_argument('--wb_entity', type=str, default="cshao676",
                               help='Name of the main directory where all different Weights & Biases Projects are present')
    logging_group.add_argument('--wb_project', type=str, default="hrw_plaintoid",
                               help='Name of the project where all the logging is done')
    
    args = parser.parse_args()
    config = vars(args)
    config['intermediate_dim'] = config['hidden_dim'] * config['intermediate_dim_multiplier']
    del config['intermediate_dim_multiplier']
    return config


def load_graph_arxiv23(data_root) -> Data:
    data = torch.load(data_root + 'arxiv_2023/graph.pt', weights_only=False)
    return data


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = get_config()

    # --- DATA LOADING & PREPROCESSING ---
    if config['data_name'] in ['Cora', 'PubMed', 'CiteSeer']:
        dataset = Planetoid(root=config['data_root'], name=config['data_name'])
        data = dataset[0].to(device)
    elif config['data_name'].startswith('TAPE'):
        data = load_graph_arxiv23(data_root=config['data_root'])
        data = data.to(device)

    # Load DeepWalk embeddings if specified
    if config['use_deepwalk_embeds']:
        print(f"Loading custom DeepWalk embeddings from: {config['deepwalk_pkl_path']}")
        if not os.path.exists(config['deepwalk_pkl_path']):
            raise FileNotFoundError(f"DeepWalk PKL file not found at: {config['deepwalk_pkl_path']}")
        with open(config['deepwalk_pkl_path'], 'rb') as f:
            saved_data = pickle.load(f)
            deepwalk_embeddings = saved_data['data'].to(device)
        print(f"Successfully loaded DeepWalk embeddings")
        original_features = data.x
        combined_features = torch.cat([original_features, deepwalk_embeddings], dim=1)
        data.x = combined_features
        print(f"Features concatenated. New feature dimension: {data.x.shape[1]}")

    # Data preprocessing
    undirected = None
    if data.is_directed():
        print("Graph is directed. Making it undirected for better connectivity.......")
        data.edge_index = to_undirected(data.edge_index)
        undirected = True

    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
    num_isolated_nodes = (deg == 0).sum().item()
    print("\n" + "="*40)
    print(f"Graph Statistics for {config['data_name']}:")
    print(f"  Num Nodes:      {data.num_nodes}")
    print(f"  Num Edges:      {data.edge_index.size(1)}")
    print(f"  Average Degree: {deg.mean().item():.4f}")
    print(f"  Max Degree:     {deg.max().item()}")
    print(f"  Min Degree:     {deg.min().item()}")
    print(f"  Isolated Nodes: {num_isolated_nodes}")
    print("="*40 + "\n")
    print(f"Nodes before removal: {data.num_nodes}")

    if config['data_name'] in ['TAPE', 'CiteSeer']:
        remover = T.RemoveIsolatedNodes()
        data = remover(data)
    print(f"Nodes after removal:  {data.num_nodes}")

    # Load Laplacian PEs if specified
    if config['use_laplacian_pe']:
        print(f"Loading Laplacian PEs from: {config['laplacian_pe_path']}")
        if not os.path.exists(config['laplacian_pe_path']):
            raise FileNotFoundError(f"Laplacian PE file not found at: {config['laplacian_pe_path']}")
        lap_pe = torch.load(config['laplacian_pe_path'], map_location=device, weights_only=False)
        if lap_pe.size(0) != data.x.size(0):
            print(f"WARNING: PE nodes ({lap_pe.size(0)}) != Graph nodes ({data.x.size(0)}).")
            lap_pe = lap_pe[:data.x.size(0)]
        data.x = torch.cat([data.x, lap_pe], dim=1)
        print(f"Laplacian PEs concatenated. New feature dimension: {data.x.shape[1]}")

    config['emb_dim'] = data.x.size(1)
    print(f"Inferred node feature dimension (emb_dim): {config['emb_dim']}")

    # Load or generate splits
    if config['use_fixed_splits']:
        split_path = os.path.join(config['split_dir'], f"{config['data_name']}_fixed_split.pt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Fixed split file not found at {split_path}")
        print(f"Loading fixed splits from: {split_path}")
        split_data = torch.load(split_path, map_location=device, weights_only=False)
        
        train_data = Data(
            x=data.x,
            edge_index=split_data['train']['edge_index'].to(device),
            edge_label_index=split_data['train']['edge_label_index'].to(device),
            edge_label=split_data['train']['edge_label'].to(device)
        )
        val_data = Data(
            x=data.x,
            edge_index=split_data['val']['edge_index'].to(device),
            edge_label_index=split_data['val']['edge_label_index'].to(device),
            edge_label=split_data['val']['edge_label'].to(device)
        )
        test_data = Data(
            x=data.x,
            edge_index=split_data['test']['edge_index'].to(device),
            edge_label_index=split_data['test']['edge_label_index'].to(device),
            edge_label=split_data['test']['edge_label'].to(device)
        )
        full_train_pos_edge_index = train_data.edge_index
        val_pos_edge_index = val_data.edge_label_index[:, val_data.edge_label == 1]
        val_neg_edge_index = val_data.edge_label_index[:, val_data.edge_label == 0]
        test_pos_edge_index = test_data.edge_label_index[:, test_data.edge_label == 1]
        test_neg_edge_index = test_data.edge_label_index[:, test_data.edge_label == 0]
        print(f"Loaded fixed split. Train edges: {full_train_pos_edge_index.size(1)}")
    else:
        print("Generating random splits on the fly...")
        transform = T.RandomLinkSplit(
            num_val=config['val_split_ratio'],
            num_test=config['test_split_ratio'],
            is_undirected=True,
            add_negative_train_samples=False
        )
        train_data, val_data, test_data = transform(data)
        full_train_pos_edge_index = train_data.edge_index.to(device)
        val_pos_edge_index = val_data.edge_label_index[:, val_data.edge_label == 1].to(device)
        val_neg_edge_index = val_data.edge_label_index[:, val_data.edge_label == 0].to(device)
        test_pos_edge_index = test_data.edge_label_index[:, test_data.edge_label == 1].to(device)
        test_neg_edge_index = test_data.edge_label_index[:, test_data.edge_label == 0].to(device)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    nodenum = data.x.size(0)

    edge_weight = torch.ones(full_train_pos_edge_index.size(1), device=full_train_pos_edge_index.device)
    adj = SparseTensor.from_edge_index(full_train_pos_edge_index, edge_weight, [nodenum, nodenum])

    # Initialize wandb
    if config['use_deepwalk_embeds']:
        run_display_name = f"dp_pos_encoding_deepwalk_recurrent_steps_{config['recurrent_steps']}_bs_{config['global_batch_size']}_muon-max-lr_{config['muon_max_lr']}_adam_max_lr_{config['adam_max_lr']}_nwalks_{config['num_walks']}_wl_{config['walk_length']}_seed_{config['seed']}"
    elif config['use_laplacian_pe']:
        run_display_name = f"dp_pos_encoding_laplacian_recurrent_steps_{config['recurrent_steps']}_bs_{config['global_batch_size']}_muon-max-lr_{config['muon_max_lr']}_adam_max_lr_{config['adam_max_lr']}_nwalks_{config['num_walks']}_wl_{config['walk_length']}_seed_{config['seed']}"
    else:
        run_display_name = f"dp_pos_encoding_False_recurrent_steps_{config['recurrent_steps']}_bs_{config['global_batch_size']}_muon-max-lr_{config['muon_max_lr']}_adam_max_lr_{config['adam_max_lr']}_nwalks_{config['num_walks']}_wl_{config['walk_length']}_seed_{config['seed']}"

    if config["train_edge_downsample_ratio"] < 1.0:
        run_display_name = f"{run_display_name}_edge_dws_{config['train_edge_downsample_ratio']}"

    if config['use_mlp']:
        wandb.init(
            entity=config['wb_entity'],
            project=f"{config['data_name']}_dp_rw-cwue_latest_dec_{config['seed']}",
            group=f"{config['data_name']} DP Random Walk Link Prediction MLP",
            name=run_display_name, config=config
        )
    else:
        wandb.init(
            entity=config['wb_entity'],
            project=f"{config['data_name']}_dp_rw-cwue_latest_dec_{config['seed']}",
            group=f"{config['data_name']} DP Random Walk Link Prediction Dot Product",
            name=run_display_name, config=config
        )
    run_id = wandb.run.id
    save_dir = f"{config['data_name']}/checkpoints_dp"

    os.makedirs(save_dir, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(save_dir, f"best_model_{run_id}.pth")

    # Initialize model and optimizer
    torch.manual_seed(config['seed'])
    mup_config = MupConfig(
        init_std=config['mup_init_std'],
        mup_width_multiplier=config['mup_width_multiplier']
    )

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
        attn_dropout_p=config['attn_dropout'],
        ffn_dropout_p=config['ffn_dropout'],
        resid_dropout_p=config['resid_dropout'],
        drop_path_p=config['drop_path'],
        config=mup_config
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    link_predictor = None
    if config['use_mlp']:
        print("Using MLP for link prediction.")
        link_predictor = LinkPredictorMLP(
            in_dim=config['hidden_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['mlp_num_layers'],
            dropout=config['mlp_dropout']
        ).to(device)
    else:
        print("Using dot product for link prediction.")

    # Setup data loaders
    global_batch_size = config['global_batch_size']
    full_samples_per_epoch = full_train_pos_edge_index.size(1)

    if config['train_edge_downsample_ratio'] < 1.0:
        samples_per_epoch = int(full_samples_per_epoch * config['train_edge_downsample_ratio'])
        print(f"Training with downsampling: {samples_per_epoch} edges per epoch (from {full_samples_per_epoch}).")
        train_loader = None
    else:
        samples_per_epoch = full_samples_per_epoch
        print(f"Training with all {samples_per_epoch} positive edges per epoch.")
        pos_edge_dataset = TensorDataset(full_train_pos_edge_index.t())
        train_loader = DataLoader(
            pos_edge_dataset,
            batch_size=global_batch_size,
            shuffle=True
        )

    batches_per_epoch = math.ceil(samples_per_epoch / global_batch_size)
    total_batches = batches_per_epoch * config['num_epochs']

    print(f"Batch size: {global_batch_size}")
    print(f"Batches per epoch (actual): {batches_per_epoch}")
    print(f"Total batches (epochs * steps): {total_batches}")

    warmup = total_batches // 10
    cool = (total_batches - warmup)

    X = data.x.to(device).bfloat16()

    # Setup optimizer
    hidden_weights = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]

    # Don't include norm_weight if recurrent_steps <= 1 (original logic)
    if config['recurrent_steps'] <= 1:
        hidden_gains_biases = [p for p in hidden_gains_biases if p is not model.transformer.norm_weight]

    param_groups = [
        dict(params=hidden_weights, use_muon=True, lr=0.04, weight_decay=0.01),
        dict(params=hidden_gains_biases, use_muon=False, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.0),
    ]

    if config['use_mlp']:
        param_groups.append(
            dict(params=link_predictor.parameters(), use_muon=False, lr=config['mlp_lr'], betas=(0.9, 0.95), weight_decay=0.0)
        )

    opt = SingleDeviceMuonWithAuxAdam(param_groups)

    best_val_eval_metric = 0.0
    best_val_epoch = 0
    epochs_without_improvement = 0
    epochs_eval_steps = 5
    best_val_metrics = {}
    best_test_metrics = {}

    pbar = tqdm.tqdm(total=total_batches)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print("Peak vRAM tracker RESET")

    running_loss = 0.0

    for epoch in range(config['num_epochs']):
        model.train()
        if config['use_mlp']:
            link_predictor.train()
        
        # Downsampling edges if applicable
        if config['train_edge_downsample_ratio'] < 1.0:
            epoch_train_pos_edge_index = downsample_edges(
                full_train_pos_edge_index,
                ratio=config['train_edge_downsample_ratio'],
                seed=config['seed'] + epoch
            )
            pos_edge_dataset = TensorDataset(epoch_train_pos_edge_index.t())
            train_loader = DataLoader(
                pos_edge_dataset,
                batch_size=global_batch_size,
                shuffle=True
            )
        else:
            epoch_train_pos_edge_index = full_train_pos_edge_index
        
        # Sample training edges for evaluation
        num_val_neg = val_neg_edge_index.size(1)
        num_train_pos_to_sample = min(num_val_neg, epoch_train_pos_edge_index.size(1))
        torch.manual_seed(config['seed'] + epoch)
        pos_perm = torch.randperm(epoch_train_pos_edge_index.size(1), device=device)[:num_train_pos_to_sample]
        eval_train_pos_edge_index = epoch_train_pos_edge_index[:, pos_perm]
        
        # Training loop
        for batch_idx, batch_data in enumerate(train_loader):
            global_batch_idx = batch_idx + epoch * batches_per_epoch
            
            # Update learning rate
            for p in opt.param_groups:
                if p.get('use_muon', False):
                    muon_lr = trapezoidal_lr_schedule(
                        global_batch_idx, config['muon_max_lr'],
                        config['muon_min_lr'], warmup, cool, total_batches
                    )
                    p["lr"] = muon_lr
            
            # Get positive edges
            batch_pos_edges = batch_data[0].t().to(device)
            
            # Sample negative edges
            local_batch_size = batch_pos_edges.size(1)
            num_neg_samples = int(local_batch_size * config['neg_sample_ratio'])
            batch_neg_edges = sample_negative_edges(
                pos_edge_index=full_train_pos_edge_index,
                num_nodes=train_data.num_nodes,
                num_neg_samples=num_neg_samples,
                device=device
            )
            
            start_time_batch = time.time()
            
            # Get unique nodes needed
            all_nodes = torch.cat([batch_pos_edges[0], batch_pos_edges[1], batch_neg_edges[0], batch_neg_edges[1]]).unique()
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward pass through DP CWUE
                embeddings = model(
                    X, adj, all_nodes,
                    p=config['node2vec_p'],
                    q=config['node2vec_q'],
                    track_level_sizes=(batch_idx == 0)
                )
                if batch_idx == 0 and hasattr(model, "last_level_sizes"):
                    print(f"DP level sizes (step 0..R): {model.last_level_sizes}")
                
                # Map nodes to indices in embeddings
                node_to_idx = {n.item(): i for i, n in enumerate(all_nodes)}
                
                # Get embeddings for edge endpoints
                pos_u = embeddings[[node_to_idx[u.item()] for u in batch_pos_edges[0]]]
                pos_v = embeddings[[node_to_idx[v.item()] for v in batch_pos_edges[1]]]
                neg_u = embeddings[[node_to_idx[u.item()] for u in batch_neg_edges[0]]]
                neg_v = embeddings[[node_to_idx[v.item()] for v in batch_neg_edges[1]]]
                
                # Link prediction
                if config['use_mlp']:
                    pos_scores = link_predictor(pos_u, pos_v)
                    neg_scores = link_predictor(neg_u, neg_v).view(-1, int(config['neg_sample_ratio']))
                else:
                    pos_scores = torch.sigmoid((pos_u * pos_v).sum(dim=-1))
                    neg_scores = torch.sigmoid((neg_u * neg_v).sum(dim=-1)).view(-1, int(config['neg_sample_ratio']))
                
                loss = binary_cross_entropy_loss(pos_scores, neg_scores)
                running_loss += loss.item()
            
            loss.backward()
            
            # Gradient clipping
            mlp_grad_norm = 0.0
            if config['use_mlp']:
                with torch.no_grad():
                    for p in link_predictor.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            mlp_grad_norm += param_norm.item() ** 2
                    mlp_grad_norm = mlp_grad_norm ** 0.5
            
            all_params_to_clip = model.parameters()
            if config['use_mlp']:
                all_params_to_clip = chain(model.parameters(), link_predictor.parameters())
            
            torch.nn.utils.clip_grad_norm_(all_params_to_clip, float(config['grad_clip_norm']))
            
            opt.step()
            opt.zero_grad(set_to_none=True)
            
            avg_loss = running_loss
            
            wandb.log({
                'loss': avg_loss,
                'mlp_grad_norm': mlp_grad_norm,
                'lr': muon_lr,
            }, step=global_batch_idx)
            running_loss = 0.0
            
            free, total = torch.cuda.mem_get_info(device)
            used = (total - free) / total
            pbar.set_description(f"loss: {loss.item():.4f}, mem: {used:.2f}, batch_time: {time.time() - start_time_batch:.2f}s")
            pbar.update(1)
        
        if 'loss' in locals() and torch.isnan(loss):
            break
        
        # Evaluation
        is_eval_epoch = (epoch == 0) or ((epoch + 1) % epochs_eval_steps == 0) or (epoch == config['num_epochs'] - 1)
        
        if is_eval_epoch:
            (
                best_val_eval_metric,
                best_val_metrics,
                best_test_metrics,
                best_val_epoch,
                epochs_without_improvement,
                early_stop
            ) = evaluate_and_log_dp(
                model=model, link_predictor=link_predictor, adj=adj, node_features=X, config=config,
                evaluator_hit=evaluator_hit, evaluator_mrr=evaluator_mrr, device=device,
                train_pos_edge_index=eval_train_pos_edge_index, train_neg_edge_index=val_neg_edge_index,
                val_pos_edge_index=val_pos_edge_index, val_neg_edge_index=val_neg_edge_index,
                test_pos_edge_index=test_pos_edge_index, test_neg_edge_index=test_neg_edge_index,
                epoch=epoch, best_val_eval_metric=best_val_eval_metric, best_val_metrics=best_val_metrics,
                best_test_metrics=best_test_metrics, best_val_epoch=best_val_epoch,
                epochs_without_improvement=epochs_without_improvement,
                BEST_MODEL_PATH=BEST_MODEL_PATH
            )
            if early_stop:
                break

    pbar.close()

    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_gib = peak_bytes / (1024**3)
        current_gib = torch.cuda.memory_allocated() / (1024**3)
        print(f"\nPYTORCH MEMORY REPORT:")
        print(f"   Peak vRAM used: {peak_gib:.3f} GiB")
        print(f"   Current vRAM: {current_gib:.3f} GiB")

    # wandb_best_val_log = {f'best_val_{k}': v for k, v in best_val_metrics.items()}
    # wandb_best_test_log = {f'best_test_{k}': v for k, v in best_test_metrics.items()}
    # wandb.log({
    #     **wandb_best_val_log,
    #     **wandb_best_test_log,
    #     'best_val_epoch': best_val_epoch
    # })

    print(f"Best Validation {config['eval_metric']}: {best_val_eval_metric:.4f} at epoch {best_val_epoch}")
    best_val_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in best_val_metrics.items()])
    best_test_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in best_test_metrics.items()])
    print(f"Best Val Metrics: [{best_val_metrics_str}] at epoch {best_val_epoch}")
    print(f"Best Test Metrics: [{best_test_metrics_str}] at epoch {best_val_epoch}")

    results = {
        'model_data_seed': f"DPCWUE_{config['data_name']}_LinkPred_{config['seed']}",
        'best_val_epoch': best_val_epoch
    }
    for k, v in best_val_metrics.items():
        results[f'best_val_{k}'] = v
    for k, v in best_test_metrics.items():
        results[f'best_test_{k}'] = v

    experiment_results = {
        'seed': config['seed'],
        'use_mlp': config['use_mlp'],
        'global_bs': config['global_batch_size'],
        'adam_max_lr': config['adam_max_lr'],
        'muon_min_lr': config['muon_min_lr'],
        'muon_max_lr': config['muon_max_lr'],
        'mlp_lr': config['mlp_lr'],
        'walk_length': config['walk_length'],
        'num_walks': config['num_walks'],
        'node2vec_p': config['node2vec_p'],
        'node2vec_q': config['node2vec_q'],
        'recurrent_steps': config['recurrent_steps'],
        'mlp_num_layers': config['mlp_num_layers'],
        'hidden_dim': config['hidden_dim'],
        'attn_dropout': config['attn_dropout'],
        'ffn_dropout': config['ffn_dropout'],
        'resid_dropout': config['resid_dropout'],
        'mlp_dropout': config['mlp_dropout'],
        'drop_path': config['drop_path'],
        'neg_sample_ratio': config['neg_sample_ratio'],
        'patience': config['patience'],
        'best_val_epoch': best_val_epoch,
        'grad_clip_norm': config['grad_clip_norm'],
    }
    for k, v in best_test_metrics.items():
        experiment_results[f'metrics(best_test_{k})'] = v
    for k, v in best_val_metrics.items():
        experiment_results[f'metrics(best_val_{k})'] = v

    os.makedirs(f"{config['data_name']}", exist_ok=True)
    csv_filename = os.path.join(f"{config['data_name']}", f"DPCWUE_{config['data_name']}_LinkPred_{run_id}.csv")
    df = pd.DataFrame([results])
    df.to_csv(csv_filename, index=False)
    results_df = pd.DataFrame([experiment_results])
    if os.path.exists(f"{config['data_name']}/experiment_results_link_prediction_dp.csv"):
        results_df.to_csv(f"{config['data_name']}/experiment_results_link_prediction_dp.csv", mode='a', header=False, index=False)
    else:
        results_df.to_csv(f"{config['data_name']}/experiment_results_link_prediction_dp.csv", mode='w', header=True, index=False)
    print(f"Saved results to {csv_filename}")
    wandb.finish()


if __name__ == "__main__":
    main()