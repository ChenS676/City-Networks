"""
GNN Baselines for GraphBench MaxClique node classification.
Supports: GCN, SAGE, GIN, Graph Transformer, GPS, NAGphormer
— identical training loop to hrw_maxcliques.py.

Usage:
    uv run baselines.py --model gcn         --dataset_name maxclique_easy
    uv run baselines.py --model sage        --dataset_name maxclique_easy
    uv run baselines.py --model gin         --dataset_name maxclique_easy
    uv run baselines.py --model gt          --dataset_name maxclique_easy
    uv run baselines.py --model gps         --dataset_name maxclique_easy --gps_attn_type multihead
    uv run baselines.py --model nagphormer  --dataset_name maxclique_easy --num_hops 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, TransformerConv
from torch_geometric.nn import GPSConv, GINEConv
from typing import Optional, Dict, Any

try:
    from performer_pytorch import FastAttention as PerformerAttention
    HAS_PERFORMER = True
except ImportError:
    HAS_PERFORMER = False
from torch_geometric.utils import to_undirected, is_undirected, degree
from torch_scatter import scatter
import torch_geometric.transforms as T
import graphbench
import wandb
import os
import math
import argparse
import csv
import time
import numpy as np
from graphbench.helpers.utils import set_seed

_original_torch_load = torch.load
def torch_load_no_weights_only(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_no_weights_only


# ==========================================
# TRANSFORM  (same as hrw_maxcliques.py)
# ==========================================

class AddUndirectedContext(object):
    def __call__(self, data):
        if is_undirected(data.edge_index):
            mp_edge_index = data.edge_index
        else:
            mp_edge_index = to_undirected(
                data.edge_index, num_nodes=data.num_nodes, reduce="mean"
            )
        data.mp_edge_index = mp_edge_index
        n = data.num_nodes
        data.x            = torch.ones(n, 1, dtype=torch.float)
        data.mp_edge_attr  = torch.ones(mp_edge_index.size(1), 1, dtype=torch.float)
        data.edge_attr     = torch.ones(data.edge_index.size(1), 1, dtype=torch.float)
        return data


# ==========================================
# LAPLACIAN PE  (shared across all models)
# ==========================================

def compute_laplacian_eigen(edge_index, num_nodes, max_freq,
                            normalized=True, normalize=True):
    A = torch.zeros((num_nodes, num_nodes))
    A[edge_index[0], edge_index[1]] = 1
    if normalized:
        D12 = torch.diag(A.sum(1).clip(1) ** -0.5)
        L   = torch.eye(A.size(0)) - D12 @ A @ D12
    else:
        L = torch.diag(A.sum(1)) - A
    eigvals, eigvecs = torch.linalg.eigh(L)
    idx     = torch.argsort(eigvals)[:max_freq]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals = torch.real(eigvals).clamp_min(0)
    if normalize:
        eigvecs = eigvecs / eigvecs.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
    if num_nodes < max_freq:
        eigvals = F.pad(eigvals, (0, max_freq - num_nodes), value=float("nan"))
        eigvecs = F.pad(eigvecs, (0, max_freq - num_nodes), value=float("nan"))
    return eigvals.unsqueeze(0).repeat(num_nodes, 1), eigvecs


class CustomLaplacianPE:
    def __init__(self, max_freq, normalized=True, normalize=True):
        self.max_freq   = max_freq
        self.normalized = normalized
        self.normalize  = normalize

    def __call__(self, data):
        eigvals, eigvecs = compute_laplacian_eigen(
            data.edge_index, data.num_nodes,
            self.max_freq, self.normalized, self.normalize
        )
        data.eigvecs = eigvecs
        data.eigvals = eigvals
        return data


class LapPEEncoder(nn.Module):
    """
    Shared Laplacian PE encoder used by all models.
    Learns sign-invariant eigenvector embeddings (random sign flip at train time)
    and adds a learnable position-aware bias on eigenvalues.
    Output: [N, embed_dim] — added to node features before message passing.
    """
    def __init__(self, embed_dim: int, lap_pe_dim: int):
        super().__init__()
        self.lap_pe_dim = lap_pe_dim
        self.phi = nn.Sequential(
            nn.Linear(2, embed_dim, bias=False), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Position-aware eigenvalue bias
        self.eps = nn.Parameter(1e-12 * torch.arange(lap_pe_dim)[None])

    def forward(self, eigvecs, eigvals, is_training):
        eigvecs = eigvecs[:, :self.lap_pe_dim]
        eigvals = eigvals[:, :self.lap_pe_dim]
        if is_training:
            sign = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign = torch.where(sign >= 0.5,
                               torch.ones_like(sign), -torch.ones_like(sign))
            eigvecs = eigvecs * sign.unsqueeze(0)
        eigvals = eigvals + self.eps[:, :self.lap_pe_dim]
        x = torch.stack((eigvecs, eigvals), dim=2)   # [N, k, 2]
        x[torch.isnan(x)] = 0
        return self.rho(self.phi(x).sum(1))           # [N, embed_dim]


# ==========================================
# MODELS
# ==========================================

class GCN(nn.Module):
    """GCN (Kipf & Welling 2017) + Laplacian PE at hidden_dim."""
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes, dropout, lap_pe_dim=16):
        super().__init__()
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.pe_encoder   = LapPEEncoder(hidden_dim, lap_pe_dim)
        self.input_norm   = nn.LayerNorm(hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, eigvecs=None, eigvals=None, **kwargs):
        x = self.input_norm(self.node_encoder(x))
        if eigvecs is not None and eigvals is not None:
            x = x + self.pe_encoder(eigvecs, eigvals, self.training)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.gelu(conv(x, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        return self.head(x)


class GraphSAGE(nn.Module):
    """GraphSAGE (Hamilton et al. 2017) + Laplacian PE at hidden_dim."""
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes, dropout, lap_pe_dim=16):
        super().__init__()
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.pe_encoder   = LapPEEncoder(hidden_dim, lap_pe_dim)
        self.input_norm   = nn.LayerNorm(hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, eigvecs=None, eigvals=None, **kwargs):
        x = self.input_norm(self.node_encoder(x))
        if eigvecs is not None and eigvals is not None:
            x = x + self.pe_encoder(eigvecs, eigvals, self.training)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.gelu(conv(x, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        return self.head(x)


class GIN(nn.Module):
    """GIN (Xu et al. 2019) + Laplacian PE at hidden_dim."""
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes, dropout, lap_pe_dim=16):
        super().__init__()
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.pe_encoder   = LapPEEncoder(hidden_dim, lap_pe_dim)
        self.input_norm   = nn.LayerNorm(hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        def mlp(d):
            return nn.Sequential(
                nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, d),
            )

        for _ in range(num_layers):
            self.convs.append(GINConv(mlp(hidden_dim), train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, eigvecs=None, eigvals=None, **kwargs):
        x = self.input_norm(self.node_encoder(x))
        if eigvecs is not None and eigvals is not None:
            x = x + self.pe_encoder(eigvecs, eigvals, self.training)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.gelu(conv(x, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        return self.head(x)


class GraphTransformer(nn.Module):
    """
    Graph Transformer (Dwivedi & Bresson 2020) + Laplacian PE.
    PE is injected AFTER the initial node projection (hidden_dim),
    so the full hidden_dim carries positional information into attention.
    """
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes,
                 num_heads, dropout, edge_dim=None, lap_pe_dim=16):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.dropout = dropout
        head_dim     = hidden_dim // num_heads

        # Project raw features to hidden_dim first, then add PE at full width
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.pe_encoder   = LapPEEncoder(hidden_dim, lap_pe_dim)   # ← hidden_dim
        self.input_norm   = nn.LayerNorm(hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # All layers now operate at hidden_dim → hidden_dim
        for _ in range(num_layers):
            self.convs.append(TransformerConv(hidden_dim, head_dim, heads=num_heads,
                              dropout=dropout, edge_dim=edge_dim, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_attr=None,
                eigvecs=None, eigvals=None, **kwargs):
        # 1. Project to hidden_dim
        x = self.input_norm(self.node_encoder(x))
        # 2. Add LapPE at full hidden_dim width — attention sees full positional signal
        if eigvecs is not None and eigvals is not None:
            x = x + self.pe_encoder(eigvecs, eigvals, self.training)
        # 3. Transformer layers with residual
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr=edge_attr)
            h = norm(F.gelu(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h   # residual (dims always match now)
        return self.head(x)




# ==========================================
# GPS  (General, Powerful, Scalable GNN)
# ==========================================

class RedrawProjection:
    """
    Periodically redraws the random projection matrix inside every
    PerformerAttention module found in the wrapped model.
    Only active during training and when attn_type == 'performer'.
    """
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model            = model
        self.redraw_interval  = redraw_interval
        self.num_last_redraw  = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if HAS_PERFORMER and isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class GPSModel(nn.Module):
    """
    GPS: General, Powerful, Scalable Graph Transformers (Rampášek et al. 2022).
    Adapted for node-level classification. Uses shared LapPEEncoder.
    """
    def __init__(self, in_dim, edge_dim, hidden_dim, lap_pe_dim,
                 num_layers, num_classes, num_heads,
                 attn_type='multihead', dropout=0.1,
                 attn_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        if attn_type == 'performer' and not HAS_PERFORMER:
            raise ImportError(
                "performer-pytorch required. Install: uv pip install performer-pytorch"
            )
        attn_kwargs = attn_kwargs or {}

        self.pe_encoder   = LapPEEncoder(hidden_dim, lap_pe_dim)
        self.node_encoder = nn.Linear(in_dim,   hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.input_norm   = nn.LayerNorm(hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GPSConv(
                channels=hidden_dim, conv=GINEConv(mlp, train_eps=True),
                heads=num_heads, dropout=dropout,
                attn_type=attn_type, attn_kwargs=attn_kwargs,
            ))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None,
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                eigvecs=None, eigvals=None, **kwargs):
        x = self.input_norm(self.node_encoder(x))

        if eigvecs is not None and eigvals is not None:
            x = x + self.pe_encoder(eigvecs, eigvals, self.training)

        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), 1, device=x.device)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr.float())

        self.redraw_projection.redraw_projections()
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        return self.head(x)



# ==========================================
# NAGphormer  (Tokenized Graph Transformer)
# ==========================================

def k_hop_aggregation(x: torch.Tensor,
                      edge_index: torch.Tensor,
                      num_nodes: int,
                      num_hops: int) -> torch.Tensor:
    """
    Compute sym-normalised K-hop neighbourhood aggregations.
    Returns [N, num_hops+1, D] where index 0 is the node itself (hop-0).

    Uses D^{-1/2} A D^{-1/2} normalisation so features stay bounded.
    Because PyG batches graphs by offsetting node indices, edges never
    cross graph boundaries — aggregation is automatically graph-local.
    """
    row, col = edge_index
    deg          = degree(col, num_nodes, dtype=x.dtype).clamp(min=1)
    deg_inv_sqrt = deg.pow(-0.5)
    norm         = deg_inv_sqrt[row] * deg_inv_sqrt[col]   # [E]

    hops = [x]      # hop-0: the node's own features
    h    = x
    for _ in range(num_hops):
        # Aggregate: h_new[i] = sum_{j in N(i)} norm[i,j] * h[j]
        agg = scatter(
            norm.unsqueeze(-1) * h[col],   # [E, D]
            row,                            # target node indices
            dim=0,
            dim_size=num_nodes,
            reduce='sum',
        )                                  # [N, D]
        h = agg
        hops.append(h)

    return torch.stack(hops, dim=1)        # [N, K+1, D]


class NAGphormer(nn.Module):
    """
    NAGphormer: A Tokenized Graph Transformer for Node Classification
    (Chen et al., 2023 — https://arxiv.org/abs/2206.04910).

    Core idea:
      For each node, compute K hop-wise neighbourhood aggregations
      h_0, h_1, ..., h_K (where h_0 is the node itself).
      Treat these as a sequence of K+1 tokens and process with a
      standard pre-norm Transformer encoder.
      The h_0 output token is used for node classification.

    Adaptations for MaxClique / this codebase:
      - Linear encoders replace graph-specific atom embeddings.
      - LapPE added to h_0 (the self-token) before Transformer.
      - Hop embeddings are learnable and added to every token.
      - Works with PyG batched graphs (aggregation stays graph-local).
    """
    def __init__(self, in_dim, hidden_dim, num_layers, num_classes,
                 num_heads, dropout, num_hops=3, lap_pe_dim=16):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.num_hops = num_hops

        # ── Input encoding ────────────────────────────────────────────────
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.input_norm   = nn.LayerNorm(hidden_dim)

        # LapPE: injected into the self-token (hop-0) only
        self.pe_encoder   = LapPEEncoder(hidden_dim, lap_pe_dim)

        # Hop-distance positional embedding (one per hop level 0..K)
        self.hop_emb = nn.Embedding(num_hops + 1, hidden_dim)

        # ── Transformer encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = hidden_dim,
            nhead           = num_heads,
            dim_feedforward = hidden_dim * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,    # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            norm       = nn.LayerNorm(hidden_dim),
        )

        # ── Classifier head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index,
                eigvecs=None, eigvals=None, **kwargs):
        N = x.size(0)
        device = x.device

        # 1. Encode raw features: [N, hidden_dim]
        h0 = self.input_norm(self.node_encoder(x))

        # 2. Compute K-hop aggregations using encoded features
        #    Result: [N, K+1, hidden_dim]
        tokens = k_hop_aggregation(h0, edge_index, N, self.num_hops)

        # 3. Add LapPE to the self-token (index 0) only
        if eigvecs is not None and eigvals is not None:
            tokens[:, 0, :] = tokens[:, 0, :] + \
                               self.pe_encoder(eigvecs, eigvals, self.training)

        # 4. Add hop-distance positional embeddings to all tokens
        hop_ids = torch.arange(self.num_hops + 1, device=device)   # [K+1]
        tokens  = tokens + self.hop_emb(hop_ids).unsqueeze(0)       # broadcast [N, K+1, D]

        # 5. Transformer over the token sequence
        out = self.transformer(tokens)    # [N, K+1, hidden_dim]

        # 6. Classify from the self-token (hop-0 output)
        return self.head(out[:, 0, :])   # [N, num_classes]


MODEL_REGISTRY = {"gcn": GCN, "sage": GraphSAGE, "gin": GIN,
                  "gt": GraphTransformer, "gps": GPSModel,
                  "nagphormer": NAGphormer}


# ==========================================
# HELPERS
# ==========================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_forward(model, data):
    x          = data.x.float()
    edge_index = getattr(data, 'mp_edge_index', data.edge_index)
    edge_attr  = getattr(data, 'mp_edge_attr',  data.edge_attr)
    if x.dim()         == 1: x         = x.unsqueeze(-1)
    if edge_attr.dim() == 1: edge_attr = edge_attr.unsqueeze(-1)
    return model(
        x, edge_index,
        edge_attr = edge_attr.float(),
        batch     = getattr(data, 'batch', None),
        rwse      = getattr(data, 'rwse',    None),
        eigvecs   = getattr(data, 'eigvecs', None),
        eigvals   = getattr(data, 'eigvals', None),
    )


def parse_metrics(metrics):
    if isinstance(metrics, dict):
        f1  = metrics.get('f1', metrics.get('F1', 0.0))
        acc = metrics.get('accuracy', metrics.get('acc', f1))
        return acc, f1
    if isinstance(metrics, (list, tuple)):
        acc = metrics[0] if len(metrics) > 0 else 0.0
        f1  = metrics[1] if len(metrics) > 1 else acc
        return acc, f1
    return metrics, metrics


def global_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total ** 0.5


# ==========================================
# EVALUATION
# ==========================================

@torch.no_grad()
def evaluate(model, loader, device, evaluator):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = run_forward(model, data)
        y_true.append(data.y.cpu())
        y_pred.append(logits.argmax(-1, keepdim=True).long().cpu())

    y_true = torch.cat(y_true); y_pred = torch.cat(y_pred)
    if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
    if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(1)

    TP = ((y_pred == 1) & (y_true == 1)).sum().float()
    FP = ((y_pred == 1) & (y_true == 0)).sum().float()
    FN = ((y_pred == 0) & (y_true == 1)).sum().float()
    precision = (TP / (TP + FP)).item() if (TP + FP) > 0 else 0.0
    recall    = (TP / (TP + FN)).item() if (TP + FN) > 0 else 0.0
    common    = ((y_pred == 1) & (y_true == 1)).sum().item()
    union     = ((y_pred == 1) | (y_true == 1)).sum().item()
    jaccard   = common / union if union > 0 else 0.0
    metrics   = evaluator.evaluate(y_true, y_pred) if evaluator else 0.0
    acc, f1   = parse_metrics(metrics)
    return dict(f1=f1, acc=acc, precision=precision, recall=recall,
                jaccard=jaccard, TP=int(TP), FP=int(FP), FN=int(FN))


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    # Model selection
    parser.add_argument("--model",          type=str,   default="gcn",
                        choices=["gcn", "sage", "gin", "gt", "gps", "nagphormer"])
    # Data
    parser.add_argument("--dataset_name",   type=str,   default="maxclique_easy")
    parser.add_argument("--data_root",      type=str,   default="./data_graphbench")
    parser.add_argument("--seed",           type=int,   default=2025)
    # Training
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--test_batch_size",type=int,   default=32)
    parser.add_argument("--train_subset_ratio", type=float, default=0.1)
    parser.add_argument("--adam_max_lr",    type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.5)
    # Architecture
    parser.add_argument("--hidden_dim",     type=int,   default=256)
    parser.add_argument("--num_layers",     type=int,   default=4)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--num_heads",      type=int,   default=8,
                        help="Attention heads (GT, GPS, NAGphormer)")
    parser.add_argument("--num_hops",       type=int,   default=3,
                        help="Hop depth K for NAGphormer (sequence length = K+1)")
    # Laplacian PE — always on for all models
    parser.add_argument("--lap_pe_dim",     type=int,   default=16,
                        help="Number of Laplacian eigenvectors (applied to all models)")
    # GPS-specific
    parser.add_argument("--gps_attn_type",  type=str,   default="multihead",
                        choices=["multihead", "performer", "flash"],
                        help="GPS global attention type")
    # Logging
    parser.add_argument("--wandb_project",  type=str,   default="bench_maxclique_baselines")
    parser.add_argument("--log_every",      type=int,   default=10)
    parser.add_argument("--eval_metric_class", type=str, default="algoreas_classification")

    args   = parser.parse_args()
    config = vars(args)

    import pprint
    pprint.pp(config)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Dataset — LapPE always applied ───────────────────────────────────
    transform_list = [
        AddUndirectedContext(),
        CustomLaplacianPE(max_freq=config["lap_pe_dim"], normalized=True, normalize=True),
    ]
    loader  = graphbench.Loader(
        root=config["data_root"],
        dataset_names=config["dataset_name"],
        transform=T.Compose(transform_list)
    )
    dataset = loader.load()
    try:
        train_dataset = dataset[0]['train']
        val_dataset   = dataset[0]['valid']
        test_dataset  = dataset[0]['test']
    except (TypeError, KeyError):
        train_dataset = val_dataset = test_dataset = dataset

    print(f"Sizes → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    val_loader  = DataLoader(val_dataset,  batch_size=config["test_batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)

    _peek       = next(iter(DataLoader(train_dataset, batch_size=4, shuffle=False)))
    node_in_dim = 1 if _peek.x.dim() == 1 else _peek.x.size(1)
    edge_in_dim = 1 if _peek.edge_attr.dim() == 1 else _peek.edge_attr.size(1)
    print(f"Input dims → node: {node_in_dim}  edge: {edge_in_dim}")

    # ── Model — lap_pe_dim passed to every model ──────────────────────────
    ModelClass   = MODEL_REGISTRY[config["model"]]
    model_kwargs = dict(
        in_dim     = node_in_dim,
        hidden_dim = config["hidden_dim"],
        num_layers = config["num_layers"],
        num_classes= 2,
        dropout    = config["dropout"],
        lap_pe_dim = config["lap_pe_dim"],   # all models use LapPE
    )
    if config["model"] in ("gt", "gps"):
        model_kwargs["num_heads"] = config["num_heads"]
        model_kwargs["edge_dim"]  = edge_in_dim
    if config["model"] == "nagphormer":
        model_kwargs["num_heads"] = config["num_heads"]
        model_kwargs["num_hops"]  = config["num_hops"]
    if config["model"] == "gps":
        model_kwargs["attn_type"]   = config["gps_attn_type"]
        model_kwargs["attn_kwargs"] = (
            {"kernel_size": 256} if config["gps_attn_type"] == "performer" else {}
        )

    model = ModelClass(**model_kwargs).to(device)

    total_params = count_parameters(model)
    print(f"Model: {config['model'].upper()}  |  Parameters: {total_params:,}")

    # ── Schedule ──────────────────────────────────────────────────────────
    num_train_total   = len(train_dataset)
    window_size       = int(num_train_total * config["train_subset_ratio"])
    batches_per_epoch = math.ceil(window_size / config["batch_size"])
    total_steps       = batches_per_epoch * config["epochs"]
    eval_every_steps  = max(1, round(max(1, int(num_train_total * 0.1)) / config["batch_size"]))

    print(f"window={window_size:,} | steps/epoch={batches_per_epoch} "
          f"| total={total_steps} | eval_every={eval_every_steps}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["adam_max_lr"],
                                  weight_decay=config["weight_decay"])

    # ── Class-weighted loss to handle clique node imbalance ───────────────
    # Count class frequencies over the full training set
    all_labels = torch.cat([train_dataset[i].y for i in range(len(train_dataset))])
    n_total    = all_labels.numel()
    n_pos      = (all_labels == 1).sum().item()
    n_neg      = n_total - n_pos
    # Weight = inverse frequency, normalised so they average to 1
    w_neg      = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
    w_pos      = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float, device=device)
    print(f"Class weights → neg: {w_neg:.3f}  pos: {w_pos:.3f}  "
          f"(clique ratio: {n_pos/n_total*100:.1f}%)")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    try:
        evaluator = graphbench.Evaluator(config["eval_metric_class"])
    except Exception:
        evaluator = None

    # ── W&B ───────────────────────────────────────────────────────────────
    model_tag = config["model"].upper()
    if config["model"] == "gps":
        model_tag += f"-{config['gps_attn_type']}"
    if config["model"] == "nagphormer":
        model_tag += f"-K{config['num_hops']}"
    run_name = (
        f"{model_tag}_"
        f"HD{config['hidden_dim']}_L{config['num_layers']}_"
        f"lapPE{config['lap_pe_dim']}_"
        f"BS{config['batch_size']}_LR{config['adam_max_lr']}"
    )
    wandb.init(
        entity="graph-diffusion-model-link-prediction",
        project=config["wandb_project"],
        group=config["dataset_name"],
        name=run_name, config=config,
    )
    wandb.summary["total_params"] = total_params

    checkpoint_dir  = os.path.join(config["dataset_name"], "checkpoints_baselines")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_{run_name}.pt")

    # ── Pre-training eval ─────────────────────────────────────────────────
    model.eval()
    v0 = evaluate(model, val_loader,  device, evaluator)
    t0 = evaluate(model, test_loader, device, evaluator)
    print(f"[step 0] val_f1={v0['f1']:.4f}  test_f1={t0['f1']:.4f}")
    wandb.log({"step": 0,
               **{f"val/{k}":  v for k, v in v0.items()},
               **{f"test/{k}": v for k, v in t0.items()}})

    # ── Training ──────────────────────────────────────────────────────────
    print("\nStarting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_wall       = time.time()
    global_step      = 0
    running_loss_sum = 0.0
    running_loss_cnt = 0
    best_val_f1, best_test_f1     = -float("inf"), -float("inf")
    best_val_step, best_test_step = 0, 0

    for epoch in range(1, config["epochs"] + 1):
        start_idx = ((epoch - 1) * window_size) % num_train_total
        indices   = [(start_idx + i) % num_train_total for i in range(window_size)]
        train_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, indices),
            batch_size=config["batch_size"], shuffle=True,
            num_workers=4, pin_memory=True
        )
        model.train()
        epoch_t0 = time.time()

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = run_forward(model, data)
                loss   = criterion(logits, data.y.long())

            loss.backward()
            gnorm = global_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            optimizer.step()

            global_step      += 1
            running_loss_sum += loss.item()
            running_loss_cnt += 1

            if global_step % config["log_every"] == 0:
                wandb.log({
                    "step": global_step, "epoch": epoch,
                    "train/loss": loss.item(), "train/grad_norm": gnorm,
                    "train/lr":   optimizer.param_groups[0]["lr"],
                }, step=global_step)

            vram = 0.0
            if torch.cuda.is_available():
                free, tmem = torch.cuda.mem_get_info(device)
                vram = (tmem - free) / tmem
            print(
                f"\rEp {epoch}/{config['epochs']} "
                f"[{batch_idx+1}/{len(train_loader)}] "
                f"step={global_step} loss={loss.item():.4f} "
                f"gnorm={gnorm:.3f} mem={vram:.2f} "
                f"t={time.time()-epoch_t0:.0f}s",
                end="", flush=True
            )

            if global_step % eval_every_steps == 0:
                avg_loss = running_loss_sum / running_loss_cnt
                running_loss_sum = 0.0; running_loss_cnt = 0

                model.eval()
                val_res  = evaluate(model, val_loader,  device, evaluator)
                test_res = evaluate(model, test_loader, device, evaluator)
                model.train()

                wandb.log({
                    "step": global_step, "epoch": epoch,
                    "train/avg_loss": avg_loss,
                    "best/val_f1":  best_val_f1,
                    "best/test_f1": best_test_f1,
                    **{f"val/{k}":  v for k, v in val_res.items()},
                    **{f"test/{k}": v for k, v in test_res.items()},
                }, step=global_step)

                print(
                    f"\n  [eval step={global_step}] "
                    f"val_f1={val_res['f1']:.4f} "
                    f"test_f1={test_res['f1']:.4f} "
                    f"avg_loss={avg_loss:.4f}"
                )

                if val_res["f1"] > best_val_f1:
                    best_val_f1   = val_res["f1"]
                    best_val_step = global_step
                    wandb.summary.update({"best_val_f1": best_val_f1,
                                          "best_val_step": best_val_step})
                    print(f"  ★ new best val  F1={best_val_f1:.4f}  (step {best_val_step})")

                if test_res["f1"] > best_test_f1:
                    best_test_f1   = test_res["f1"]
                    best_test_step = global_step
                    wandb.summary.update({"best_test_f1": best_test_f1,
                                          "best_test_step": best_test_step})
                    torch.save({
                        "step": global_step, "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_test_f1": best_test_f1, "config": config,
                    }, best_model_path)
                    print(f"  ★ new best test F1={best_test_f1:.4f}  (step {best_test_step}) → saved")

        print(
            f"\n[epoch {epoch}/{config['epochs']} done] "
            f"wall={time.time()-start_wall:.0f}s  "
            f"best_val={best_val_f1:.4f}  best_test={best_test_f1:.4f}"
        )

    # ── Final eval ────────────────────────────────────────────────────────
    print("\nFinal evaluation...")
    model.eval()
    final_val  = evaluate(model, val_loader,  device, evaluator)
    final_test = evaluate(model, test_loader, device, evaluator)
    peak_mem   = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    total_time = time.time() - start_wall
    print(f"Final → val_f1={final_val['f1']:.4f}  test_f1={final_test['f1']:.4f}")
    print(f"Peak VRAM: {peak_mem:.2f} GiB  |  Total time: {total_time:.0f}s")

    wandb.log({
        "step": global_step,
        "final/val_f1":        final_val["f1"],
        "final/test_f1":       final_test["f1"],
        "final/peak_vram_gib": peak_mem,
        "final/runtime_sec":   total_time,
    }, step=global_step)
    wandb.summary.update({
        "best_val_f1":   best_val_f1,
        "best_test_f1":  best_test_f1,
        "best_val_step": best_val_step,
        "final_val_f1":  final_val["f1"],
        "final_test_f1": final_test["f1"],
        "total_params":  total_params,
    })

    # ── CSV ───────────────────────────────────────────────────────────────
    os.makedirs(config["dataset_name"], exist_ok=True)
    csv_path = f"{config['dataset_name']}/baseline_results.csv"
    row = {
        "model":          config["model"],
        "dataset":        config["dataset_name"],
        "seed":           config["seed"],
        "hidden_dim":     config["hidden_dim"],
        "num_layers":     config["num_layers"],
        "batch_size":     config["batch_size"],
        "adam_max_lr":    config["adam_max_lr"],
        "epochs":         config["epochs"],
        "dropout":        config["dropout"],
        "total_params":   total_params,
        "best_val_f1":    round(best_val_f1,       4),
        "best_test_f1":   round(best_test_f1,      4),
        "best_val_step":  best_val_step,
        "final_val_f1":   round(final_val["f1"],   4),
        "final_test_f1":  round(final_test["f1"],  4),
        "runtime_sec":    round(total_time,         1),
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists: w.writeheader()
        w.writerow(row)
    print(f"Logged to {csv_path}")
    wandb.finish()


if __name__ == "__main__":
    main()