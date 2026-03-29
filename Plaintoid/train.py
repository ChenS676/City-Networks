"""
train_heart.py
--------------
Training, evaluation, and logging script for HeART link prediction.

Imports the model and method components from heart_model.py.

Memory and training time profiling is collected throughout training and saved
to a dedicated CSV file at the end of each run:
    <data_name>/profiling/profiling_<run_id>.csv          (per-epoch rows)
    <data_name>/profiling/profiling_summary.csv           (one row per run, appended)

Usage:
    python train_heart.py --data_name "Cora" --data_root "data/Cora"
    python train_heart.py --data_name "CiteSeer" --data_root "data/CiteSeer"
"""

import os
os.environ["WANDB_MODE"] = "offline"
import wandb
import psutil                          # pip install psutil  (CPU RAM tracking)
import numpy as np
import argparse
import pickle
import math
import time
import datetime
from itertools import chain

import torch
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
import pandas as pd
import tqdm.auto as tqdm
from muon import SingleDeviceMuonWithAuxAdam

from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc

# Import all model components from the extracted model file
from model import (
    MupConfig,
    Transformer,
    LinkPredictorMLP,
    binary_cross_entropy_loss,
    get_random_walk_batch,
    anonymize_rws,
)

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
])


# ===========================================================================
# Profiling Helpers
# ===========================================================================

def get_cpu_ram_gib() -> float:
    """Return current process RSS memory in GiB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def get_gpu_stats(device) -> dict:
    """
    Return a dict of current GPU memory stats (GiB).
    Returns zeros if no GPU is available.
    """
    if not torch.cuda.is_available():
        return {
            "vram_allocated_gib": 0.0,
            "vram_reserved_gib":  0.0,
            "vram_peak_gib":      0.0,
            "vram_total_gib":     0.0,
            "vram_utilization_pct": 0.0,
        }
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved  = torch.cuda.memory_reserved(device)  / (1024 ** 3)
    peak      = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    total     = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return {
        "vram_allocated_gib":   round(allocated, 4),
        "vram_reserved_gib":    round(reserved,  4),
        "vram_peak_gib":        round(peak,      4),
        "vram_total_gib":       round(total,     2),
        "vram_utilization_pct": round(allocated / total * 100, 2) if total > 0 else 0.0,
    }


def save_profiling_csv(epoch_profile_rows: list, summary_row: dict, output_dir: str, run_id: str):
    """
    Save two profiling CSV files:
      1. per-epoch detailed log  → profiling_<run_id>.csv
      2. single-row run summary  → profiling_summary.csv  (appended across runs)
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Per-epoch detailed log ---
    per_epoch_path = os.path.join(output_dir, f"profiling_{run_id}.csv")
    pd.DataFrame(epoch_profile_rows).to_csv(per_epoch_path, index=False)
    print(f"[Profiling] Per-epoch log saved → {per_epoch_path}")

    # --- Run summary (append-friendly) ---
    summary_path = os.path.join(output_dir, "profiling_summary.csv")
    summary_df = pd.DataFrame([summary_row])
    if os.path.exists(summary_path):
        summary_df.to_csv(summary_path, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(summary_path, mode='w', header=True, index=False)
    print(f"[Profiling] Run summary appended → {summary_path}")


# ===========================================================================
# Argument Parsing
# ===========================================================================

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


def get_config():
    parser = argparse.ArgumentParser(description="HeART Link Prediction on Planetoid data")

    run_group = parser.add_argument_group('Run Configuration')
    run_group.add_argument('--seed', type=int, default=2025, help='Random seed')
    run_group.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
    run_group.add_argument('--global_batch_size', type=int, default=256, help='Batch size for positive edges')
    run_group.add_argument('--patience', type=int, default=4, help='Epochs for early stopping patience')
    run_group.add_argument('--train_edge_downsample_ratio', type=float, default=1.0,
                           help='Ratio to downsample training edges. 1.0 = no downsampling.')
    run_group.add_argument('--eval_metric', type=str, default='MRR',
                           help='Evaluation metric for choosing the best model')

    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data_root', type=str, default="data/Cora")
    data_group.add_argument('--data_name', type=str, default="Cora")
    data_group.add_argument('--val_split_ratio', type=float, default=0.15)
    data_group.add_argument('--test_split_ratio', type=float, default=0.05)
    data_group.add_argument('--deepwalk_pkl_path', type=str, default=None)
    data_group.add_argument('--use_fixed_splits', type=str_to_bool, default=False)
    data_group.add_argument('--split_dir', type=str, default="data/Cora/fixed_splits")
    data_group.add_argument('--use_laplacian_pe', type=str_to_bool, default=False)
    data_group.add_argument('--laplacian_pe_path', type=str, default="data/Cora/laplacian_pe.pt")
    data_group.add_argument('--laplacian_edge_subsampling_ratio', type=float, default=1.0)

    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--num_layers', type=int, default=1)
    model_group.add_argument('--hidden_dim', type=int, default=128)
    model_group.add_argument('--intermediate_dim_multiplier', type=int, default=4)
    model_group.add_argument('--num_heads', type=int, default=16)
    model_group.add_argument('--recurrent_steps', type=int, default=1)
    model_group.add_argument('--mup_init_std', type=float, default=0.01)
    model_group.add_argument('--mup_width_multiplier', type=float, default=2.0)

    walk_group = parser.add_argument_group('Random Walk')
    walk_group.add_argument('--walk_length', type=int, default=8)
    walk_group.add_argument('--num_walks', type=int, default=16)
    walk_group.add_argument('--node2vec_p', type=float, default=1.0)
    walk_group.add_argument('--node2vec_q', type=float, default=1.0)

    optim_group = parser.add_argument_group('Optimizer')
    optim_group.add_argument('--muon_min_lr', type=float, default=1e-4)
    optim_group.add_argument('--muon_max_lr', type=float, default=1e-3)
    optim_group.add_argument('--adam_max_lr', type=float, default=1e-4)
    optim_group.add_argument('--adam_min_lr', type=float, default=0.0)
    optim_group.add_argument('--grad_clip_norm', type=float, default=0.1)

    mlp_group = parser.add_argument_group('MLP Link Predictor')
    mlp_group.add_argument('--use_mlp', type=str_to_bool, default=True)
    mlp_group.add_argument('--mlp_num_layers', type=int, default=3)
    mlp_group.add_argument('--mlp_lr', type=float, default=1e-3)
    mlp_group.add_argument('--mlp_dropout', type=float, default=0.1)

    reg_group = parser.add_argument_group('Regularization')
    reg_group.add_argument('--attn_dropout', type=float, default=0.1)
    reg_group.add_argument('--ffn_dropout', type=float, default=0.1)
    reg_group.add_argument('--resid_dropout', type=float, default=0.1)
    reg_group.add_argument('--drop_path', type=float, default=0.05)

    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--neg_sample_ratio', type=int, default=1)
    misc_group.add_argument('--hits_k', type=int, nargs='+', default=[1, 10, 50, 100])
    misc_group.add_argument('--use_deepwalk_embeds', type=str_to_bool, default=False)

    logging_group = parser.add_argument_group('Weights & Biases Logging')
    logging_group.add_argument('--wb_entity', type=str, default="graph-diffusion-model-link-prediction")
    logging_group.add_argument('--wb_project', type=str, default="ani-cwue-link-prediction-final")

    args = parser.parse_args()
    config = vars(args)
    config['intermediate_dim'] = config['hidden_dim'] * config['intermediate_dim_multiplier']
    del config['intermediate_dim_multiplier']
    return config


# ===========================================================================
# Data Loading
# ===========================================================================

def load_graph_arxiv23(data_root) -> Data:
    data = torch.load(data_root + 'arxiv_2023/graph.pt', weights_only=False)
    return data


# ===========================================================================
# Training Utilities
# ===========================================================================

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


def trapezoidal_lr_schedule(global_batch_idx, max_lr, min_lr, warmup, cool, total_batches):
    if global_batch_idx <= warmup:
        lr = (global_batch_idx / warmup) * (max_lr - min_lr) + min_lr
    elif warmup < global_batch_idx <= (total_batches - cool):
        lr = max_lr
    else:
        lr_scale = (total_batches - global_batch_idx) / cool
        lr = lr_scale * (max_lr - min_lr) + min_lr
    return lr


# ===========================================================================
# Evaluation
# ===========================================================================

def get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, k_list: list):
    result = {}

    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    for K in k_list:
        result[f'Hits@{K}'] = result_hit_val[f'Hits@{K}']

    result_mrr_val = evaluate_mrr(
        evaluator_mrr, pos_val_pred,
        neg_val_pred.repeat(pos_val_pred.size(0), 1)
    )
    result['MRR'] = result_mrr_val['MRR']

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([
        torch.ones(pos_val_pred.size(0), dtype=int),
        torch.zeros(neg_val_pred.size(0), dtype=int)
    ])
    result_auc_val = evaluate_auc(val_pred, val_true)
    result['AUC'] = result_auc_val['AUC']
    result['AP']  = result_auc_val['AP']
    return result


@torch.no_grad()
def test_edge(model, link_predictor, adj, X, config, edge_index, batch_size):
    input_data = edge_index.t()
    all_scores = []

    for perm in DataLoader(range(input_data.size(0)), batch_size=batch_size):
        batch_edge_index = input_data[perm].t()
        nodes = batch_edge_index.unique()

        batch, anon_indices = get_random_walk_batch(
            adj, X, nodes,
            walk_length=config['walk_length'],
            num_walks=config['num_walks'],
            recurrent_steps=config['recurrent_steps'],
            p=config['node2vec_p'],
            q=config['node2vec_q']
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings = model(batch, anon_indices)
            node_to_idx = {n.item(): i for i, n in enumerate(nodes)}
            u = embeddings[[node_to_idx[n.item()] for n in batch_edge_index[0]]]
            v = embeddings[[node_to_idx[n.item()] for n in batch_edge_index[1]]]

            if config['use_mlp']:
                scores = link_predictor(u, v)
            else:
                scores = torch.sigmoid((u * v).sum(dim=-1))
        all_scores.append(scores.cpu())

    return torch.cat(all_scores, dim=0).float()


@torch.no_grad()
def evaluate_link_prediction(model, link_predictor, edge_index, neg_edge_index,
                              adj, X, config, evaluator_hit, evaluator_mrr,
                              device, eval_batch_size=512):
    model.eval()
    if config['use_mlp']:
        link_predictor.eval()

    pos_scores = test_edge(model, link_predictor, adj, X, config, edge_index, eval_batch_size)
    neg_scores = test_edge(model, link_predictor, adj, X, config, neg_edge_index, eval_batch_size)

    neg_valid_pred = torch.flatten(neg_scores)
    pos_valid_pred = torch.flatten(pos_scores)
    k_list = config.get('hits_k', [1, 10, 50, 100])

    return get_metric_score(
        evaluator_hit=evaluator_hit, evaluator_mrr=evaluator_mrr,
        pos_val_pred=pos_valid_pred, neg_val_pred=neg_valid_pred, k_list=k_list
    )


@torch.no_grad()
def evaluate_and_log(
    model, link_predictor, adj, X, config,
    evaluator_hit, evaluator_mrr, device,
    train_pos_edge_index, train_neg_edge_index,
    val_pos_edge_index, val_neg_edge_index,
    test_pos_edge_index, test_neg_edge_index,
    epoch, best_val_eval_metric, best_val_metrics,
    best_test_metrics, best_val_epoch,
    epochs_without_improvement, BEST_MODEL_PATH
):
    """
    Runs evaluation, logs metrics to W&B, saves the best model, and checks for early stopping.
    Tracks peak value for each metric independently.
    """
    model.eval()
    if config['use_mlp']:
        link_predictor.eval()

    train_results = evaluate_link_prediction(
        model, link_predictor,
        edge_index=train_pos_edge_index, neg_edge_index=train_neg_edge_index,
        adj=adj, X=X, config=config,
        evaluator_hit=evaluator_hit, evaluator_mrr=evaluator_mrr,
        eval_batch_size=config['global_batch_size'], device=device
    )
    val_results = evaluate_link_prediction(
        model, link_predictor,
        edge_index=val_pos_edge_index, neg_edge_index=val_neg_edge_index,
        adj=adj, X=X, config=config,
        evaluator_hit=evaluator_hit, evaluator_mrr=evaluator_mrr,
        eval_batch_size=config['global_batch_size'], device=device
    )
    test_results = evaluate_link_prediction(
        model, link_predictor,
        edge_index=test_pos_edge_index, neg_edge_index=test_neg_edge_index,
        adj=adj, X=X, config=config,
        evaluator_hit=evaluator_hit, evaluator_mrr=evaluator_mrr,
        eval_batch_size=config['global_batch_size'], device=device
    )

    wandb.log({
        **{f'train_{k}': v for k, v in train_results.items()},
        **{f'val_{k}':   v for k, v in val_results.items()},
        **{f'test_{k}':  v for k, v in test_results.items()},
        'epoch': epoch + 1
    })

    print(f"Epoch {epoch + 1}: "
          f"Train [{', '.join(f'{k}: {v:.4f}' for k, v in train_results.items())}] | "
          f"Val   [{', '.join(f'{k}: {v:.4f}' for k, v in val_results.items())}] | "
          f"Test  [{', '.join(f'{k}: {v:.4f}' for k, v in test_results.items())}]")

    for k, v in val_results.items():
        best_val_metrics[k] = max(v, best_val_metrics.get(k, 0.0))
        if wandb.run is not None:
            wandb.run.summary[f"best_val_{k}"] = v
    for k, v in test_results.items():
        best_test_metrics[k] = max(v, best_test_metrics.get(k, 0.0))
        if wandb.run is not None:
            wandb.run.summary[f"best_test_{k}"] = v

    val_metric     = val_results[config['eval_metric']]
    early_stop_flag = False

    if val_metric > best_val_eval_metric:
        best_val_eval_metric = val_metric
        best_val_epoch       = epoch + 1
        if wandb.run is not None:
            wandb.run.summary['best_val_epoch'] = best_val_epoch

        save_dict = {'model_state_dict': model.state_dict(), 'config': config}
        if config['use_mlp']:
            save_dict['link_predictor_state_dict'] = link_predictor.state_dict()
        torch.save(save_dict, BEST_MODEL_PATH)
        print(f"\n✅ Best model (by {config['eval_metric']}) saved → {BEST_MODEL_PATH}  "
              f"val {config['eval_metric']}: {val_metric:.4f}\n")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= config['patience']:
        print(f"Early stopping triggered at epoch {epoch + 1} "
              f"(no improvement for {config['patience']} eval checks)")
        early_stop_flag = True

    return (
        best_val_eval_metric, best_val_metrics, best_test_metrics,
        best_val_epoch, epochs_without_improvement, early_stop_flag
    )


# ===========================================================================
# Main
# ===========================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config = get_config()

# ---------------------------------------------------------------------------
# System / Hardware Info (captured once at startup)
# ---------------------------------------------------------------------------
_process = psutil.Process(os.getpid())
cpu_ram_at_start_gib = _process.memory_info().rss / (1024 ** 3)

hw_info = {
    "device":              str(device),
    "cpu_physical_cores":  psutil.cpu_count(logical=False),
    "cpu_logical_cores":   psutil.cpu_count(logical=True),
    "cpu_ram_total_gib":   round(psutil.virtual_memory().total / (1024 ** 3), 2),
    "gpu_name":            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    "vram_total_gib":      round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
                           if torch.cuda.is_available() else 0.0,
}
print(f"\n{'='*55}")
print(f"  Hardware")
print(f"{'='*55}")
for k, v in hw_info.items():
    print(f"  {k:<28}: {v}")
print(f"{'='*55}\n")

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
if config['data_name'] in ['Cora', 'PubMed', 'CiteSeer']:
    dataset = Planetoid(root=config['data_root'], name=config['data_name'])
    data = dataset[0].to(device)
elif config['data_name'].startswith('TAPE'):
    data = load_graph_arxiv23(data_root=config['data_root'])
    data = data.to(device)

# Optional: DeepWalk embeddings
if config['use_deepwalk_embeds']:
    print(f"Loading DeepWalk embeddings from: {config['deepwalk_pkl_path']}")
    if not os.path.exists(config['deepwalk_pkl_path']):
        raise FileNotFoundError(f"DeepWalk PKL not found: {config['deepwalk_pkl_path']}")
    with open(config['deepwalk_pkl_path'], 'rb') as f:
        saved_data = pickle.load(f)
        deepwalk_embeddings = saved_data['data'].to(device)
    data.x = torch.cat([data.x, deepwalk_embeddings], dim=1)
    print(f"DeepWalk concatenated. New feature dim: {data.x.shape[1]}")

# Data preprocessing
if data.is_directed():
    print("Graph is directed → converting to undirected.")
    data.edge_index = to_undirected(data.edge_index)

data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
data.edge_index, _ = remove_self_loops(data.edge_index)

deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
print(f"\nGraph Statistics — {config['data_name']}:")
print(f"  Nodes: {data.num_nodes}  |  Edges: {data.edge_index.size(1)}  |  "
      f"Avg degree: {deg.mean().item():.2f}  |  Isolated: {(deg == 0).sum().item()}")

if config['data_name'] in ['TAPE', 'CiteSeer']:
    data = T.RemoveIsolatedNodes()(data)
    print(f"  Nodes after isolated removal: {data.num_nodes}")

# Optional: Laplacian PEs
if config['use_laplacian_pe']:
    print(f"Loading Laplacian PEs from: {config['laplacian_pe_path']}")
    if not os.path.exists(config['laplacian_pe_path']):
        raise FileNotFoundError(f"Laplacian PE file not found: {config['laplacian_pe_path']}")
    lap_pe = torch.load(config['laplacian_pe_path'], map_location=device, weights_only=False)
    if lap_pe.size(0) != data.x.size(0):
        print(f"  WARNING: PE size mismatch — truncating to {data.x.size(0)} nodes.")
        lap_pe = lap_pe[:data.x.size(0)]
    data.x = torch.cat([data.x, lap_pe], dim=1)
    print(f"  Laplacian PE concatenated. New feature dim: {data.x.shape[1]}")

config['emb_dim'] = data.x.size(1)
print(f"  emb_dim: {config['emb_dim']}\n")

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
if config['use_fixed_splits']:
    split_path = os.path.join(config['split_dir'], f"{config['data_name']}_fixed_split.pt")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Fixed split not found: {split_path}")
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
    val_pos_edge_index  = val_data.edge_label_index[:, val_data.edge_label == 1]
    val_neg_edge_index  = val_data.edge_label_index[:, val_data.edge_label == 0]
    test_pos_edge_index = test_data.edge_label_index[:, test_data.edge_label == 1]
    test_neg_edge_index = test_data.edge_label_index[:, test_data.edge_label == 0]
    print(f"Fixed split loaded. Train edges: {full_train_pos_edge_index.size(1)}")

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
    val_pos_edge_index  = val_data.edge_label_index[:, val_data.edge_label == 1].to(device)
    val_neg_edge_index  = val_data.edge_label_index[:, val_data.edge_label == 0].to(device)
    test_pos_edge_index = test_data.edge_label_index[:, test_data.edge_label == 1].to(device)
    test_neg_edge_index = test_data.edge_label_index[:, test_data.edge_label == 0].to(device)

# ---------------------------------------------------------------------------
# Adjacency matrix & Evaluators
# ---------------------------------------------------------------------------
evaluator_hit = Evaluator(name='ogbl-collab')
evaluator_mrr = Evaluator(name='ogbl-citation2')

nodenum = data.x.size(0)
edge_weight = torch.ones(full_train_pos_edge_index.size(1), device=full_train_pos_edge_index.device)
adj = SparseTensor.from_edge_index(full_train_pos_edge_index, edge_weight, [nodenum, nodenum])

# ---------------------------------------------------------------------------
# W&B Initialization
# ---------------------------------------------------------------------------
if config['use_deepwalk_embeds']:
    enc_tag = "deepwalk"
elif config['use_laplacian_pe']:
    enc_tag = "laplacian"
else:
    enc_tag = "none"

run_display_name = (
    f"enc-{enc_tag}_rec{config['recurrent_steps']}_"
    f"bs{config['global_batch_size']}_"
    f"muon{config['muon_max_lr']}_adam{config['adam_max_lr']}_"
    f"nw{config['num_walks']}_wl{config['walk_length']}_"
    f"seed{config['seed']}"
)
if config['train_edge_downsample_ratio'] < 1.0:
    run_display_name += f"_dws{config['train_edge_downsample_ratio']}"

predictor_tag = "MLP" if config['use_mlp'] else "DotProduct"
wandb.init(
    entity=config['wb_entity'],
    project=f"{config['data_name']}_rw-cwue_latest_dec_{config['seed']}",
    group=f"{config['data_name']} Random Walk Link Prediction {predictor_tag}",
    name=run_display_name,
    config=config
)
run_id = wandb.run.id

# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------
torch.manual_seed(config['seed'])
mup_config = MupConfig(
    init_std=config['mup_init_std'],
    mup_width_multiplier=config['mup_width_multiplier']
)

model = Transformer(
    emb_dim=config['emb_dim'],
    num_layers=config['num_layers'],
    hidden_dim=config['hidden_dim'],
    intermediate_dim=config['intermediate_dim'],
    num_heads=config['num_heads'],
    seq_len=config['walk_length'],
    num_walks=config['num_walks'],
    attn_dropout_p=config['attn_dropout'],
    ffn_dropout_p=config['ffn_dropout'],
    resid_dropout_p=config['resid_dropout'],
    drop_path_p=config['drop_path'],
    config=mup_config
).to(device)

link_predictor = None
if config['use_mlp']:
    link_predictor = LinkPredictorMLP(
        in_dim=config['hidden_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['mlp_num_layers'],
        dropout=config['mlp_dropout']
    ).to(device)
    print("Using MLP for link prediction.")
else:
    print("Using dot product for link prediction.")

num_params_transformer = sum(p.numel() for p in model.parameters())
num_params_mlp         = sum(p.numel() for p in link_predictor.parameters()) if link_predictor else 0
num_params_total       = num_params_transformer + num_params_mlp
print(f"Parameters — Transformer: {num_params_transformer:,}  |  MLP: {num_params_mlp:,}  |  Total: {num_params_total:,}")

# ---------------------------------------------------------------------------
# DataLoader & Scheduler Setup
# ---------------------------------------------------------------------------
global_batch_size      = config['global_batch_size']
full_samples_per_epoch = full_train_pos_edge_index.size(1)

if config['train_edge_downsample_ratio'] < 1.0:
    samples_per_epoch = int(full_samples_per_epoch * config['train_edge_downsample_ratio'])
    train_loader = None   # created per-epoch inside loop
else:
    samples_per_epoch = full_samples_per_epoch
    pos_edge_dataset = TensorDataset(full_train_pos_edge_index.t())
    train_loader = DataLoader(pos_edge_dataset, batch_size=global_batch_size, shuffle=True)

batches_per_epoch = math.ceil(samples_per_epoch / global_batch_size)
total_batches     = batches_per_epoch * config['num_epochs']

print(f"\nBatch size: {global_batch_size}  |  Batches/epoch: {batches_per_epoch}  |  "
      f"Total batches: {total_batches}")

warmup = total_batches // 10
cool   = total_batches - warmup

X = train_data.x.to(device).bfloat16()

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
hidden_weights      = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2  and p.requires_grad]

if config['recurrent_steps'] <= 1:
    hidden_gains_biases = [p for p in hidden_gains_biases if p is not model.norm_weight]

param_groups = [
    dict(params=hidden_weights,      use_muon=True,  lr=0.04,  weight_decay=0.01),
    dict(params=hidden_gains_biases, use_muon=False, lr=5e-5,  betas=(0.9, 0.95), weight_decay=0.0),
]
if config['use_mlp']:
    param_groups.append(
        dict(params=link_predictor.parameters(), use_muon=False,
             lr=config['mlp_lr'], betas=(0.9, 0.95), weight_decay=0.0)
    )

opt = SingleDeviceMuonWithAuxAdam(param_groups)

# ---------------------------------------------------------------------------
# Checkpoint path
# ---------------------------------------------------------------------------
save_dir = f"{config['data_name']}/checkpoints"
os.makedirs(save_dir, exist_ok=True)
BEST_MODEL_PATH = os.path.join(save_dir, f"best_model_{run_id}.pth")

# ---------------------------------------------------------------------------
# Training State
# ---------------------------------------------------------------------------
best_val_eval_metric       = 0.0
best_val_epoch             = 0
epochs_without_improvement = 0
epochs_eval_steps          = 5
best_val_metrics           = {}
best_test_metrics          = {}

# Gradient accumulation for recurrent_steps > 1 to avoid memory overflow
if config['recurrent_steps'] > 1:
    target_batch_size  = 256
    accumulation_steps = max(1, target_batch_size // global_batch_size)
    print(f"Gradient accumulation enabled: accumulation_steps = {accumulation_steps}")
else:
    accumulation_steps = 1

# ---------------------------------------------------------------------------
# Profiling State
# ---------------------------------------------------------------------------
epoch_profile_rows: list = []    # one row per epoch
batch_times_all:    list = []    # wall-clock seconds per batch (across all epochs)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)
    print("Peak vRAM tracker RESET")

run_start_time     = time.perf_counter()
cpu_ram_model_gib  = get_cpu_ram_gib()   # after model is on device

# ===========================================================================
# Training Loop
# ===========================================================================
pbar = tqdm.tqdm(total=total_batches, desc="Training")
running_loss = 0.0
muon_lr = config['muon_max_lr']  # initialise so it's always defined

for epoch in range(config['num_epochs']):
    model.train()
    if config['use_mlp']:
        link_predictor.train()

    epoch_start_time = time.perf_counter()

    # ---- Build per-epoch dataloader if downsampling ----
    if config['train_edge_downsample_ratio'] < 1.0:
        epoch_train_pos_edge_index = downsample_edges(
            full_train_pos_edge_index,
            ratio=config['train_edge_downsample_ratio'],
            seed=config['seed'] + epoch
        )
        pos_edge_dataset = TensorDataset(epoch_train_pos_edge_index.t())
        train_loader = DataLoader(pos_edge_dataset, batch_size=global_batch_size, shuffle=True)
    else:
        epoch_train_pos_edge_index = full_train_pos_edge_index

    # ---- Sample a subset of train edges for evaluation (HeART style) ----
    num_val_neg             = val_neg_edge_index.size(1)
    num_train_pos_to_sample = min(num_val_neg, epoch_train_pos_edge_index.size(1))
    torch.manual_seed(config['seed'] + epoch)
    pos_perm = torch.randperm(epoch_train_pos_edge_index.size(1), device=device)[:num_train_pos_to_sample]
    eval_train_pos_edge_index = epoch_train_pos_edge_index[:, pos_perm]

    epoch_batch_times = []

    # ---- Batch loop ----
    for batch_idx, batch_data in enumerate(train_loader):
        global_batch_idx = batch_idx + epoch * batches_per_epoch

        # Apply trapezoidal LR schedule to Muon groups
        for p in opt.param_groups:
            if p.get('use_muon', False):
                muon_lr = trapezoidal_lr_schedule(
                    global_batch_idx,
                    config['muon_max_lr'], config['muon_min_lr'],
                    warmup, cool, total_batches
                )
                p["lr"] = muon_lr

        # 1. Positive edges from DataLoader
        batch_pos_edges = batch_data[0].t().to(device)

        # 2. Negative edge sampling
        local_batch_size = batch_pos_edges.size(1)
        num_neg_samples  = int(local_batch_size * config['neg_sample_ratio'])
        batch_neg_edges  = sample_negative_edges(
            pos_edge_index=full_train_pos_edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=num_neg_samples,
            device=device
        )

        # ---- Timed section: walk sampling + forward + backward ----
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        batch_start = time.perf_counter()

        # 3. Random walk batch for all unique nodes
        all_nodes = torch.cat([
            batch_pos_edges[0], batch_pos_edges[1],
            batch_neg_edges[0], batch_neg_edges[1]
        ]).unique()

        batch, anon_indices = get_random_walk_batch(
            adj, X, all_nodes,
            walk_length=config['walk_length'],
            num_walks=config['num_walks'],
            recurrent_steps=config['recurrent_steps'],
            p=config['node2vec_p'],
            q=config['node2vec_q']
        )

        # 4. Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings = model(batch, anon_indices)
            node_to_idx = {n.item(): i for i, n in enumerate(all_nodes)}

            pos_u = embeddings[[node_to_idx[u.item()] for u in batch_pos_edges[0]]]
            pos_v = embeddings[[node_to_idx[v.item()] for v in batch_pos_edges[1]]]
            neg_u = embeddings[[node_to_idx[u.item()] for u in batch_neg_edges[0]]]
            neg_v = embeddings[[node_to_idx[v.item()] for v in batch_neg_edges[1]]]

            if config['use_mlp']:
                pos_scores = link_predictor(pos_u, pos_v)
                neg_scores = link_predictor(neg_u, neg_v).view(-1, int(config['neg_sample_ratio']))
            else:
                pos_scores = torch.sigmoid((pos_u * pos_v).sum(dim=-1))
                neg_scores = torch.sigmoid((neg_u * neg_v).sum(dim=-1)).view(-1, int(config['neg_sample_ratio']))

            loss = binary_cross_entropy_loss(pos_scores, neg_scores)

        running_loss += loss.item()
        loss = loss / accumulation_steps

        # 5. Backward
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            mlp_grad_norm = 0.0
            if config['use_mlp']:
                with torch.no_grad():
                    for p in link_predictor.parameters():
                        if p.grad is not None:
                            mlp_grad_norm += p.grad.data.norm(2).item() ** 2
                    mlp_grad_norm = mlp_grad_norm ** 0.5

            all_params_to_clip = (
                chain(model.parameters(), link_predictor.parameters())
                if config['use_mlp'] else model.parameters()
            )
            torch.nn.utils.clip_grad_norm_(all_params_to_clip, float(config['grad_clip_norm']))

            opt.step()
            opt.zero_grad(set_to_none=True)

            avg_loss = running_loss / accumulation_steps
            wandb.log(dict(loss=avg_loss, mlp_grad_norm=mlp_grad_norm,
                           step=global_batch_idx, lr=muon_lr))
            running_loss = 0.0

        # ---- Measure batch wall time ----
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        batch_elapsed = time.perf_counter() - batch_start
        epoch_batch_times.append(batch_elapsed)
        batch_times_all.append(batch_elapsed)

        # Progress bar
        if torch.cuda.is_available():
            free, total_mem = torch.cuda.mem_get_info(device)
            mem_util = (total_mem - free) / total_mem
        else:
            mem_util = psutil.virtual_memory().percent / 100.0

        real_loss_display = loss.item() * accumulation_steps
        pbar.set_description(
            f"E{epoch+1} | loss: {real_loss_display:.4f} | "
            f"mem: {mem_util:.2f} | {batch_elapsed*1000:.0f}ms/batch"
        )
        pbar.update(1)

    # NaN check
    if 'loss' in locals() and torch.isnan(loss):
        print("NaN loss detected — stopping.")
        break

    # ---- Per-epoch profiling snapshot ----
    epoch_wall_time  = time.perf_counter() - epoch_start_time
    avg_batch_ms     = (sum(epoch_batch_times) / len(epoch_batch_times)) * 1000 if epoch_batch_times else 0.0
    gpu_stats        = get_gpu_stats(device)
    cpu_ram_now_gib  = get_cpu_ram_gib()

    epoch_row = {
        "run_id":                  run_id,
        "data_name":               config['data_name'],
        "epoch":                   epoch + 1,
        "epoch_wall_time_s":       round(epoch_wall_time, 3),
        "avg_batch_time_ms":       round(avg_batch_ms, 3),
        "num_batches":             len(epoch_batch_times),
        "cpu_ram_process_gib":     round(cpu_ram_now_gib, 4),
        **{f"gpu_{k}": v for k, v in gpu_stats.items()},
    }
    epoch_profile_rows.append(epoch_row)

    # ---- Evaluation ----
    is_eval_epoch = (
        (epoch == 0) or
        ((epoch + 1) % epochs_eval_steps == 0) or
        (epoch == config['num_epochs'] - 1)
    )

    if is_eval_epoch:
        (
            best_val_eval_metric,
            best_val_metrics,
            best_test_metrics,
            best_val_epoch,
            epochs_without_improvement,
            early_stop
        ) = evaluate_and_log(
            model=model, link_predictor=link_predictor, adj=adj, X=X, config=config,
            evaluator_hit=evaluator_hit, evaluator_mrr=evaluator_mrr, device=device,
            train_pos_edge_index=eval_train_pos_edge_index,
            train_neg_edge_index=val_neg_edge_index,
            val_pos_edge_index=val_pos_edge_index,
            val_neg_edge_index=val_neg_edge_index,
            test_pos_edge_index=test_pos_edge_index,
            test_neg_edge_index=test_neg_edge_index,
            epoch=epoch,
            best_val_eval_metric=best_val_eval_metric,
            best_val_metrics=best_val_metrics,
            best_test_metrics=best_test_metrics,
            best_val_epoch=best_val_epoch,
            epochs_without_improvement=epochs_without_improvement,
            BEST_MODEL_PATH=BEST_MODEL_PATH
        )
        if early_stop:
            break

pbar.close()

# ===========================================================================
# Final Profiling Measurements
# ===========================================================================
total_wall_time_s   = time.perf_counter() - run_start_time
epochs_trained      = len(epoch_profile_rows)
avg_epoch_time_s    = total_wall_time_s / max(epochs_trained, 1)
avg_batch_time_ms   = (sum(batch_times_all) / len(batch_times_all)) * 1000 if batch_times_all else 0.0
final_gpu_stats     = get_gpu_stats(device)
final_cpu_ram_gib   = get_cpu_ram_gib()

# Peak vRAM from PyTorch's tracker (most accurate)
peak_vram_gib = (
    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    if torch.cuda.is_available() else 0.0
)

print(f"\n{'='*55}")
print(f"  FINAL PROFILING REPORT")
print(f"{'='*55}")
print(f"  Total wall time          : {total_wall_time_s:.1f}s  ({total_wall_time_s/3600:.3f} hrs)")
print(f"  Epochs trained           : {epochs_trained}")
print(f"  Avg time per epoch       : {avg_epoch_time_s:.2f}s")
print(f"  Avg time per batch       : {avg_batch_time_ms:.2f}ms")
print(f"  GPU vRAM peak (PyTorch)  : {peak_vram_gib:.4f} GiB")
print(f"  GPU vRAM allocated (end) : {final_gpu_stats['vram_allocated_gib']:.4f} GiB")
print(f"  GPU vRAM reserved  (end) : {final_gpu_stats['vram_reserved_gib']:.4f} GiB")
print(f"  CPU RAM (process, end)   : {final_cpu_ram_gib:.4f} GiB")
print(f"{'='*55}\n")

# ===========================================================================
# Save Profiling CSVs
# ===========================================================================
profiling_summary_row = {
    # --- Identity ---
    "run_id":                       run_id,
    "timestamp":                    datetime.datetime.now().isoformat(timespec='seconds'),
    "data_name":                    config['data_name'],
    "seed":                         config['seed'],
    # --- Hardware ---
    **{f"hw_{k}": v for k, v in hw_info.items()},
    # --- Model size ---
    "params_transformer":           num_params_transformer,
    "params_mlp":                   num_params_mlp,
    "params_total":                 num_params_total,
    # --- Key hyperparams (those that most affect memory/time) ---
    "hidden_dim":                   config['hidden_dim'],
    "num_layers":                   config['num_layers'],
    "num_heads":                    config['num_heads'],
    "walk_length":                  config['walk_length'],
    "num_walks":                    config['num_walks'],
    "recurrent_steps":              config['recurrent_steps'],
    "global_batch_size":            config['global_batch_size'],
    "accumulation_steps":           accumulation_steps,
    "effective_batch_size":         global_batch_size * accumulation_steps,
    # --- Training timing ---
    "epochs_trained":               epochs_trained,
    "total_wall_time_s":            round(total_wall_time_s, 3),
    "total_wall_time_hrs":          round(total_wall_time_s / 3600, 5),
    "avg_epoch_wall_time_s":        round(avg_epoch_time_s, 3),
    "avg_batch_wall_time_ms":       round(avg_batch_time_ms, 3),
    "total_batches_executed":       len(batch_times_all),
    # --- GPU memory ---
    "vram_peak_pytorch_gib":        round(peak_vram_gib, 4),
    "vram_allocated_end_gib":       final_gpu_stats["vram_allocated_gib"],
    "vram_reserved_end_gib":        final_gpu_stats["vram_reserved_gib"],
    "vram_total_gib":               final_gpu_stats["vram_total_gib"],
    "vram_peak_utilization_pct":    round(peak_vram_gib / final_gpu_stats["vram_total_gib"] * 100, 2)
                                    if final_gpu_stats["vram_total_gib"] > 0 else 0.0,
    # --- CPU memory ---
    "cpu_ram_at_start_gib":         round(cpu_ram_at_start_gib, 4),
    "cpu_ram_after_model_init_gib": round(cpu_ram_model_gib, 4),
    "cpu_ram_at_end_gib":           round(final_cpu_ram_gib, 4),
    "cpu_ram_delta_gib":            round(final_cpu_ram_gib - cpu_ram_at_start_gib, 4),
    # --- Best metrics ---
    "best_val_epoch":               best_val_epoch,
    **{f"best_val_{k}": v for k, v in best_val_metrics.items()},
    **{f"best_test_{k}": v for k, v in best_test_metrics.items()},
}

profiling_dir = os.path.join(config['data_name'], "profiling")
save_profiling_csv(
    epoch_profile_rows=epoch_profile_rows,
    summary_row=profiling_summary_row,
    output_dir=profiling_dir,
    run_id=run_id
)

# ===========================================================================
# Existing Results CSVs (unchanged)
# ===========================================================================
wandb_best_val_log  = {f'best_val_{k}': v for k, v in best_val_metrics.items()}
wandb_best_test_log = {f'best_test_{k}': v for k, v in best_test_metrics.items()}
wandb.log({**wandb_best_val_log, **wandb_best_test_log, 'best_val_epoch': best_val_epoch})

print(f"Best Validation {config['eval_metric']}: {best_val_eval_metric:.4f} at epoch {best_val_epoch}")
print(f"Best Val  [{', '.join(f'{k}: {v:.4f}' for k, v in best_val_metrics.items())}]")
print(f"Best Test [{', '.join(f'{k}: {v:.4f}' for k, v in best_test_metrics.items())}]")

results = {
    'model_data_seed': f"Transformer_{config['data_name']}_LinkPred_{config['seed']}",
    'best_val_epoch':  best_val_epoch,
    **{f'best_val_{k}': v  for k, v in best_val_metrics.items()},
    **{f'best_test_{k}': v for k, v in best_test_metrics.items()},
}

experiment_results = {
    'seed':             config['seed'],
    'use_mlp':          config['use_mlp'],
    'global_bs':        config['global_batch_size'],
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
    **{f'metrics(best_val_{k})': v  for k, v in best_val_metrics.items()},
}

os.makedirs(config['data_name'], exist_ok=True)
csv_filename = os.path.join(
    config['data_name'],
    f"Transformer_{config['data_name']}_LinkPred_{run_id}.csv"
)
pd.DataFrame([results]).to_csv(csv_filename, index=False)

summary_csv = os.path.join(config['data_name'],
                           "experiment_results_link_prediction_updated_final.csv")
results_df = pd.DataFrame([experiment_results])
if os.path.exists(summary_csv):
    results_df.to_csv(summary_csv, mode='a', header=False, index=False)
else:
    results_df.to_csv(summary_csv, mode='w', header=True, index=False)

print(f"\nSaved run results → {csv_filename}")
wandb.finish()