"""
precompute_lap_pe.py
Pre-computes Laplacian PE eigenvectors for all graphs in a graphbench
dataset and saves them to disk. Run this ONCE before training.

Usage:
    python precompute_lap_pe.py --dataset_name maxclique_easy --lap_pe_dim 16
    python precompute_lap_pe.py --dataset_name maxclique_hard --lap_pe_dim 16
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
import graphbench
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected, is_undirected
from tqdm import tqdm

_original_torch_load = torch.load
def torch_load_no_weights_only(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_no_weights_only


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


def cache_path(data_root, dataset_name, split, lap_pe_dim):
    return os.path.join(data_root, f"{dataset_name}_{split}_lapPE{lap_pe_dim}.pt")


def precompute_split(dataset, split_name, cache_file, lap_pe_dim):
    if os.path.exists(cache_file):
        print(f"  [{split_name}] Cache exists → {cache_file}  (skipping)")
        return

    print(f"  [{split_name}] Computing LapPE for {len(dataset)} graphs ...")
    all_eigvecs = []
    all_eigvals = []
    t0 = time.time()

    for i, data in enumerate(tqdm(dataset, desc=split_name, ncols=80)):
        eigvals, eigvecs = compute_laplacian_eigen(
            data.edge_index, data.num_nodes, lap_pe_dim
        )
        all_eigvecs.append(eigvecs)   # [N_i, k]
        all_eigvals.append(eigvals)   # [N_i, k]

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(dataset) - i - 1) / rate
            print(f"    {i+1}/{len(dataset)}  {rate:.1f} graphs/s  ETA {eta:.0f}s")

    torch.save({"eigvecs": all_eigvecs, "eigvals": all_eigvals}, cache_file)
    elapsed = time.time() - t0
    print(f"  [{split_name}] Done in {elapsed:.1f}s → {cache_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="maxclique_easy")
    parser.add_argument("--data_root",    type=str, default="./data_graphbench")
    parser.add_argument("--lap_pe_dim",   type=int, default=16)
    args = parser.parse_args()

    loader  = graphbench.Loader(
        root=args.data_root,
        dataset_names=args.dataset_name,
        transform=T.Compose([AddUndirectedContext()])
    )
    dataset = loader.load()

    print(f"Dataset: {args.dataset_name}  lap_pe_dim={args.lap_pe_dim}")
    print(f"Raw dataset type: {type(dataset)}  len: {len(dataset)}")

    # Inspect structure to find splits robustly
    splits = {}
    try:
        # Case 1: list of dicts  e.g. [{train: ..., valid: ..., test: ...}]
        if isinstance(dataset, (list, tuple)) and len(dataset) > 0:
            first = dataset[0]
            if isinstance(first, dict):
                splits = {k: first[k] for k in first if k in ('train', 'valid', 'test')}
            else:
                # Case 2: list of Data objects — no splits
                splits = {"all": dataset}
        # Case 3: dict directly
        elif isinstance(dataset, dict):
            splits = {k: dataset[k] for k in dataset if k in ('train', 'valid', 'test')}
        else:
            splits = {"all": dataset}

        if not splits:
            splits = {"all": dataset}

    except Exception as e:
        print(f"  Could not parse splits ({e}), treating as single split.")
        splits = {"all": dataset}

    print(f"Splits found: { {k: len(v) for k, v in splits.items()} }")

    for split_name, split_data in splits.items():
        cf = cache_path(args.data_root, args.dataset_name,
                        split_name, args.lap_pe_dim)
        precompute_split(split_data, split_name, cf, args.lap_pe_dim)

    print("\nDone. Run baselines.py with --use_cached_lap_pe true")

    print("\nDone. Run baselines.py with --use_cached_lap_pe true")


if __name__ == "__main__":
    main()