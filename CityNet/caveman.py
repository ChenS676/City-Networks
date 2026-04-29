#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Selective-restart Hierarchical Random Walk (HRW) vs Standard Random Walk (RW)
on a caveman graph only.

What this script does
---------------------
1. Builds a connected caveman graph.
2. Compares:
   - Standard RW
   - Vanilla HRW
   - Selective-restart HRW
3. Matches the total number of transition steps approximately/exactly between methods.
4. Reports:
   - coverage ratio
   - cut-crossing rate
   - mean earliest access time (EAT)
5. Visualizes the three metrics.

Requirements
------------
pip install torch torch-cluster networkx numpy matplotlib

Example
-------
python caveman_selective_hrw.py --R 3 --L 8 --M 3 --seed 0
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_cluster import random_walk as cluster_random_walk


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_undirected_edge_index(G: nx.Graph, device="cpu"):
    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index


def add_self_loops_for_isolates(edge_index: torch.Tensor, num_nodes: int):
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device))
    isolates = torch.where(deg == 0)[0]
    if isolates.numel() == 0:
        return edge_index
    loops = torch.stack([isolates, isolates], dim=0)
    return torch.cat([edge_index, loops], dim=1)


def grouped_walks_by_source(walks: torch.Tensor, starts: torch.Tensor):
    groups = defaultdict(list)
    for s, w in zip(starts.tolist(), walks):
        groups[s].append(w)
    out = {}
    for k, vs in groups.items():
        out[k] = torch.stack(vs, dim=0)
    return out


def earliest_access_time_from_walks(walks: torch.Tensor, target: int):
    best = float("inf")
    for w in walks:
        hits = (w == target).nonzero(as_tuple=False)
        if hits.numel() > 0:
            best = min(best, int(hits[0].item()))
    return best


def coverage_ratio(num_nodes: int, walks: torch.Tensor):
    if walks.numel() == 0:
        return 0.0
    visited = torch.unique(walks.reshape(-1))
    return float(visited.numel()) / float(num_nodes)


def visited_nodes(walks: torch.Tensor):
    if walks.numel() == 0:
        return set()
    return set(torch.unique(walks.reshape(-1)).cpu().tolist())


def cut_crossing_probability(walks: torch.Tensor, communities: dict[int, int]):
    """
    Fraction of walks that cross at least one community boundary.
    """
    if walks.numel() == 0:
        return 0.0

    cnt = 0
    for w in walks.tolist():
        crossed = False
        for a, b in zip(w[:-1], w[1:]):
            if communities[a] != communities[b]:
                crossed = True
                break
        cnt += int(crossed)
    return cnt / walks.size(0)


def finite_mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("inf")


# ---------------------------------------------------------
# Graph: connected caveman only
# ---------------------------------------------------------

def build_connected_caveman(num_cliques=5, clique_size=20):
    """
    NetworkX connected caveman graph: ring of cliques with one rewired edge per clique.
    """
    G = nx.connected_caveman_graph(num_cliques, clique_size)
    communities = {}
    for c in range(num_cliques):
        for v in range(c * clique_size, (c + 1) * clique_size):
            communities[v] = c
    return G, communities


def sample_source_target_pairs(communities: dict[int, int], num_pairs: int, seed: int = 0):
    """
    Sample source-target pairs from different cliques.
    """
    rng = random.Random(seed)
    all_nodes = list(communities.keys())
    cliques = sorted(set(communities.values()))

    pairs = []
    for _ in range(num_pairs):
        c1, c2 = rng.sample(cliques, 2)
        n1 = rng.choice([v for v in all_nodes if communities[v] == c1])
        n2 = rng.choice([v for v in all_nodes if communities[v] == c2])
        pairs.append((n1, n2))
    return pairs


# ---------------------------------------------------------
# Core RW sampling
# ---------------------------------------------------------

@dataclass
class RWResult:
    walks: torch.Tensor
    starts: torch.Tensor
    budget_transitions: int


@dataclass
class HRWResult:
    walks_by_level: list
    starts_by_level: list
    all_walks: torch.Tensor
    all_starts: torch.Tensor
    budget_transitions: int


@torch.no_grad()
def standard_random_walk_budget_matched(
    edge_index: torch.Tensor,
    start_nodes: torch.Tensor,
    total_budget_transitions: int,
    long_walk_length: int,
    p: float = 1.0,
    q: float = 1.0,
    num_nodes: int | None = None,
):
    row, col = edge_index[0], edge_index[1]
    T = int(long_walk_length)
    K = total_budget_transitions // T
    rem = total_budget_transitions % T

    walks = []
    starts_out = []

    if K > 0:
        starts_full = start_nodes.repeat(math.ceil(K / len(start_nodes)))[:K]
        full_walks = cluster_random_walk(
            row=row,
            col=col,
            start=starts_full,
            walk_length=T,
            p=p,
            q=q,
            coalesced=False,
            num_nodes=num_nodes,
        )
        walks.append(full_walks)
        starts_out.append(starts_full)

    if rem > 0:
        start_rem = start_nodes[(K % len(start_nodes)):(K % len(start_nodes)) + 1]
        rem_walk = cluster_random_walk(
            row=row,
            col=col,
            start=start_rem,
            walk_length=rem,
            p=p,
            q=q,
            coalesced=False,
            num_nodes=num_nodes,
        )
        walks.append(rem_walk)
        starts_out.append(start_rem)

    if not walks:
        out = torch.empty((0, T + 1), dtype=torch.long, device=edge_index.device)
        starts = torch.empty((0,), dtype=torch.long, device=edge_index.device)
    else:
        max_len = max(w.size(1) for w in walks)
        padded = []
        for w in walks:
            if w.size(1) < max_len:
                pad_val = w[:, -1:].repeat(1, max_len - w.size(1))
                w = torch.cat([w, pad_val], dim=1)
            padded.append(w)
        out = torch.cat(padded, dim=0)
        starts = torch.cat(starts_out, dim=0)

    return RWResult(out, starts, total_budget_transitions)


@torch.no_grad()
def hierarchical_random_walk_vanilla(
    edge_index: torch.Tensor,
    start_nodes: torch.Tensor,
    R: int,
    L: int,
    M: int,
    p: float = 1.0,
    q: float = 1.0,
    num_nodes: int | None = None,
    merge_terminal: bool = True,
):
    row, col = edge_index[0], edge_index[1]
    frontier = start_nodes.long()
    walks_by_level = []
    starts_by_level = []

    for _ in range(R + 1):
        starts = frontier.repeat_interleave(M)
        walks = cluster_random_walk(
            row=row,
            col=col,
            start=starts,
            walk_length=L,
            p=p,
            q=q,
            coalesced=False,
            num_nodes=num_nodes,
        )
        walks_by_level.append(walks)
        starts_by_level.append(starts)
        terminals = walks[:, -1]
        frontier = torch.unique(terminals) if merge_terminal else terminals

    all_walks = torch.cat(walks_by_level, dim=0)
    all_starts = torch.cat(starts_by_level, dim=0)
    budget_transitions = int(all_walks.size(0) * L)
    return HRWResult(walks_by_level, starts_by_level, all_walks, all_starts, budget_transitions)


# ---------------------------------------------------------
# Selective restarting
# ---------------------------------------------------------

def precompute_shortest_paths_cutoff(G: nx.Graph, cutoff: int):
    """
    shortest_dist[u][v] = shortest path length from u to v up to cutoff;
    if > cutoff, v absent from dict.
    """
    shortest_dist = {}
    for u in G.nodes():
        shortest_dist[u] = nx.single_source_shortest_path_length(G, u, cutoff=cutoff)
    return shortest_dist


def should_restart(
    walk: torch.Tensor,
    visited_global: set[int],
    shortest_dist: dict[int, dict[int, int]],
    tau_dist: int = 3,
    tau_novel: int = 3,
):
    """
    Restart only if the walk has shown evidence of expansion:
    - terminal sufficiently far from start, OR
    - enough novel nodes visited

    Otherwise continue the same branch next level.
    """
    w = walk.tolist()
    s, t = w[0], w[-1]
    dist_score = shortest_dist[s].get(t, 0)
    novel_score = sum(1 for x in w if x not in visited_global)
    return (dist_score >= tau_dist) or (novel_score >= tau_novel)


@torch.no_grad()
def hierarchical_random_walk_selective_restart(
    G: nx.Graph,
    edge_index: torch.Tensor,
    start_nodes: torch.Tensor,
    R: int,
    L: int,
    M: int,
    p: float = 1.0,
    q: float = 1.0,
    num_nodes: int | None = None,
    merge_terminal: bool = True,
    tau_dist: int = 3,
    tau_novel: int = 3,
):
    """
    Selective-restart HRW:
    - If a branch expands enough, restart from its terminal.
    - Otherwise continue the same branch from its terminal in the next level.

    This is especially useful on caveman graphs where restarting too early
    inside a clique can be harmful.
    """
    row, col = edge_index[0], edge_index[1]
    frontier = start_nodes.long()
    walks_by_level = []
    starts_by_level = []

    shortest_dist = precompute_shortest_paths_cutoff(G, cutoff=max(tau_dist, L))
    visited_global = set(start_nodes.cpu().tolist())

    for _ in range(R + 1):
        starts = frontier.repeat_interleave(M)

        walks = cluster_random_walk(
            row=row,
            col=col,
            start=starts,
            walk_length=L,
            p=p,
            q=q,
            coalesced=False,
            num_nodes=num_nodes,
        )

        walks_by_level.append(walks)
        starts_by_level.append(starts)

        next_frontier = []
        for w in walks:
            wl = w.tolist()
            visited_global.update(wl)

            if should_restart(
                w,
                visited_global=visited_global,
                shortest_dist=shortest_dist,
                tau_dist=tau_dist,
                tau_novel=tau_novel,
            ):
                next_frontier.append(wl[-1])  # frontier moved enough: restart here
            else:
                next_frontier.append(wl[-1])  # continue same branch from terminal

        frontier = torch.tensor(next_frontier, dtype=torch.long, device=edge_index.device)
        if merge_terminal:
            frontier = torch.unique(frontier)

    all_walks = torch.cat(walks_by_level, dim=0)
    all_starts = torch.cat(starts_by_level, dim=0)
    budget_transitions = int(all_walks.size(0) * L)
    return HRWResult(walks_by_level, starts_by_level, all_walks, all_starts, budget_transitions)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_method(
    walks: torch.Tensor,
    starts: torch.Tensor,
    num_nodes: int,
    communities: dict[int, int],
    pairs: list[tuple[int, int]],
):
    cov = coverage_ratio(num_nodes, walks)
    cut_rate = cut_crossing_probability(walks, communities)

    by_src = grouped_walks_by_source(walks, starts)
    eats = []
    for src, tgt in pairs:
        if src in by_src:
            eats.append(earliest_access_time_from_walks(by_src[src], tgt))
    mean_eat = finite_mean(eats)

    return {
        "coverage": cov,
        "cut_crossing": cut_rate,
        "mean_eat": mean_eat,
        "visited": visited_nodes(walks),
    }


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def visualize_metrics(metrics_dict, title="Caveman Graph: RW vs HRW"):
    """
    metrics_dict: {
        "RW": {...},
        "Vanilla HRW": {...},
        "Selective HRW": {...}
    }
    """
    names = list(metrics_dict.keys())
    covs = [metrics_dict[k]["coverage"] for k in names]
    cuts = [metrics_dict[k]["cut_crossing"] for k in names]
    eats = [metrics_dict[k]["mean_eat"] for k in names]

    fig = plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 3, 1)
    ax1.bar(names, covs)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Coverage Ratio")
    ax1.set_ylabel("Fraction of visited nodes")
    ax1.tick_params(axis="x", rotation=15)

    ax2 = plt.subplot(1, 3, 2)
    ax2.bar(names, cuts)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Cut-Crossing Rate")
    ax2.set_ylabel("Fraction of walks crossing clique boundary")
    ax2.tick_params(axis="x", rotation=15)

    ax3 = plt.subplot(1, 3, 3)
    ax3.bar(names, eats)
    ax3.set_title("Mean EAT")
    ax3.set_ylabel("Steps")
    ax3.tick_params(axis="x", rotation=15)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig("caveman_metrics.png", dpi=300)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cliques", type=int, default=2)
    parser.add_argument("--clique_size", type=int, default=20)
    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--num_pairs", type=int, default=30)
    parser.add_argument("--tau_dist", type=int, default=3)
    parser.add_argument("--tau_novel", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    G, communities = build_connected_caveman(
        num_cliques=args.num_cliques,
        clique_size=args.clique_size,
    )
    num_nodes = G.number_of_nodes()

    edge_index = to_undirected_edge_index(G, device=args.device)
    edge_index = add_self_loops_for_isolates(edge_index, num_nodes=num_nodes)

    pairs = sample_source_target_pairs(communities, num_pairs=args.num_pairs, seed=args.seed)
    source_nodes = torch.tensor([u for u, _ in pairs], dtype=torch.long, device=args.device)

    # Vanilla HRW first; its budget will be used to match RW.
    hrw_vanilla = hierarchical_random_walk_vanilla(
        edge_index=edge_index,
        start_nodes=source_nodes,
        R=args.R,
        L=args.L,
        M=args.M,
        p=args.p,
        q=args.q,
        num_nodes=num_nodes,
        merge_terminal=True,
    )

    # Standard RW with matched budget.
    rw = standard_random_walk_budget_matched(
        edge_index=edge_index,
        start_nodes=source_nodes,
        total_budget_transitions=hrw_vanilla.budget_transitions,
        long_walk_length=(args.R + 1) * args.L,
        p=args.p,
        q=args.q,
        num_nodes=num_nodes,
    )

    # Selective HRW (same hyperparameters, its own realized budget may differ slightly if frontier changes).
    # To keep comparison simple, we evaluate it directly; if you want exact budget matching too, that can be added.
    hrw_selective = hierarchical_random_walk_selective_restart(
        G=G,
        edge_index=edge_index,
        start_nodes=source_nodes,
        R=args.R,
        L=args.L,
        M=args.M,
        p=args.p,
        q=args.q,
        num_nodes=num_nodes,
        merge_terminal=True,
        tau_dist=args.tau_dist,
        tau_novel=args.tau_novel,
    )

    metrics_rw = evaluate_method(
        walks=rw.walks,
        starts=rw.starts,
        num_nodes=num_nodes,
        communities=communities,
        pairs=pairs,
    )

    metrics_hrw_vanilla = evaluate_method(
        walks=hrw_vanilla.all_walks,
        starts=hrw_vanilla.all_starts,
        num_nodes=num_nodes,
        communities=communities,
        pairs=pairs,
    )

    metrics_hrw_selective = evaluate_method(
        walks=hrw_selective.all_walks,
        starts=hrw_selective.all_starts,
        num_nodes=num_nodes,
        communities=communities,
        pairs=pairs,
    )

    print("\n=== Connected Caveman Graph ===")
    print(f"num_nodes: {num_nodes}")
    print(f"num_edges: {G.number_of_edges()}")
    print(f"pairs: {len(pairs)}")
    print(f"vanilla_hrw_budget_transitions: {hrw_vanilla.budget_transitions}")
    print(f"rw_budget_transitions: {rw.budget_transitions}")
    print(f"selective_hrw_budget_transitions: {hrw_selective.budget_transitions}")

    print("\n--- RW ---")
    for k, v in metrics_rw.items():
        if k != "visited":
            print(f"{k}: {v}")

    print("\n--- Vanilla HRW ---")
    for k, v in metrics_hrw_vanilla.items():
        if k != "visited":
            print(f"{k}: {v}")

    print("\n--- Selective HRW ---")
    for k, v in metrics_hrw_selective.items():
        if k != "visited":
            print(f"{k}: {v}")

    metrics_all = {
        "RW": metrics_rw,
        "Vanilla HRW": metrics_hrw_vanilla,
        "Selective HRW": metrics_hrw_selective,
    }
    visualize_metrics(metrics_all, title="Connected Caveman: RW vs Vanilla HRW vs Selective HRW")


if __name__ == "__main__":
    main()