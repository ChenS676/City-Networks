#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_cluster import random_walk as cluster_random_walk


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_undirected_edge_index(G: nx.Graph, device="cpu") -> torch.Tensor:
    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def add_self_loops_for_isolates(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        isolates = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        return torch.stack([isolates, isolates], dim=0)

    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg.scatter_add_(
        0,
        edge_index[0],
        torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device),
    )

    isolates = torch.where(deg == 0)[0]

    if isolates.numel() == 0:
        return edge_index

    loops = torch.stack([isolates, isolates], dim=0)
    return torch.cat([edge_index, loops], dim=1)


def anonymous_walk_code(walk: torch.Tensor):
    seen = {}
    code = []
    next_id = 0

    for x in walk.tolist():
        if x not in seen:
            seen[x] = next_id
            next_id += 1
        code.append(seen[x])

    return tuple(code)


def deduplicate_walks_by_anonymous_code(
    walks: torch.Tensor,
    starts: torch.Tensor | None = None,
    sources: torch.Tensor | None = None,
):
    if walks.numel() == 0:
        return walks, starts, sources

    keep = []
    seen_codes = set()

    for i, w in enumerate(walks):
        code = anonymous_walk_code(w)
        if code not in seen_codes:
            seen_codes.add(code)
            keep.append(i)

    if not keep:
        empty_walks = walks[:0]
        empty_starts = starts[:0] if starts is not None else None
        empty_sources = sources[:0] if sources is not None else None
        return empty_walks, empty_starts, empty_sources

    keep = torch.tensor(keep, dtype=torch.long, device=walks.device)

    walks = walks[keep]
    starts = starts[keep] if starts is not None else None
    sources = sources[keep] if sources is not None else None

    return walks, starts, sources


# ============================================================
# Result containers
# ============================================================

@dataclass
class HRWResult:
    walks_by_level: list
    all_walks: torch.Tensor
    starts_by_level: list
    all_starts: torch.Tensor
    sources_by_level: list
    all_sources: torch.Tensor
    budget_transitions: int


@dataclass
class RWResult:
    walks: torch.Tensor
    starts: torch.Tensor
    sources: torch.Tensor
    budget_transitions: int
    walk_length: int


# ============================================================
# Hierarchical random walk with source lineage
# ============================================================

@torch.no_grad()
def hierarchical_random_walk_fast(
    edge_index: torch.Tensor,
    start_nodes: torch.Tensor,
    R: int,
    L: int,
    M: int,
    p: float = 1.0,
    q: float = 1.0,
    num_nodes: int | None = None,
    merge_terminal: bool = False,
    dedup_anonymous: bool = False,
):
    """
    Hierarchical random walk.

    At each level:
        - start M short walks from every frontier node
        - collect terminal nodes
        - use terminals as next-level frontier

    Additionally tracks the original source seed for every sampled walk.
    """

    row, col = edge_index[0], edge_index[1]

    frontier = start_nodes.long()
    frontier_sources = start_nodes.long()

    walks_by_level = []
    starts_by_level = []
    sources_by_level = []

    for level in range(R + 1):
        if frontier.numel() == 0:
            break

        starts = frontier.repeat_interleave(M)
        sources = frontier_sources.repeat_interleave(M)

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

        if dedup_anonymous:
            walks, starts, sources = deduplicate_walks_by_anonymous_code(
                walks=walks,
                starts=starts,
                sources=sources,
            )

        walks_by_level.append(walks)
        starts_by_level.append(starts)
        sources_by_level.append(sources)

        terminals = walks[:, -1]

        if merge_terminal:
            # Important:
            # torch.unique changes the sampling distribution.
            # We keep the first source associated with each unique terminal.
            # For clean stochastic lineage analysis, merge_terminal=False is preferred.
            unique_frontier, inverse = torch.unique(terminals, sorted=False, return_inverse=True)

            unique_sources = []
            for i in range(unique_frontier.numel()):
                idx = torch.where(inverse == i)[0][0]
                unique_sources.append(sources[idx])

            frontier = unique_frontier
            frontier_sources = torch.stack(unique_sources).to(edge_index.device)
        else:
            frontier = terminals
            frontier_sources = sources

    if walks_by_level:
        all_walks = torch.cat(walks_by_level, dim=0)
        all_starts = torch.cat(starts_by_level, dim=0)
        all_sources = torch.cat(sources_by_level, dim=0)
    else:
        all_walks = torch.empty((0, L + 1), dtype=torch.long, device=edge_index.device)
        all_starts = torch.empty((0,), dtype=torch.long, device=edge_index.device)
        all_sources = torch.empty((0,), dtype=torch.long, device=edge_index.device)

    budget_transitions = int(all_walks.size(0) * L)

    return HRWResult(
        walks_by_level=walks_by_level,
        all_walks=all_walks,
        starts_by_level=starts_by_level,
        all_starts=all_starts,
        sources_by_level=sources_by_level,
        all_sources=all_sources,
        budget_transitions=budget_transitions,
    )


# ============================================================
# Standard random walk baselines
# ============================================================

@torch.no_grad()
def standard_random_walk_budget_matched(
    edge_index: torch.Tensor,
    start_nodes: torch.Tensor,
    total_budget_transitions: int,
    walk_length: int,
    p: float = 1.0,
    q: float = 1.0,
    num_nodes: int | None = None,
):
    """
    Samples standard random walks under the same transition budget.

    If total_budget_transitions is not divisible by walk_length,
    one shorter remainder walk is sampled and padded by repeating its terminal node.
    """

    row, col = edge_index[0], edge_index[1]
    T = int(walk_length)

    if T <= 0:
        raise ValueError("walk_length must be positive.")

    K = total_budget_transitions // T
    rem = total_budget_transitions % T

    walks = []
    starts_out = []
    sources_out = []

    if K > 0:
        starts_full = start_nodes.repeat(math.ceil(K / len(start_nodes)))[:K]
        sources_full = starts_full.clone()

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
        sources_out.append(sources_full)

    if rem > 0:
        start_rem = start_nodes[(K % len(start_nodes)):(K % len(start_nodes)) + 1]
        source_rem = start_rem.clone()

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
        sources_out.append(source_rem)

    if not walks:
        out = torch.empty((0, T + 1), dtype=torch.long, device=edge_index.device)
        starts = torch.empty((0,), dtype=torch.long, device=edge_index.device)
        sources = torch.empty((0,), dtype=torch.long, device=edge_index.device)
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
        sources = torch.cat(sources_out, dim=0)

    return RWResult(
        walks=out,
        starts=starts,
        sources=sources,
        budget_transitions=total_budget_transitions,
        walk_length=T,
    )


# ============================================================
# Metrics
# ============================================================

def visited_nodes(walks: torch.Tensor):
    if walks.numel() == 0:
        return set()
    return set(torch.unique(walks.reshape(-1)).cpu().tolist())


def coverage_ratio(num_nodes: int, walks: torch.Tensor) -> float:
    if walks.numel() == 0:
        return 0.0
    return float(len(visited_nodes(walks))) / float(num_nodes)


def cut_crossing_probability(walks: torch.Tensor, S_cut: set[int]) -> float:
    """
    Segment-level cut-crossing probability.

    Measures:
        fraction of walk segments that cross the cut at least once.

    This metric is sensitive to walk length.
    """

    if walks.numel() == 0:
        return 0.0

    S = set(S_cut)
    cnt = 0

    for w in walks.tolist():
        crossed = False
        for a, b in zip(w[:-1], w[1:]):
            if (a in S) != (b in S):
                crossed = True
                break
        cnt += int(crossed)

    return cnt / walks.size(0)


def cut_crossing_count(walks: torch.Tensor, S_cut: set[int]) -> int:
    """
    Number of walk segments that cross the cut at least once.
    """

    if walks.numel() == 0:
        return 0

    S = set(S_cut)
    cnt = 0

    for w in walks.tolist():
        crossed = False
        for a, b in zip(w[:-1], w[1:]):
            if (a in S) != (b in S):
                crossed = True
                break
        cnt += int(crossed)

    return cnt


def cut_crossing_transition_rate(walks: torch.Tensor, S_cut: set[int]) -> float:
    """
    Transition-level cut-crossing rate.

    Measures:
        number of cut-crossing transitions / total transitions.

    This is more budget-faithful than segment-level crossing probability.
    """

    if walks.numel() == 0 or walks.size(1) <= 1:
        return 0.0

    S = set(S_cut)
    total = 0
    crossed = 0

    for w in walks.tolist():
        for a, b in zip(w[:-1], w[1:]):
            total += 1
            crossed += int((a in S) != (b in S))

    return crossed / total if total > 0 else 0.0


def source_cut_reachability(
    walks: torch.Tensor,
    sources: torch.Tensor,
    original_seeds: torch.Tensor,
    S_cut: set[int],
) -> float:
    """
    Source-level reachability.

    Measures:
        fraction of original seed nodes whose descendant walks ever reach the opposite side.

    This is especially meaningful for HRW.
    """

    if walks.numel() == 0:
        return 0.0

    S = set(S_cut)
    reached = {int(s): False for s in original_seeds.tolist()}

    for src, w in zip(sources.tolist(), walks):
        src = int(src)
        if src not in reached:
            continue

        src_side = src in S

        for x in w.tolist():
            if (int(x) in S) != src_side:
                reached[src] = True
                break

    return sum(reached.values()) / len(reached) if reached else 0.0


def source_average_opposite_side_coverage(
    walks: torch.Tensor,
    sources: torch.Tensor,
    original_seeds: torch.Tensor,
    S_cut: set[int],
) -> float:
    """
    For each source seed, compute how many unique opposite-side nodes are reached.
    Then average over source seeds.

    This gives a softer alternative to binary source reachability.
    """

    if walks.numel() == 0:
        return 0.0

    S = set(S_cut)
    source_to_nodes = {int(s): set() for s in original_seeds.tolist()}

    for src, w in zip(sources.tolist(), walks):
        src = int(src)
        if src not in source_to_nodes:
            continue

        src_side = src in S

        for x in w.tolist():
            x = int(x)
            if (x in S) != src_side:
                source_to_nodes[src].add(x)

    if not source_to_nodes:
        return 0.0

    return float(np.mean([len(v) for v in source_to_nodes.values()]))


def earliest_access_time_from_walks(walks: torch.Tensor, target: int):
    best = float("inf")

    for w in walks:
        hits = (w == target).nonzero(as_tuple=False)
        if hits.numel() > 0:
            best = min(best, int(hits[0].item()))

    return best


def finite_mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("inf")


def grouped_walks_by_source(walks: torch.Tensor, sources: torch.Tensor):
    groups = defaultdict(list)

    for s, w in zip(sources.tolist(), walks):
        groups[int(s)].append(w)

    return {k: torch.stack(v, dim=0) for k, v in groups.items()}


def source_level_mean_eat(
    walks: torch.Tensor,
    sources: torch.Tensor,
    seed_target_pairs: list[tuple[int, int]],
):
    """
    Source-aware EAT.

    For each source-target pair, use all descendant walks of that source.
    """

    by_src = grouped_walks_by_source(walks, sources)
    eats = []

    for src, tgt in seed_target_pairs:
        if src in by_src:
            eats.append(earliest_access_time_from_walks(by_src[src], tgt))

    return finite_mean(eats)


# ============================================================
# Graph builders
# ============================================================

def build_barbell_graph(left_size=20, bridge_len=2):
    G = nx.barbell_graph(left_size, bridge_len)
    S_cut = set(range(left_size))
    return G, S_cut


def build_sbm_graph(sizes=(50, 50), p_in=0.12, p_out=0.01, seed=0):
    probs = [[p_in, p_out], [p_out, p_in]]
    G = nx.stochastic_block_model(list(sizes), probs, seed=seed)
    G = nx.convert_node_labels_to_integers(G)
    S_cut = set(range(sizes[0]))
    return G, S_cut


def build_lollipop_graph(clique_size=20, path_len=30):
    G = nx.lollipop_graph(clique_size, path_len)
    S_cut = set(range(clique_size))
    return G, S_cut


def build_grid_graph(m=10, n=10):
    G = nx.grid_2d_graph(m, n)
    G = nx.convert_node_labels_to_integers(G)

    left_half = set()
    for idx in range(G.number_of_nodes()):
        col = idx % n
        if col < n // 2:
            left_half.add(idx)

    return G, left_half


def build_random_regular_graph(num_nodes=100, degree=4, seed=0):
    G = nx.random_regular_graph(degree, num_nodes, seed=seed)
    S_cut = set(range(num_nodes // 2))
    return G, S_cut


def build_er_graph(num_nodes=100, p=0.05, seed=0):
    G = nx.erdos_renyi_graph(num_nodes, p, seed=seed)

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    S_cut = set(range(G.number_of_nodes() // 2))
    return G, S_cut


def build_caveman_graph(num_cliques=5, clique_size=20):
    G = nx.connected_caveman_graph(num_cliques, clique_size)
    S_cut = set(range(clique_size))
    return G, S_cut


def build_watts_strogatz_graph(num_nodes=100, k=6, p=0.05, seed=0):
    G = nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    S_cut = set(range(G.number_of_nodes() // 2))
    return G, S_cut


# ============================================================
# Experiment
# ============================================================

@torch.no_grad()
def summarize_walk_method(
    name: str,
    num_nodes: int,
    walks: torch.Tensor,
    sources: torch.Tensor,
    original_seeds: torch.Tensor,
    S_cut: set[int],
    seed_target_pairs: list[tuple[int, int]],
    budget_transitions: int,
):
    return {
        f"{name}_num_walks": int(walks.size(0)),
        f"{name}_budget_transitions": int(budget_transitions),
        f"{name}_coverage": coverage_ratio(num_nodes, walks),
        f"{name}_cut_cross_prob_segment": cut_crossing_probability(walks, S_cut),
        f"{name}_cut_cross_count_segment": cut_crossing_count(walks, S_cut),
        f"{name}_cut_cross_rate_transition": cut_crossing_transition_rate(walks, S_cut),
        f"{name}_source_cut_reachability": source_cut_reachability(
            walks=walks,
            sources=sources,
            original_seeds=original_seeds,
            S_cut=S_cut,
        ),
        f"{name}_source_avg_opposite_coverage": source_average_opposite_side_coverage(
            walks=walks,
            sources=sources,
            original_seeds=original_seeds,
            S_cut=S_cut,
        ),
        f"{name}_source_level_mean_eat": source_level_mean_eat(
            walks=walks,
            sources=sources,
            seed_target_pairs=seed_target_pairs,
        ),
        f"{name}_visited": visited_nodes(walks),
    }


@torch.no_grad()
def run_experiment(
    G: nx.Graph,
    S_cut: set[int],
    R=3,
    L=8,
    M=3,
    p=1.0,
    q=1.0,
    num_seed_nodes=20,
    num_pairs=20,
    merge_terminal=False,
    dedup_anonymous=False,
    device="cpu",
):
    num_nodes = G.number_of_nodes()

    edge_index = to_undirected_edge_index(G, device=device)
    edge_index = add_self_loops_for_isolates(edge_index, num_nodes)

    left = sorted(list(S_cut))
    right = sorted(list(set(G.nodes()) - S_cut))

    num_seed_nodes = min(num_seed_nodes, len(left))
    num_pairs = min(num_pairs, num_seed_nodes, len(right))

    seeds = torch.tensor(
        random.sample(left, num_seed_nodes),
        dtype=torch.long,
        device=device,
    )

    targets = random.sample(right, num_pairs)
    seed_target_pairs = list(zip(seeds[:num_pairs].cpu().tolist(), targets))

    # ------------------------------
    # HRW
    # ------------------------------
    hrw = hierarchical_random_walk_fast(
        edge_index=edge_index,
        start_nodes=seeds,
        R=R,
        L=L,
        M=M,
        p=p,
        q=q,
        num_nodes=num_nodes,
        merge_terminal=merge_terminal,
        dedup_anonymous=dedup_anonymous,
    )

    total_budget = hrw.budget_transitions

    # ------------------------------
    # RW-long baseline
    # length = (R + 1) * L
    # ------------------------------
    long_walk_length = (R + 1) * L

    rw_long = standard_random_walk_budget_matched(
        edge_index=edge_index,
        start_nodes=seeds,
        total_budget_transitions=total_budget,
        walk_length=long_walk_length,
        p=p,
        q=q,
        num_nodes=num_nodes,
    )

    # ------------------------------
    # RW-local baseline
    # length = L
    # ------------------------------
    rw_local = standard_random_walk_budget_matched(
        edge_index=edge_index,
        start_nodes=seeds,
        total_budget_transitions=total_budget,
        walk_length=L,
        p=p,
        q=q,
        num_nodes=num_nodes,
    )

    results = {
        "num_nodes": num_nodes,
        "num_edges": G.number_of_edges(),
        "R": R,
        "L": L,
        "M": M,
        "p": p,
        "q": q,
        "merge_terminal": merge_terminal,
        "dedup_anonymous": dedup_anonymous,
        "matched_budget_transitions": total_budget,
        "seeds": seeds.cpu().tolist(),
        "targets": targets,
    }

    results.update(
        summarize_walk_method(
            name="hrw",
            num_nodes=num_nodes,
            walks=hrw.all_walks,
            sources=hrw.all_sources,
            original_seeds=seeds,
            S_cut=S_cut,
            seed_target_pairs=seed_target_pairs,
            budget_transitions=hrw.budget_transitions,
        )
    )

    results.update(
        summarize_walk_method(
            name="rw_long",
            num_nodes=num_nodes,
            walks=rw_long.walks,
            sources=rw_long.sources,
            original_seeds=seeds,
            S_cut=S_cut,
            seed_target_pairs=seed_target_pairs,
            budget_transitions=rw_long.budget_transitions,
        )
    )

    results.update(
        summarize_walk_method(
            name="rw_local",
            num_nodes=num_nodes,
            walks=rw_local.walks,
            sources=rw_local.sources,
            original_seeds=seeds,
            S_cut=S_cut,
            seed_target_pairs=seed_target_pairs,
            budget_transitions=rw_local.budget_transitions,
        )
    )

    return results, hrw, rw_long, rw_local


# ============================================================
# Visualization
# ============================================================

def visualize_results(G, S_cut, results, title_prefix=""):
    Path("plots").mkdir(parents=True, exist_ok=True)

    pos = nx.spring_layout(G, seed=0)

    hrw_visited = results["hrw_visited"]
    rw_long_visited = results["rw_long_visited"]
    rw_local_visited = results["rw_local_visited"]

    seeds = set(results["seeds"])
    targets = set(results["targets"])

    labels = ["RW-long", "RW-local", "HRW"]

    coverage_values = [
        results["rw_long_coverage"],
        results["rw_local_coverage"],
        results["hrw_coverage"],
    ]

    segment_cut_values = [
        results["rw_long_cut_cross_prob_segment"],
        results["rw_local_cut_cross_prob_segment"],
        results["hrw_cut_cross_prob_segment"],
    ]

    transition_cut_values = [
        results["rw_long_cut_cross_rate_transition"],
        results["rw_local_cut_cross_rate_transition"],
        results["hrw_cut_cross_rate_transition"],
    ]

    source_reach_values = [
        results["rw_long_source_cut_reachability"],
        results["rw_local_source_cut_reachability"],
        results["hrw_source_cut_reachability"],
    ]

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(labels, coverage_values)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Coverage Ratio")
    ax1.set_ylabel("Fraction of visited nodes")

    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(labels, segment_cut_values)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Segment-Level Cut-Crossing Probability")
    ax2.set_ylabel("Fraction of walks crossing cut")

    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(labels, transition_cut_values)
    ax3.set_ylim(0, 1.0)
    ax3.set_title("Transition-Level Cut-Crossing Rate")
    ax3.set_ylabel("Crossing transitions / total transitions")

    ax4 = plt.subplot(2, 3, 4)
    ax4.bar(labels, source_reach_values)
    ax4.set_ylim(0, 1.0)
    ax4.set_title("Source-Level Cut Reachability")
    ax4.set_ylabel("Fraction of seeds reaching opposite side")

    ax5 = plt.subplot(2, 3, 5)
    eat_values = [
        results["rw_long_source_level_mean_eat"],
        results["rw_local_source_level_mean_eat"],
        results["hrw_source_level_mean_eat"],
    ]
    ax5.bar(labels, eat_values)
    ax5.set_title("Source-Level Mean EAT")
    ax5.set_ylabel("Steps within sampled segments")

    ax6 = plt.subplot(2, 3, 6)

    node_colors = []
    for v in G.nodes():
        if v in seeds:
            node_colors.append("tab:red")
        elif v in targets:
            node_colors.append("tab:green")
        elif v in hrw_visited and v in rw_long_visited:
            node_colors.append("tab:purple")
        elif v in hrw_visited:
            node_colors.append("tab:blue")
        elif v in rw_long_visited:
            node_colors.append("tab:orange")
        elif v in rw_local_visited:
            node_colors.append("tab:brown")
        elif v in S_cut:
            node_colors.append("lightblue")
        else:
            node_colors.append("lightgray")

    nx.draw(
        G,
        pos,
        node_size=80,
        node_color=node_colors,
        with_labels=False,
        ax=ax6,
    )

    ax6.set_title("Visited Nodes: HRW blue, RW-long orange, both purple")

    plt.suptitle(f"{title_prefix} Budget-Matched RW vs HRW", fontsize=14)
    plt.tight_layout()

    out_path = Path(f"plots/{title_prefix}_rw_vs_hrw_full_metrics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ============================================================
# Printing
# ============================================================

def print_results(results):
    print("\n=== Budget-Matched HRW vs RW ===\n")

    general_keys = [
        "num_nodes",
        "num_edges",
        "R",
        "L",
        "M",
        "p",
        "q",
        "merge_terminal",
        "dedup_anonymous",
        "matched_budget_transitions",
    ]

    print("[General]")
    for k in general_keys:
        print(f"{k}: {results[k]}")

    methods = ["rw_long", "rw_local", "hrw"]

    metric_suffixes = [
        "num_walks",
        "budget_transitions",
        "coverage",
        "cut_cross_prob_segment",
        "cut_cross_count_segment",
        "cut_cross_rate_transition",
        "source_cut_reachability",
        "source_avg_opposite_coverage",
        "source_level_mean_eat",
    ]

    print("\n[Metrics]")
    for method in methods:
        print(f"\n--- {method} ---")
        for suffix in metric_suffixes:
            k = f"{method}_{suffix}"
            print(f"{k}: {results[k]}")

    print("\n[Interpretation hints]")
    print("rw_long uses walk length (R + 1) * L.")
    print("rw_local uses walk length L.")
    print("hrw uses multi-level length-L walks with lineage tracking.")
    print("segment-level cut probability is walk-length sensitive.")
    print("transition-level cut rate is more budget-faithful.")
    print("source-level cut reachability is usually the most meaningful HRW metric.")


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph",
        type=str,
        default="barbell",
        choices=[
            "barbell",
            "sbm",
            "lollipop",
            "grid",
            "regular",
            "er",
            "caveman",
            "ws",
        ],
    )

    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--M", type=int, default=3)

    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)

    parser.add_argument("--num_seed_nodes", type=int, default=20)
    parser.add_argument("--num_pairs", type=int, default=20)

    parser.add_argument("--merge_terminal", action="store_true")
    parser.add_argument("--dedup_anonymous", action="store_true")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.graph == "barbell":
        G, S_cut = build_barbell_graph(left_size=20, bridge_len=2)

    elif args.graph == "sbm":
        G, S_cut = build_sbm_graph(
            sizes=(50, 50),
            p_in=0.12,
            p_out=0.01,
            seed=args.seed,
        )

    elif args.graph == "lollipop":
        G, S_cut = build_lollipop_graph(clique_size=20, path_len=30)

    elif args.graph == "grid":
        G, S_cut = build_grid_graph(m=10, n=10)

    elif args.graph == "regular":
        G, S_cut = build_random_regular_graph(
            num_nodes=100,
            degree=4,
            seed=args.seed,
        )

    elif args.graph == "er":
        G, S_cut = build_er_graph(
            num_nodes=100,
            p=0.05,
            seed=args.seed,
        )

    elif args.graph == "caveman":
        G, S_cut = build_caveman_graph(
            num_cliques=5,
            clique_size=20,
        )

    elif args.graph == "ws":
        G, S_cut = build_watts_strogatz_graph(
            num_nodes=100,
            k=6,
            p=0.05,
            seed=args.seed,
        )

    else:
        raise ValueError(f"Unknown graph type: {args.graph}")

    results, hrw, rw_long, rw_local = run_experiment(
        G=G,
        S_cut=S_cut,
        R=args.R,
        L=args.L,
        M=args.M,
        p=args.p,
        q=args.q,
        num_seed_nodes=args.num_seed_nodes,
        num_pairs=args.num_pairs,
        merge_terminal=args.merge_terminal,
        dedup_anonymous=args.dedup_anonymous,
        device=args.device,
    )

    print_results(results)

    out_path = visualize_results(
        G=G,
        S_cut=S_cut,
        results=results,
        title_prefix=args.graph,
    )

    print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    main()