import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import graphbench
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx, is_undirected, to_undirected


# ── Same transform used in training ───────────────────────────────────────
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


def load_dataset(dataset_name, data_root):
    loader  = graphbench.Loader(
        root=data_root,
        dataset_names=dataset_name,
        transform=T.Compose([AddUndirectedContext()])
    )
    dataset = loader.load()
    try:
        train_dataset = dataset[0]['train']
        val_dataset   = dataset[0]['valid']
        test_dataset  = dataset[0]['test']
    except (TypeError, KeyError):
        train_dataset = val_dataset = test_dataset = dataset
    return train_dataset, val_dataset, test_dataset


# ── Per-graph visualisation ────────────────────────────────────────────────
def visualize_sample(dataset, idx=0, save_path=None):
    data   = dataset[idx]
    G      = to_networkx(data, to_undirected=True)
    labels = data.y.numpy() if data.y is not None else None

    color_map = [
        "crimson"   if labels is not None and labels[n] == 1
        else "steelblue"
        for n in G.nodes()
    ]

    n_clique = int((labels == 1).sum()) if labels is not None else "?"
    title    = (f"Graph {idx}  |  nodes={G.number_of_nodes()}  "
                f"edges={G.number_of_edges()}  clique_nodes={n_clique}")

    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos,
            node_color=color_map, node_size=80,
            with_labels=False, edge_color="#ccc", width=0.6)
    plt.title(title, fontsize=9)
    plt.tight_layout()

    path = save_path or f"graph_{idx}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved → {path}")


# ── Dataset-level statistics ───────────────────────────────────────────────
def visualize_stats(train_dataset, val_dataset, test_dataset, max_samples=500):
    def collect(ds, n):
        node_counts, avg_degrees, clique_ratios = [], [], []
        for i in range(min(n, len(ds))):
            d = ds[i]
            num_nodes = d.num_nodes
            num_edges = d.edge_index.size(1)
            node_counts.append(num_nodes)
            avg_degrees.append(num_edges / max(num_nodes, 1))
            if d.y is not None:
                clique_ratios.append(float((d.y == 1).sum()) / max(num_nodes, 1))
        return node_counts, avg_degrees, clique_ratios

    tr_n, tr_d, tr_r = collect(train_dataset, max_samples)
    va_n, va_d, va_r = collect(val_dataset,   max_samples)
    te_n, te_d, te_r = collect(test_dataset,  max_samples)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, data, title, xlabel in zip(
        axes,
        [(tr_n, va_n, te_n), (tr_d, va_d, te_d), (tr_r, va_r, te_r)],
        ["Node count", "Avg degree", "Clique node ratio"],
        ["Nodes", "Avg degree", "Ratio"],
    ):
        ax.hist(data[0], bins=30, alpha=0.6, label="train", color="steelblue", edgecolor="white")
        ax.hist(data[1], bins=30, alpha=0.6, label="val",   color="salmon",    edgecolor="white")
        ax.hist(data[2], bins=30, alpha=0.6, label="test",  color="seagreen",  edgecolor="white")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.legend(fontsize=7)

    plt.suptitle("Dataset Statistics", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("dataset_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved → dataset_stats.png")

    # Print summary table
    print(f"\n{'Split':<8} {'Graphs':>8} {'Nodes μ':>10} {'Nodes σ':>10} "
          f"{'Deg μ':>8} {'Clique%':>9}")
    print("-" * 57)
    for name, ns, ds, rs in [("train", tr_n, tr_d, tr_r),
                               ("val",   va_n, va_d, va_r),
                               ("test",  te_n, te_d, te_r)]:
        print(f"{name:<8} {len(ns):>8} {np.mean(ns):>10.1f} {np.std(ns):>10.1f} "
              f"{np.mean(ds):>8.2f} {np.mean(rs)*100:>8.1f}%")


# ── Entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="maxclique_easy")
    parser.add_argument("--data_root",    type=str, default="./data_graphbench")
    parser.add_argument("--num_samples",  type=int, default=6,
                        help="Number of individual graphs to plot")
    parser.add_argument("--max_stats",    type=int, default=500,
                        help="Max graphs to include in stat histograms")
    args = parser.parse_args()

    print(f"Loading {args.dataset_name} from {args.data_root} ...")
    train_dataset, val_dataset, test_dataset = load_dataset(
        args.dataset_name, args.data_root
    )
    print(f"Train={len(train_dataset)}  Val={len(val_dataset)}  Test={len(test_dataset)}")

    # Individual graph plots
    print(f"\nPlotting {args.num_samples} training graphs ...")
    for i in range(min(args.num_samples, len(train_dataset))):
        visualize_sample(train_dataset, idx=i, save_path=f"graph_train_{i}.png")

    # Stats across all splits
    print("\nComputing dataset statistics ...")
    visualize_stats(train_dataset, val_dataset, test_dataset,
                    max_samples=args.max_stats)


if __name__ == "__main__":
    main()