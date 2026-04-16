import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

# works for 40, 80, 120 particularly well on 120
def visualize_structural_backbone(num_nodes=80, recurrent_steps=3):
    # 1. Generate a Base Graph (Watts-Strogatz for local clustering + some long-range shortcuts)
    G = nx.watts_strogatz_graph(num_nodes, k=4, p=0.1)
    pos = nx.spring_layout(G, k=0.12, seed=42)
    seed_node = 0
    
    # Discovery tracking
    node_discovery = {seed_node: 0}
    edge_discovery = {} # (u, v) -> step_index
    
    current_seeds = {seed_node}
    walk_len, num_walks = 8, 2
    
    # 2. Simulate Recursive HRW
    for k in range(1, recurrent_steps + 1):
        next_seeds = set()
        for s in current_seeds:
            for _ in range(num_walks):
                curr = s
                for _ in range(walk_len):
                    neighbors = list(G.neighbors(curr))
                    nxt = np.random.choice(neighbors)
                    
                    # Mark edge as captured in this level
                    e = tuple(sorted((curr, nxt)))
                    if e not in edge_discovery:
                        edge_discovery[e] = k
                    
                    if nxt not in node_discovery:
                        node_discovery[nxt] = k
                    curr = nxt
                next_seeds.add(curr)
        current_seeds = next_seeds

    # 3. Plotting
    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    # --- BACKGROUND: The "Original Connection" ---
    # We draw all edges in G that were NOT captured by the walks
    all_edges = set(tuple(sorted(e)) for e in G.edges())
    captured_edges = set(edge_discovery.keys())
    uncaptured_edges = all_edges - captured_edges
    
    nx.draw_networkx_edges(G, pos, edgelist=list(uncaptured_edges), 
                           width=0.6, edge_color='grey', alpha=0.9, label="Original Adjacency (Unsampled)")
    nx.draw_networkx_nodes(G, pos, node_size=15, node_color='#dfe6e9', alpha=0.3)

    # --- FOREGROUND: Hierarchical Induced Adjacency ---
    level_colors = {1: '#3498db', 2: '#2ecc71', 3: '#9b59b6'}
    level_names = {1: "Local Context (L1)", 2: "Expansion (L2)", 3: "Global Bridge (L3)"}

    for k in range(recurrent_steps, 0, -1):
        # Nodes and edges at this specific hierarchy level
        nodes_k = [n for n, lev in node_discovery.items() if lev <= k and lev > 0]
        edges_k = [e for e, lev in edge_discovery.items() if lev <= k]
        c = level_colors[k]
        
        # Draw bold edges for the captured structure
        nx.draw_networkx_edges(G, pos, edgelist=edges_k, width=2.0, 
                               edge_color=c, alpha=0.7, label=f"Induced Edges {level_names[k]}")
        
        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_k, node_size=60, 
                               node_color=c, label=f"Nodes {level_names[k]}")
        
        # Add the "Bubble" contour
        if len(nodes_k) > 2:
            pts = np.array([pos[n] for n in nodes_k])
            hull = ConvexHull(pts)
            poly = plt.Polygon(pts[hull.vertices], color=c, alpha=0.04, ec=c, lw=2, ls='--')
            ax.add_patch(poly)

    # Highlight Seed
    nx.draw_networkx_nodes(G, pos, nodelist=[seed_node], node_size=200, 
                           node_color='#ff4757', edgecolors='black', label="Seed $s_0$")

    plt.title("HWIS Induced Adjacency over Original Graph Structure", fontsize=16)
    
    # Legend management
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('nn.png')

visualize_structural_backbone()