import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

def visualize_comparison(num_nodes=120, recurrent_steps=3):
    # 1. Generate a Base Graph
    G = nx.watts_strogatz_graph(num_nodes, k=4, p=0.1)
    pos = nx.spring_layout(G, k=0.15, seed=42)
    seed_node = 0
    
    # Discovery tracking for HWIS (Recurrent Walks)
    node_discovery_walk = {seed_node: 0}
    edge_discovery_walk = {} 
    current_seeds = {seed_node}
    walk_len, num_walks = 8, 4

    num_edges = G.number_of_edges()
    avg_d = 2 * num_edges / num_nodes
    # Small-world diameter approximation: ln(N) / ln(avg_degree)
    num_walks = int(avg_d)+1 
    walk_len = int(nx.diameter(G)/recurrent_steps) # replace it with graph diameter

    for k in range(1, recurrent_steps + 1):
        next_seeds = set()
        for s in current_seeds:
            for _ in range(num_walks):
                curr = s
                for _ in range(walk_len):
                    neighbors = list(G.neighbors(curr))
                    nxt = np.random.choice(neighbors)
                    e = tuple(sorted((curr, nxt)))
                    if e not in edge_discovery_walk:
                        edge_discovery_walk[e] = k
                    if nxt not in node_discovery_walk:
                        node_discovery_walk[nxt] = k
                    curr = nxt
                next_seeds.add(curr)
        current_seeds = next_seeds

    # Discovery tracking for K-Order Neighbors (Shortest Path)
    # path_lengths is a dict: {node: distance_from_seed}
    path_lengths = nx.single_source_shortest_path_length(G, seed_node)
    
    # 2. Plotting Setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    level_colors = {1: '#3498db', 2: '#2ecc71', 3: '#9b59b6'}
    
    def draw_structure(ax, node_data, edge_filter_func, title):
        # Background
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, edge_color='grey', alpha=0.3)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=15, node_color='#dfe6e9', alpha=0.3)
        
        for k in range(recurrent_steps, 0, -1):
            c = level_colors[k]
            # Filter nodes/edges based on the specific logic (walk vs distance)
            nodes_k = [n for n, dist in node_data.items() if 0 < dist <= k]
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes_k, node_size=70, node_color=c)
            
            # Draw edges that connect nodes within this distance/hierarchy
            current_edges = [e for e in G.edges() if edge_filter_func(e, k)]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=current_edges, width=1.5, edge_color=c, alpha=0.6)
            
            # Convex Hull "Bubble"
            if len(nodes_k) > 2:
                pts = np.array([pos[n] for n in nodes_k])
                hull = ConvexHull(pts)
                poly = plt.Polygon(pts[hull.vertices], color=c, alpha=0.05, ec=c, lw=1.5, ls='--')
                ax.add_patch(poly)

        # Seed node
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[seed_node], node_size=180, node_color='#ff4757', edgecolors='black')
        ax.set_title(title, fontsize=16)
        ax.axis('off')

    # --- Plot 1: HWIS (Recurrent Walk) ---
    def walk_edge_filter(e, k):
        return edge_discovery_walk.get(tuple(sorted(e)), 99) <= k
    print(f"node discovery walk: {node_discovery_walk}")
    draw_structure(ax1, node_discovery_walk, walk_edge_filter, "HWIS Induced Adjacency (Recurrent Walks)")

    # --- Plot 2: K-Order Neighbors ---
    def dist_edge_filter(e, k):
        # An edge is part of the k-order subgraph if both endpoints are within distance k
        u, v = e
        return path_lengths.get(u, 99) <= k and path_lengths.get(v, 99) <= k
    print(f"path lengths: {dist_edge_filter}")
    draw_structure(ax2, path_lengths, dist_edge_filter, f"{recurrent_steps}-Order Neighbors (Shortest Path)")

    plt.tight_layout()
    plt.savefig('visual_mpnn.png')

visualize_comparison()