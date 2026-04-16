import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

def visualize_comparison(num_nodes=100, recurrent_steps=3):
    # 1. Generate a Base Graph (Watts-Strogatz for local clustering + shortcuts)
    G = nx.watts_strogatz_graph(num_nodes, k=6, p=0.1)
    pos = nx.spring_layout(G, k=0.15, seed=42)
    seed_node = 0
    
    # --- HEURISTIC ESTIMATION ---
    num_edges = G.number_of_edges()
    avg_d = 2 * num_edges / num_nodes
    # Small-world diameter approximation: ln(N) / ln(avg_degree)
    log_avg_d = np.log(avg_d) if avg_d > 1 else 0.5
    approx_diam = nx.diameter(G)# replace it with graph diameter
    
    # walk_len (l): 1.5x factor to account for random walk back-tracking
    walk_len = max(1, int(1.5 * approx_diam / recurrent_steps))
    # num_walks (m): scaling with sqrt of degree and its entropy
    num_walks = max(1, int(np.sqrt(avg_d) * np.log(avg_d + 1)))
    
    print(f"Graph Metrics: Avg Degree={avg_d:.2f}, Approx Diam={approx_diam:.2f}")
    print(f"Heuristics: Walk Length (l)={walk_len}, Num Walks (m)={num_walks}")

    # 2. Discovery tracking for HWIS (Recurrent Walks)
    node_discovery_walk = {seed_node: 0}
    edge_discovery_walk = {} 
    current_seeds = {seed_node}
    
    for k in range(1, recurrent_steps + 1):
        next_seeds = set()
        for s in current_seeds:
            for _ in range(num_walks):
                curr = s
                for _ in range(walk_len):
                    neighbors = list(G.neighbors(curr))
                    if not neighbors: break
                    nxt = np.random.choice(neighbors)
                    e = tuple(sorted((curr, nxt)))
                    if e not in edge_discovery_walk:
                        edge_discovery_walk[e] = k
                    if nxt not in node_discovery_walk:
                        node_discovery_walk[nxt] = k
                    curr = nxt
                next_seeds.add(curr)
        current_seeds = next_seeds

    # 3. Ground Truth: Geodesic Neighbors (Shortest Path)
    path_lengths = nx.single_source_shortest_path_length(G, seed_node)
    
    # 4. Plotting Setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    level_colors = {1: '#3498db', 2: '#2ecc71', 3: '#9b59b6'}
    
    def draw_structure(ax, node_data, edge_filter_func, title):
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, edge_color='grey', alpha=0.15)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=10, node_color='#dfe6e9', alpha=0.2)
        
        for k in range(recurrent_steps, 0, -1):
            c = level_colors[k]
            nodes_k = [n for n, dist in node_data.items() if 0 < dist <= k]
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes_k, node_size=60, node_color=c)
            
            current_edges = [e for e in G.edges() if edge_filter_func(e, k)]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=current_edges, width=1.5, edge_color=c, alpha=0.5)
            
            if len(nodes_k) > 2:
                pts = np.array([pos[n] for n in nodes_k])
                hull = ConvexHull(pts)
                poly = plt.Polygon(pts[hull.vertices], color=c, alpha=0.04, ec=c, lw=1.5, ls='--')
                ax.add_patch(poly)

        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[seed_node], node_size=150, node_color='#ff4757', edgecolors='black')
        ax.set_title(title, fontsize=16)
        ax.axis('off')

    # Filters
    walk_edge_filter = lambda e, k: edge_discovery_walk.get(tuple(sorted(e)), 99) <= k
    dist_edge_filter = lambda e, k: path_lengths.get(e[0], 99) <= k and path_lengths.get(e[1], 99) <= k

    draw_structure(ax1, node_discovery_walk, walk_edge_filter, f"HWIS: l={walk_len}, m={num_walks}\n(Stochastic Expansion)")
    draw_structure(ax2, path_lengths, dist_edge_filter, f"{recurrent_steps}-Order Neighbors\n(Deterministic Message Passing Limit)")

    plt.suptitle(f"Long-Range Capture Analysis ($N={num_nodes}$)", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('dynamic_hwis_comparison.png')
    plt.show()

visualize_comparison()