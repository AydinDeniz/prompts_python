
import networkx as nx
import matplotlib.pyplot as plt

# Generate a sample graph
def create_graph():
    G = nx.Graph()
    edges = [(1, 2, 4), (1, 3, 2), (2, 3, 1), (2, 4, 5), (3, 4, 8), (4, 5, 6)]
    G.add_weighted_edges_from(edges)
    return G

# Visualize graph
def draw_graph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Graph Visualization")
    plt.show()

# Dijkstra's algorithm visualization
def dijkstra_shortest_path(G, start, end):
    path = nx.shortest_path(G, source=start, target=end, weight='weight')
    print(f"Shortest path from {start} to {end}: {path}")

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color='gray', node_size=2000, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    path_edges = list(zip(path, path[1:]))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=12)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    plt.title("Dijkstra's Shortest Path")
    plt.show()

if __name__ == "__main__":
    G = create_graph()
    draw_graph(G)
    dijkstra_shortest_path(G, 1, 5)
