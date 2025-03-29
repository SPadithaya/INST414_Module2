import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("influencers_data.csv", low_memory=False)

print("DataFrame Sample:")
print(df.head())

df['connections'] = df['connections'].replace(r'\+','', regex=True)
df['connections'] = pd.to_numeric(df['connections'], errors='coerce')
df['connections'].fillna(0, inplace=True)

df = df.sort_values(by=["connections", "reactions"], ascending=False).head(10)

print("Top 10 Influencers (after sorting):")
print(df[['name', 'connections', 'reactions']])

G = nx.Graph()

df['normalized_name'] = df['name'].str.lower().str.strip()

for _, row in df.iterrows():
    G.add_node(row['normalized_name'], followers=row['followers'], connections=row['connections'], industry=row.get('industry', 'Unknown'))

print("Nodes added to graph:", list(G.nodes))

for i, row1 in df.iterrows():
    for j, row2 in df.iterrows():
        if i != j:
            shared_engagement = abs(row1['reactions'] - row2['reactions']) + abs(row1['comments'] - row2['comments'])
            if shared_engagement > 0:
                G.add_edge(row1['normalized_name'], row2['normalized_name'], weight=shared_engagement)

print("Total edges added:", G.number_of_edges())

pagerank = nx.pagerank(G)
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

important_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:3]

def plot_graph(G):
    plt.figure(figsize=(12, 12))
    pos = nx.circular_layout(G)
    node_sizes = [G.nodes[n]['followers'] / 10 for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', edgecolors='black')
    edges = [(u, v) for (u, v, d) in G.edges(data=True)]
    weights = [d['weight'] / 100 for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='gray', alpha=0.7)
    labels = {n: f"{n}\n({G.nodes[n]['connections']} conn.)" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    plt.title("LinkedIn Influencer Network Graph (Based on Engagement)")
    plt.show()

if len(G.nodes) > 0 and len(G.edges) > 0:
    plot_graph(G)
else:
    print("No nodes or edges to plot.")

print("Total nodes:", G.number_of_nodes())
print("Total edges:", G.number_of_edges())
print("Top 3 Most Influential Users:")
for user in important_nodes:
    print(f"{user} - Centrality: {pagerank[user]:.4f}")