from rdflib import Graph
import networkx as nx
import matplotlib.pyplot as plt

# Load RDF data
rdf_graph = Graph()
rdf_data = "enriched_carshare-schema.ttl"  # Replace with your RDF content or load from a file
# Parse the Turtle file
try:
    rdf_graph.parse(rdf_data, format="turtle")
    print(f"Graph has {len(rdf_graph)} triples.")
except Exception as e:
    print(f"Error: {e}")

# Helper function to extract the concept from a URI
def extract_concept(uri):
    if "#" in uri:
        return uri.split("#")[-1]  # Get the part after the last '#'
    elif "/" in uri:
        return uri.split("/")[-1]  # Get the part after the last '/'
    return uri  # If no '#' or '/' is present, return the full URI

# Convert RDF graph to NetworkX graph with simplified labels
nx_graph = nx.DiGraph()

for s, p, o in rdf_graph:
    s_label = extract_concept(str(s))
    p_label = extract_concept(str(p))
    o_label = extract_concept(str(o))
    nx_graph.add_edge(s_label, o_label, label=p_label)

# Visualize the graph
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(nx_graph, k=0.15, iterations=20)
nx.draw(nx_graph, pos, with_labels=True, node_size=3000, font_size=10, font_weight="bold", alpha=0.7)
nx.draw_networkx_edge_labels(
    nx_graph, pos, edge_labels={(u, v): d["label"] for u, v, d in nx_graph.edges(data=True)}
)
plt.title("Simplified RDF Graph Visualization")
plt.show()

