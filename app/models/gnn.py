from rdflib import Graph, URIRef
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from itertools import product  # Used for cross-graph candidate link generation


# Function to convert RDF graph to PyTorch Geometric graph
def rdf_to_graph(graph, global_nodes=None):
    edges = []
    if global_nodes is None:
        global_nodes = {}
    idx = len(global_nodes)  # Start with existing node indices

    for subj, pred, obj in graph:
        if subj not in global_nodes:
            global_nodes[subj] = idx
            idx += 1
        if obj not in global_nodes:
            global_nodes[obj] = idx
            idx += 1

        edges.append([global_nodes[subj], global_nodes[obj]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create reverse mapping (index to node)
    index_to_node = {v: k for k, v in global_nodes.items()}

    return edge_index, global_nodes, index_to_node


# Define the GNN Model
class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.dropout = Dropout(p=0.5)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        # Dot-product-based link prediction
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return self.decode(z, data.edge_index)


# Load RDF datasets
g1 = Graph()
g2 = Graph()

# Parse the RDF files
g1.parse("carshare-schema.ttl", format="turtle")
g2.parse("dynamic_data.ttl", format="turtle")

# Convert RDF graphs to PyTorch Geometric graphs
global_nodes = {}
edge_index1, global_nodes, index_to_node = rdf_to_graph(g1, global_nodes)
edge_index2, global_nodes, index_to_node = rdf_to_graph(g2, global_nodes)

# Combine the graphs
edge_index_combined = torch.cat([edge_index1, edge_index2], dim=1)
num_nodes = len(global_nodes)
x_combined = torch.eye(num_nodes, dtype=torch.float)  # Node features
combined_data = Data(x=x_combined, edge_index=edge_index_combined)

# Split the data into train/val/test
splitter = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True)
train_data, val_data, test_data = splitter(combined_data)

# Initialize the GNN model, optimizer, and loss function
model = SimpleGNN(in_channels=num_nodes, hidden_channels=16, out_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()


# Training function
def train():
    model.train()
    optimizer.zero_grad()

    # Encode node embeddings
    z = model.encode(train_data.x, train_data.edge_index)

    # Decode link probabilities
    edge_label_index = train_data.edge_label_index
    edge_labels = train_data.edge_label

    preds = model.decode(z, edge_label_index)  # Predicted logits
    loss = criterion(preds, edge_labels)  # Compute loss
    loss.backward()
    optimizer.step()
    return loss.item()


# Training loop
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

# Generate cross-graph candidate links
nodes_in_g1 = [node for node in global_nodes.keys() if "carshare" in str(node)]
nodes_in_g2 = [node for node in global_nodes.keys() if "dbpedia" in str(node)]
g1_indices = [global_nodes[node] for node in nodes_in_g1]
g2_indices = [global_nodes[node] for node in nodes_in_g2]

# Create candidate links between all pairs of nodes in g1 and g2
candidate_links = list(product(g1_indices, g2_indices))
candidate_edge_index = torch.tensor(candidate_links, dtype=torch.long).t().contiguous()

# Predict probabilities for cross-graph candidate links
model.eval()
with torch.no_grad():
    z = model.encode(combined_data.x, combined_data.edge_index)
    preds = model.decode(z, candidate_edge_index).sigmoid()

# Filter high-probability links
threshold = 0.95
high_probability_indices = preds > threshold
high_probability_links = candidate_edge_index[:, high_probability_indices]

# Print high-probability predicted links
print("\nHigh-Probability Cross-Graph Links (Probability > 95%):")
for i, (src, dst) in enumerate(high_probability_links.t().tolist()):
    src_name = index_to_node[src]
    dst_name = index_to_node[dst]
    probability = preds[high_probability_indices][i].item()  # Use filtered probabilities
    print(f"Predicted Link: {src_name} -> {dst_name}, Probability: {probability:.4f}")

# Define the predictedLink predicate
predicted_link_predicate = URIRef("http://example.org/predictedLink")

# Add high-probability links to the carshare-schema graph
for src, dst in high_probability_links.t().tolist():
    g1.add((index_to_node[src], predicted_link_predicate, index_to_node[dst]))

# Save the enriched graph
g1.serialize("enriched_carshare-schema.ttl", format="turtle")

print("\nEnriched graph saved as 'enriched_carshare-schema.ttl'")
