from rdflib import Graph, URIRef
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from itertools import combinations


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


# Function to evaluate link prediction with node names
def evaluate_link_prediction(data, index_to_node):
    model.eval()

    with torch.no_grad():
        # Encode node embeddings
        z = model.encode(data.x, data.edge_index)

        # Decode link probabilities
        edge_label_index = data.edge_label_index
        edge_labels = data.edge_label  # True labels (1 for positive, 0 for negative)
        preds = model.decode(z, edge_label_index).sigmoid()  # Predicted probabilities

        # Print link probabilities and their corresponding true labels with node names
        print("Predicted Links:")
        for i, (src, dst) in enumerate(edge_label_index.t().tolist()):
            src_name = index_to_node[src]  # Get the source node name
            dst_name = index_to_node[dst]  # Get the destination node name
            print(f"Link ({src_name} -> {dst_name}): Predicted probability = {preds[i]:.4f}, True label = {edge_labels[i].item()}")

        # Compute accuracy
        acc = ((preds > 0.5) == edge_labels).sum().item() / edge_labels.size(0)
        return acc


# Load RDF datasets
g1 = Graph()
g2 = Graph()

# Parse the RDF files
g1.parse("carshare-schema.ttl", format="turtle")
g2.parse("open_data_car_properties.ttl", format="turtle")

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
        print("\nValidation Links:")
        val_acc = evaluate_link_prediction(val_data, index_to_node)
        print(f"Validation Accuracy: {val_acc:.4f}")

# Test the model
print("\nTest Links:")
test_acc = evaluate_link_prediction(test_data, index_to_node)
print(f"Test Accuracy: {test_acc:.4f}")


# Generate candidate links for carshare-schema
nodes_in_carshare = [node for node in global_nodes.keys() if "carshare" in str(node)]
carshare_indices = [global_nodes[node] for node in nodes_in_carshare]

# Generate all possible pairs of nodes in carshare-schema
candidate_links = list(combinations(carshare_indices, 2))
candidate_edge_index = torch.tensor(candidate_links, dtype=torch.long).t().contiguous()

# Predict probabilities for candidate links
model.eval()
with torch.no_grad():
    z = model.encode(combined_data.x, combined_data.edge_index)
    preds = model.decode(z, candidate_edge_index).sigmoid()

# Filter predictions above a certain threshold
threshold = 0.9
high_probability_indices = preds > threshold
high_probability_links = candidate_edge_index[:, high_probability_indices]

# Print high-probability predicted links
print("\nHigh-Probability Missing Links (Probability > 90%):")
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
