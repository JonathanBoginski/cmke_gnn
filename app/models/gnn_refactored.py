import logging
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from rdflib import Graph, URIRef
from itertools import combinations
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)

# Paths and parameters
SCHEMA_PATH = "carshare-schema.ttl"
PROPERTIES_PATH = "open_data_car_properties.ttl"
ENRICHED_GRAPH_PATH = "enriched_carshare-schema.ttl"
NUM_EPOCHS = 100
THRESHOLD = 0.9

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
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

def rdf_to_graph(graph, global_nodes=None):
    edges = []
    if global_nodes is None:
        global_nodes = {}
    idx = len(global_nodes)

    for subj, pred, obj in graph:
        if subj not in global_nodes:
            global_nodes[subj] = idx
            idx += 1
        if obj not in global_nodes:
            global_nodes[obj] = idx
            idx += 1

        edges.append([global_nodes[subj], global_nodes[obj]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    index_to_node = {v: k for k, v in global_nodes.items()}

    return edge_index, global_nodes, index_to_node

def evaluate_link_prediction(data, model, index_to_node):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        edge_label_index = data.edge_label_index
        edge_labels = data.edge_label
        preds = model.decode(z, edge_label_index).sigmoid()

        logging.info("Predicted Links:")
        for i, (src, dst) in enumerate(edge_label_index.t().tolist()):
            src_name = index_to_node[src]
            dst_name = index_to_node[dst]
            logging.debug(f"Link ({src_name} -> {dst_name}): Predicted probability = {preds[i]:.4f}, True label = {edge_labels[i].item()}")

        acc = ((preds > 0.5) == edge_labels).sum().item() / edge_labels.size(0)
        return acc

def load_graphs(schema_path, properties_path):
    logging.info("Loading RDF datasets...")
    g1 = Graph()
    g2 = Graph()
    g1.parse(schema_path, format="turtle")
    g2.parse(properties_path, format="turtle")
    return g1, g2

def prepare_graphs(g1, g2):
    logging.info("Converting RDF graphs to PyTorch Geometric format...")
    global_nodes = {}
    edge_index1, global_nodes, index_to_node = rdf_to_graph(g1, global_nodes)
    edge_index2, global_nodes, index_to_node = rdf_to_graph(g2, global_nodes)
    edge_index_combined = torch.cat([edge_index1, edge_index2], dim=1)
    return edge_index_combined, global_nodes, index_to_node

def create_data_object(edge_index_combined, num_nodes):
    logging.info("Creating PyTorch Geometric Data object...")
    x_combined = torch.eye(num_nodes, dtype=torch.float)
    return Data(x=x_combined, edge_index=edge_index_combined)

def train_model(data, num_epochs, index_to_node):
    logging.info("Initializing GNN model...")
    model = SimpleGNN(in_channels=data.num_nodes, hidden_channels=16, out_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    splitter = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True)
    train_data, val_data, test_data = splitter(data)

    def train_one_epoch():
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        preds = model.decode(z, train_data.edge_label_index)
        loss = criterion(preds, train_data.edge_label)
        loss.backward()
        optimizer.step()
        return loss.item()

    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch()
        if epoch % 10 == 0:
            val_acc = evaluate_link_prediction(val_data, model, index_to_node)
            logging.info(f"Epoch {epoch}: Loss = {loss:.4f}, Validation Accuracy = {val_acc:.4f}")

    return model

def enrich_graph(model, data, global_nodes, index_to_node, g1, threshold, output_path):
    logging.info("Predicting missing links...")
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        all_candidates = torch.tensor(list(combinations(range(len(global_nodes)), 2)), dtype=torch.long).t()
        predictions = model.decode(z, all_candidates).sigmoid()

    high_prob_links = all_candidates[:, predictions > threshold]
    predicted_link_predicate = URIRef("http://example.org/predictedLink")

    for src, dst in high_prob_links.t().tolist():
        g1.add((index_to_node[src], predicted_link_predicate, index_to_node[dst]))

    logging.info(f"Saving enriched graph to {output_path}...")
    g1.serialize(output_path, format="turtle")

def enrich_graph(model, data, global_nodes, index_to_node, g1, threshold, output_path):
    logging.info("Predicting missing links...")
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        all_candidates = torch.tensor(list(combinations(range(len(global_nodes)), 2)), dtype=torch.long).t()
        predictions = model.decode(z, all_candidates).sigmoid()

    # Filter links with high probability
    high_prob_links = all_candidates[:, predictions > threshold]
    high_prob_values = predictions[predictions > threshold]

    predicted_link_predicate = URIRef("http://example.org/predictedLink")

    for src, dst in high_prob_links.t().tolist():
        g1.add((index_to_node[src], predicted_link_predicate, index_to_node[dst]))

    logging.info(f"Saving enriched graph to {output_path}...")
    g1.serialize(output_path, format="turtle")

    # Collect high-probability links for output
    high_probability_links = []
    for i, (src, dst) in enumerate(high_prob_links.t().tolist()):
        src_name = index_to_node[src]
        dst_name = index_to_node[dst]
        probability = high_prob_values[i].item()
        high_probability_links.append((src_name, dst_name, probability))

    return high_probability_links

def main():
    g1, g2 = load_graphs(SCHEMA_PATH, PROPERTIES_PATH)
    edge_index_combined, global_nodes, index_to_node = prepare_graphs(g1, g2)
    combined_data = create_data_object(edge_index_combined, len(global_nodes))
    model = train_model(combined_data, NUM_EPOCHS, index_to_node)
    high_probability_links = enrich_graph(
        model, combined_data, global_nodes, index_to_node, g1, THRESHOLD, ENRICHED_GRAPH_PATH
    )
    logging.info("Workflow completed successfully!")

    # Display high-probability links
    print("\nHigh-Probability Links (Probability > 95%):")
    for src, dst, prob in high_probability_links:
        print(f"Predicted Link: {src} -> {dst}, Probability: {prob:.4f}")

if __name__ == "__main__":
    main()
