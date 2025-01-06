# app/routes/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from rdflib import Graph
from itertools import combinations
from app.models.gnn import SimpleGNN  # Assuming you'll move your GNN model to a models directory
from app.models.gnn import rdf_to_graph

router = APIRouter(
    prefix="/predict",
    tags=["prediction"]
)

class GraphData(BaseModel):
    rdf_data: str
    format: str = "turtle"

class LinkPrediction(BaseModel):
    source_node: str
    target_node: str
    probability: float

class PredictionResponse(BaseModel):
    predictions: List[LinkPrediction]
    model_version: str = "0.1.0"

# Global variables for model and mappings
global_model = None
global_nodes = {}
index_to_node = {}

def initialize_model():
    """Initialize the GNN model with saved weights"""
    global global_model
    num_nodes = len(global_nodes)
    global_model = SimpleGNN(in_channels=num_nodes, hidden_channels=16, out_channels=16)
    # Load model weights if available
    # global_model.load_state_dict(torch.load("path_to_weights"))
    global_model.eval()

@router.post("/links", response_model=PredictionResponse)
async def predict_links(graph_data: GraphData):
    """Predict potential links in the input RDF graph"""
    try:
        if global_model is None:
            initialize_model()
        
        # Parse input RDF
        g = Graph()
        g.parse(data=graph_data.rdf_data, format=graph_data.format)
        
        # Convert to PyTorch Geometric format
        edge_index, nodes_map, idx_to_node = rdf_to_graph(g, global_nodes.copy())
        num_nodes = len(nodes_map)
        x = torch.eye(num_nodes, dtype=torch.float)
        
        # Generate candidate links for carshare nodes
        nodes_in_graph = [node for node in nodes_map.keys() if "carshare" in str(node)]
        graph_indices = [nodes_map[node] for node in nodes_in_graph]
        candidate_links = list(combinations(graph_indices, 2))
        candidate_edge_index = torch.tensor(candidate_links, dtype=torch.long).t().contiguous()
        
        # Make predictions
        with torch.no_grad():
            z = global_model.encode(x, edge_index)
            preds = global_model.decode(z, candidate_edge_index).sigmoid()
        
        # Filter high probability predictions (>90%)
        threshold = 0.9
        high_prob_mask = preds > threshold
        high_prob_links = candidate_edge_index[:, high_prob_mask]
        high_prob_preds = preds[high_prob_mask]
        
        # Format predictions
        predictions = [
            LinkPrediction(
                source_node=str(idx_to_node[src]),
                target_node=str(idx_to_node[dst]),
                probability=float(prob)
            )
            for (src, dst), prob in zip(high_prob_links.t().tolist(), high_prob_preds.tolist())
        ]
        
        return PredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Get information about the current model"""
    return {
        "model_type": "SimpleGNN",
        "hidden_channels": 16,
        "output_channels": 16,
        "version": "0.1.0"
    }