# app/routes/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from rdflib import Graph, URIRef
from itertools import combinations
from app.models.gnn import (
    rdf_to_graph, 
    SimpleGNN,
    global_nodes,
    index_to_node
)

router = APIRouter(
    prefix="/predict",
    tags=["prediction"]
)

class LinkPrediction(BaseModel):
    source_node: str
    target_node: str
    probability: float

class PredictionResponse(BaseModel):
    predictions: List[LinkPrediction]
    model_version: str = "0.1.0"

# Global variable for model
global_model = None

def initialize_model():
    """Initialize the GNN model with the trained weights"""
    global global_model
    num_nodes = len(global_nodes)
    global_model = SimpleGNN(in_channels=num_nodes, hidden_channels=16, out_channels=16)
    # Here you would load your trained model weights
    # global_model.load_state_dict(torch.load("path_to_your_saved_weights"))
    global_model.eval()

@router.post("/predict-links", response_model=PredictionResponse)
async def predict_links():
    """Predict potential links using the pre-trained model on the fixed RDF files"""
    try:
        if global_model is None:
            initialize_model()
        
        # Load your fixed RDF files
        g1 = Graph()
        g2 = Graph()
        g1.parse("carshare-schema.ttl", format="turtle")
        g2.parse("open_data_car_properties.ttl", format="turtle")

        # Convert RDF graphs to PyTorch Geometric format
        edge_index1, _, _ = rdf_to_graph(g1, global_nodes.copy())
        edge_index2, _, _ = rdf_to_graph(g2, global_nodes.copy())
        
        # Combine the graphs
        edge_index_combined = torch.cat([edge_index1, edge_index2], dim=1)
        num_nodes = len(global_nodes)
        x_combined = torch.eye(num_nodes, dtype=torch.float)
        
        # Generate candidate links for carshare nodes
        nodes_in_carshare = [node for node in global_nodes.keys() if "carshare" in str(node)]
        carshare_indices = [global_nodes[node] for node in nodes_in_carshare]
        candidate_links = list(combinations(carshare_indices, 2))
        candidate_edge_index = torch.tensor(candidate_links, dtype=torch.long).t().contiguous()
        
        # Make predictions
        with torch.no_grad():
            z = global_model.encode(x_combined, edge_index_combined)
            preds = global_model.decode(z, candidate_edge_index).sigmoid()
        
        # Filter high probability predictions
        threshold = 0.9
        high_prob_mask = preds > threshold
        high_prob_links = candidate_edge_index[:, high_prob_mask]
        high_prob_preds = preds[high_prob_mask]
        
        # Format predictions
        predictions = [
            LinkPrediction(
                source_node=str(index_to_node[src]),
                target_node=str(index_to_node[dst]),
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
        "version": "0.1.0",
        "rdf_files": ["carshare-schema.ttl", "open_data_car_properties.ttl"]
    }