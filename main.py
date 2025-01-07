from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from app.models.gnn_refactored import load_graphs, prepare_graphs, create_data_object, train_model, enrich_graph
import subprocess

app = FastAPI()

# Constants (modify if necessary)
SCHEMA_PATH = "carshare-schema.ttl"
PROPERTIES_PATH = "open_data_car_properties.ttl"
NUM_EPOCHS = 100
THRESHOLD = 0.9

# Initialize GNN model
g1, g2 = load_graphs(SCHEMA_PATH, PROPERTIES_PATH)
edge_index_combined, global_nodes, index_to_node = prepare_graphs(g1, g2)
combined_data = create_data_object(edge_index_combined, len(global_nodes))
model = train_model(combined_data, NUM_EPOCHS, index_to_node)

# Define Input Schema
class PredictLinksInput(BaseModel):
    threshold: float = THRESHOLD  # Default threshold of 0.9


@app.post("/predict-links/", response_class=PlainTextResponse)
def predict_links(input_data: PredictLinksInput):
    """
    Predict links with probabilities above the specified threshold.
    Returns a plain-text formatted output for better readability.
    """
    high_probability_links = enrich_graph(
        model, combined_data, global_nodes, index_to_node, g1, input_data.threshold, None
    )

    # Generate a clean, console-style string
    output = "\nHigh-Probability Links (Probability > {:.2f}):\n".format(input_data.threshold)
    for src, dst, prob in high_probability_links:
        output += f"Predicted Link: {src} -> {dst}, Probability: {prob:.4f}\n"

    return output

@app.post("/predict", response_class=PlainTextResponse)
def predict():
    """
    Execute the gnn.py script and return its output as plain text.
    """
    try:
        # Run the `gnn.py` script
        result = subprocess.run(["python", "app/models/gnn.py"], capture_output=True, text=True)
        
        # Check if the script ran successfully
        if result.returncode == 0:
            return f"Script executed successfully.\n\nOutput:\n{result.stdout}"
        else:
            return f"Script execution failed.\n\nError:\n{result.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"
    
@app.post("/get_open_data", response_class=PlainTextResponse)
def get_open_data():
    """
    Execute the open_data.py script and return its output as plain text.
    """
    try:
        # Run the `open_data.py` script
        result = subprocess.run(["python", "open_data.py"], capture_output=True, text=True)
        
        # Check if the script ran successfully
        if result.returncode == 0:
            return f"Script executed successfully.\n\nOutput:\n{result.stdout}"
        else:
            return f"Script execution failed.\n\nError:\n{result.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"