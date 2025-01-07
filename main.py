from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from app.models.gnn_refactored import load_graphs, prepare_graphs, create_data_object, train_model, enrich_graph
import subprocess

app = FastAPI()

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