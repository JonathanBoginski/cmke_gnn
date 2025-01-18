from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from app.models.gnn_refactored import load_graphs, prepare_graphs, create_data_object, train_model, enrich_graph
from spotlight import annotate_text_with_spotlight, fetch_data_from_sparql, save_to_ttl
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List
import spacy



import subprocess

app = FastAPI()

# Function to fetch the abstract from DBpedia
def fetch_abstract(uri: str):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?abstract
    WHERE {{
        <{uri}> dbo:abstract ?abstract .
        FILTER (LANG(?abstract) = 'en')
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        if bindings:
            return bindings[0]["abstract"]["value"]
        else:
            raise HTTPException(status_code=404, detail="Abstract not found for the given URI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching abstract: {str(e)}")

# Function to generate keywords using LLM
def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from the given text using SpaCy.
    """
    doc = nlp(text)
    # Extract noun chunks and proper nouns as keywords
    keywords = [chunk.text for chunk in doc.noun_chunks]
    # Deduplicate and limit the number of keywords
    unique_keywords = list(dict.fromkeys(keywords))[:max_keywords]
    return unique_keywords

# Load the SpaCy model globally
nlp = spacy.load("en_core_web_sm")

# FastAPI Endpoint
@app.post("/generate-keywords/")
async def generate_keywords(
    text: str = Query(..., description="Input text for keyword extraction"),
    max_keywords: int = Query(10, ge=1, le=100, description="Maximum number of keywords to return")
):
    """
    Generate keywords from input text using SpaCy.
    """
    try:
        # Extract keywords from the input text
        keywords = extract_keywords(text, max_keywords=max_keywords)
        return {"text": text, "keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating keywords: {str(e)}")

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

@app.post("/spotlight/")
async def annotate_and_fetch(
    text: str,
    confidence: float = Query(0.5, ge=0.0, le=1.0),
    min_similarity_score: float = Query(0.8, ge=0.0, le=1.0)
):
    """
    Annotate text using DBpedia Spotlight and fetch relevant data.
    """
    try:
        # Step 1: Call the Spotlight API with specified confidence
        annotations = annotate_text_with_spotlight(text, confidence=confidence)
        entities = [
            {"uri": resource["@URI"], "surface_form": resource["@surfaceForm"], "score": float(resource["@similarityScore"])}
            for resource in annotations.get("Resources", [])
            if float(resource["@similarityScore"]) >= min_similarity_score
        ]
        if not entities:
            return {"message": "No entities with sufficient confidence were found."}

        # Step 2: Fetch data from SPARQL
        #sparql_results = fetch_data_from_sparql(entities)

        # Step 3: Save results to Turtle format
        #save_to_ttl(sparql_results, entities)
        
        return {"message": "Data saved successfully to 'dynamic_data.ttl'.", "entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))