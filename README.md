# cmke_gnn

## Overview
This project exposes two endpoints via a FastAPI application, accessible through Swagger UI at:

```
http://localhost:8001/docs
```

The application is currently in its foundational stage and serves as a starting point for more advanced functionality to be added in the future.

---

## How to Run the Application

### 1. Build the Docker Image
```bash
docker build -t cmke_gnn .
```

### 2. Run the Docker Container
```bash
docker run -p 8001:8001 --name cmke_gnn cmke_gnn
```

### 3. Access Swagger UI
Open your browser and navigate to:
```
http://localhost:8001/docs
```

---

## Endpoints

### 1. `/predict`
- **Purpose:** Predicts which nodes from a secondary RDF graph (`open_data_car_properties.ttl`) can append to a base RDF graph (`carshare-schema.ttl`).
- **Description:**
  - Uses a Graph Neural Network (GNN) model to evaluate and suggest links between nodes in the secondary graph and the base graph.
  - Currently it has 100 epochs and will show the value of it and after that show all the predicted links above the threshhold of 95%.
  - Both graphs have an RDF schema and are currently hardcoded into the application.
  - The endpoint provides basic predictions of potential node integrations based on schema compatibility.
- **Future Enhancements:**
  - Dynamic input of RDF graphs.
  - Improved prediction algorithms with richer RDF data support.

### 2. `/get_open_data`
- **Purpose:** Queries properties from DBpedia related to the topic of vehicles.
- **Description:**
  - Retrieves data from DBpedia using SPARQL queries.
  - The data fetched by this endpoint is used by `/predict` to enrich the base graph with new nodes and properties.
- **Future Enhancements:**
  - Support for querying additional topics.
  - Enhanced SPARQL queries for broader data coverage.

---

We will expand and enrich the capabilities of this application the following days!
