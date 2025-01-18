# import requests
# from SPARQLWrapper import SPARQLWrapper, JSON
# from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS

# # 1. Funktion: DBpedia Spotlight API verwenden
# def annotate_text_with_spotlight(text):
#     url = "https://api.dbpedia-spotlight.org/en/annotate"
#     headers = {"Accept": "application/json"}
#     data = {"text": text, "confidence": 0.5}
    
#     response = requests.post(url, headers=headers, data=data)
    
#     if response.status_code == 200:
#         return response.json()
#     else:
#         raise Exception(f"Spotlight API failed with status {response.status_code}: {response.text}")

# # 2. Funktion: SPARQL-Query dynamisch erzeugen und Daten abfragen
# def fetch_data_from_sparql(entities):
#     sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    
#     # Query basierend auf den Entitäten erstellen
#     entity_uris = " ".join([f"<{entity['uri']}>" for entity in entities])
#     query = f"""
#     PREFIX dbo: <http://dbpedia.org/ontology/>
#     PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

#     SELECT DISTINCT ?property ?label
#     WHERE {{
#         VALUES ?entity {{ {entity_uris} }}
#         ?entity ?property ?value .
#         ?property rdfs:label ?label .
#         FILTER (LANG(?label) = 'en')
#         FILTER NOT EXISTS {{
#             VALUES ?property {{
#                 dbo:wikiPageID
#                 dbo:wikiPageRevisionID
#                 dbo:wikiPageWikiLink
#                 dbo:wikiPageExternalLink
#             }}
#         }}
#     }}
#     LIMIT 50
#     """
    
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
    
#     results = sparql.query().convert()
#     return results

# # 3. Funktion: Ergebnisse in Turtle speichern
# def save_to_ttl(results, output_file="dynamic_data.ttl"):
#     g = Graph()
    
#     # Namespace definieren
#     auto_ns = Namespace("http://example.org/auto/")
#     g.bind("auto", auto_ns)

#     for result in results["results"]["bindings"]:
#         property_uri = result["property"]["value"]
#         property_label = result["label"]["value"]

#         property_node = URIRef(property_uri)
#         g.add((property_node, RDFS.label, Literal(property_label, lang="en")))

#     g.serialize(destination=output_file, format="turtle")
#     print(f"\nData has been saved to '{output_file}' in Turtle format.")

# # 4. Hauptprozess: Dynamische Query & Speicherung
# if __name__ == "__main__":
#     text = input("Enter a text to annotate: ")
    
#     try:
#         # Schritt 1: Spotlight API aufrufen
#         annotations = annotate_text_with_spotlight(text)
#         print(f"\nAnnotations:\n{annotations}")

#         # Entitäten extrahieren
#         entities = []
#         for resource in annotations.get("Resources", []):
#             uri = resource["@URI"]
#             surface_form = resource["@surfaceForm"]
#             score = float(resource["@similarityScore"])
#             if score > 0.8:  # Filtere Entitäten mit geringem Score heraus
#                 entities.append({"uri": uri, "surface_form": surface_form, "score": score})
        
#         print(f"Extracted entities: {entities}")

#         if entities:
#             # Schritt 2: SPARQL-Query erstellen und ausführen
#             sparql_results = fetch_data_from_sparql(entities)
#             print(f"\nSPARQL Results:\n{sparql_results}")

#             # Schritt 3: Ergebnisse speichern
#             save_to_ttl(sparql_results)
#         else:
#             print("No entities with sufficient confidence were found.")

#     except Exception as e:
#         print(f"An error occurred: {e}")

import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS

# 1. Function: Use DBpedia Spotlight API
def annotate_text_with_spotlight(text, confidence=0.5):
    url = "https://api.dbpedia-spotlight.org/en/annotate"
    headers = {"Accept": "application/json"}
    data = {"text": text, "confidence": confidence}

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Spotlight API failed with status {response.status_code}: {response.text}")



# 2. Function: Create SPARQL query dynamically and fetch data
def fetch_data_from_sparql(entities):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    
    # Construct the SPARQL query using the annotated entities
    entity_uris = " ".join([f"<{entity['uri']}>" for entity in entities])
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?property ?label
    WHERE {{
        VALUES ?entity {{ {entity_uris} }}
        ?entity ?property ?value .
        ?property rdfs:label ?label .
        FILTER (LANG(?label) = 'en')
        FILTER NOT EXISTS {{
            VALUES ?property {{
                dbo:wikiPageID
                dbo:wikiPageRevisionID
                dbo:wikiPageWikiLink
                dbo:wikiPageExternalLink
            }}
        }}
    }}
    LIMIT 50
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    results = sparql.query().convert()
    return results

# 3. Function: Save results to Turtle format
def save_to_ttl(results, entities, output_file="dynamic_data.ttl"):
    g = Graph()
    
    # Determine namespace based on the primary entity URI
    primary_entity_uri = entities[0]["uri"]
    entity_name = primary_entity_uri.split("/")[-1]  # Extract the entity name from the URI
    ns_base = f"http://example.org/{entity_name.lower()}/"
    ns = Namespace(ns_base)
    g.bind("context", ns)

    # Add properties and labels to the graph
    for result in results["results"]["bindings"]:
        property_uri = result["property"]["value"]
        property_label = result["label"]["value"]

        property_node = URIRef(property_uri)
        g.add((property_node, RDFS.label, Literal(property_label, lang="en")))

    # Serialize and save the graph in Turtle format
    g.serialize(destination=output_file, format="turtle")
    return(f"\nData has been saved to '{output_file}' in Turtle format.")

# 4. Main Process: Dynamically generate SPARQL query and save results
if __name__ == "__main__":
    text = input("Enter a text to annotate: ")
    
    try:
        # Step 1: Call the Spotlight API
        annotations = annotate_text_with_spotlight(text)
        print(f"\nAnnotations:\n{annotations}")

        # Extract entities from Spotlight API response
        entities = []
        for resource in annotations.get("Resources", []):
            uri = resource["@URI"]
            surface_form = resource["@surfaceForm"]
            score = float(resource["@similarityScore"])
            if score > 0.8:  # Filter entities with low confidence scores
                entities.append({"uri": uri, "surface_form": surface_form, "score": score})
        
        print(f"Extracted entities: {entities}")

        if entities:
            # Step 2: Generate and execute the SPARQL query
            sparql_results = fetch_data_from_sparql(entities)
            print(f"\nSPARQL Results:\n{sparql_results}")

            # Step 3: Save results to Turtle format
            save_to_ttl(sparql_results, entities)
        else:
            print("No entities with sufficient confidence were found.")

    except Exception as e:
        print(f"An error occurred: {e}")
