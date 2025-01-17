import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef, Namespace, RDF, RDFS

def annotate_text_with_spotlight(text):
    url = "https://api.dbpedia-spotlight.org/en/annotate"
    headers = {"Accept": "application/json"}
    data = {"text": text, "confidence": 0.5}

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        try:
            return response.json()  # Properly parse the JSON response
        except ValueError as e:
            raise Exception(f"Failed to decode JSON response: {e}")
    else:
        raise Exception(f"Spotlight API failed with status {response.status_code}: {response.text}")

def fetch_data_from_sparql(entities):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    entity_uris = " ".join([f"<{entity['uri']}>" for entity in entities])
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>

    SELECT DISTINCT ?class ?property ?domain ?range
    WHERE {{
        VALUES ?entity {{ {entity_uris} }}
        ?entity a ?class .

        OPTIONAL {{
            ?property a rdf:Property ;
                      rdfs:domain ?domain ;
                      rdfs:range ?range .
        }}
    }}
    LIMIT 10
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        return results
    except Exception as e:
        raise Exception(f"Failed to execute SPARQL query: {e}")

def save_to_ttl(results, output_file="dynamic_data.ttl"):
    # Initialize RDF graph
    g = Graph()
    carshare_ns = Namespace("http://example.org/carshare#")
    xsd = Namespace("http://www.w3.org/2001/XMLSchema#")

    # Bind namespaces
    g.bind("carshare", carshare_ns)
    g.bind("xsd", xsd)

    for result in results.get("results", {}).get("bindings", []):
        # Extract data safely
        class_uri = URIRef(result["class"]["value"]) if "class" in result else None
        property_uri = URIRef(result["property"]["value"]) if "property" in result else None
        domain_uri = URIRef(result["domain"]["value"]) if "domain" in result else None
        range_uri = URIRef(result["range"]["value"]) if "range" in result else None

        # Add classes
        if class_uri:
            g.add((class_uri, RDF.type, RDFS.Class))

        # Add properties
        if property_uri:
            g.add((property_uri, RDF.type, RDF.Property))
            if domain_uri:
                g.add((property_uri, RDFS.domain, domain_uri))
            if range_uri:
                g.add((property_uri, RDFS.range, range_uri))

    # Serialize graph to Turtle format
    g.serialize(destination=output_file, format="turtle")
    print(f"Schema saved to {output_file}")

if __name__ == "__main__":
    text = input("Enter a text to annotate: ")

    try:
        annotations = annotate_text_with_spotlight(text)
        print(f"\nAnnotations:\n{annotations}")

        entities = []
        for resource in annotations.get("Resources", []):
            uri = resource["@URI"]
            surface_form = resource["@surfaceForm"]
            score = float(resource["@similarityScore"])
            if score > 0.8:
                entities.append({"uri": uri, "surface_form": surface_form, "score": score})

        print(f"Extracted entities: {entities}")

        if entities:
            sparql_results = fetch_data_from_sparql(entities)
            print(f"\nSPARQL Results:\n{sparql_results}")

            save_to_ttl(sparql_results)
        else:
            print("No entities with sufficient confidence were found.")

    except Exception as e:
        print(f"An error occurred: {e}")
