from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS

def query_and_save_to_ttl():
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    # Query to fetch filtered property types for dbo:Automobile
    query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?property ?label
    WHERE {
        ?car a dbo:Automobile .
        ?car ?property ?value .
        ?property rdfs:label ?label .
        FILTER (LANG(?label) = 'en')
        FILTER NOT EXISTS {
            VALUES ?property {
                dbo:wikiPageID
                dbo:wikiPageRevisionID
                dbo:wikiPageWikiLink
                dbo:wikiPageExternalLink
            }
        }
    }
    LIMIT 50
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()

        # Initialize RDF Graph
        g = Graph()

        # Define namespace for automobile properties
        auto_ns = Namespace("http://example.org/auto/")
        g.bind("auto", auto_ns)

        print("Filtered Properties for Automobiles:\n")
        for result in results["results"]["bindings"]:
            property_uri = result["property"]["value"]
            property_label = result["label"]["value"]
            
            print(f"Property: {property_label}")

            # Add triples to the graph
            property_node = URIRef(property_uri)
            g.add((property_node, RDFS.label, Literal(property_label, lang="en")))

        # Save the graph in Turtle format
        output_file = "open_data_car_properties.ttl"
        g.serialize(destination=output_file, format="turtle")
        print(f"\nData has been saved to '{output_file}' in Turtle format.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    query_and_save_to_ttl()
