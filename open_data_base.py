from SPARQLWrapper import SPARQLWrapper, JSON, XML, RDF
import pandas as pd

sparql = SPARQLWrapper("http://dbpedia.org/sparql") #determine SPARQL endpoint
sparql.setReturnFormat(JSON) #determine the output format

from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS
from rdflib.namespace import XSD

def query_car_info():
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    # Modified query to get one car per manufacturer using GROUP BY and SAMPLE
    query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT DISTINCT ?manufacturer (SAMPLE(?car) as ?car)
           (SAMPLE(?name) as ?name)
           (SAMPLE(?manufacturerName) as ?manufacturerName)
           (SAMPLE(?year) as ?year)
           (SAMPLE(?engineType) as ?engineType)
           (SAMPLE(?horsepower) as ?horsepower)
           (SAMPLE(?transmission) as ?transmission)
           (SAMPLE(?bodyStyle) as ?bodyStyle)
           (SAMPLE(?assembly) as ?assembly)
           (SAMPLE(?weight) as ?weight)
           (SAMPLE(?length) as ?length)
           (SAMPLE(?width) as ?width)
           (SAMPLE(?height) as ?height)
           (SAMPLE(?fuelType) as ?fuelType)
    WHERE {
        ?car a dbo:Automobile ;
             rdfs:label ?name ;
             dbo:manufacturer ?manufacturer .
        ?manufacturer rdfs:label ?manufacturerName .

        OPTIONAL { ?car dbp:productionStartYear ?year }
        OPTIONAL { ?car dbo:engineType ?engineType }
        OPTIONAL { ?car dbp:horsepower ?horsepower }
        OPTIONAL { ?car dbo:transmission ?transmission }
        OPTIONAL { ?car dbo:bodyStyle ?bodyStyle }
        OPTIONAL { ?car dbp:assembly ?assembly }
        OPTIONAL { ?car dbo:weight ?weight }
        OPTIONAL { ?car dbo:length ?length }
        OPTIONAL { ?car dbo:width ?width }
        OPTIONAL { ?car dbo:height ?height }
        OPTIONAL { ?car dbp:fuelType ?fuelType }

        FILTER (LANG(?name) = 'en')
        FILTER (LANG(?manufacturerName) = 'en')
    }
    GROUP BY ?manufacturer
    LIMIT 20
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()

        # Create RDF graph for Turtle export
        g = Graph()

        # Define namespaces
        auto = Namespace("http://example.org/auto/")
        g.bind("auto", auto)
        g.bind("dbo", "http://dbpedia.org/ontology/")
        g.bind("dbp", "http://dbpedia.org/property/")

        for result in results["results"]["bindings"]:
            print("\nCar Details:")
            print(f"Name: {result['name']['value']}")
            print(f"Manufacturer: {result['manufacturerName']['value']}")

            # Create URI for the car
            car_uri = URIRef(result['car']['value'])

            # Add triples to the graph
            g.add((car_uri, RDF.type, URIRef("http://dbpedia.org/ontology/Automobile")))
            g.add((car_uri, RDFS.label, Literal(result['name']['value'])))
            g.add((car_uri, URIRef("http://dbpedia.org/ontology/manufacturer"),
                  URIRef(result['manufacturer']['value'])))

            # Dictionary of optional fields with their properties
            optional_fields = {
                'year': ('productionStartYear', XSD.integer),
                'engineType': ('engineType', None),
                'horsepower': ('horsepower', XSD.integer),
                'transmission': ('transmission', None),
                'bodyStyle': ('bodyStyle', None),
                'assembly': ('assembly', None),
                'weight': ('weight', XSD.decimal),
                'length': ('length', XSD.decimal),
                'width': ('width', XSD.decimal),
                'height': ('height', XSD.decimal),
                'fuelType': ('fuelType', None)
            }

            # Print and add to graph all available optional fields
            for field, (prop, datatype) in optional_fields.items():
                if field in result:
                    value = result[field]['value']
                    # Clean up URIs if present
                    if value.startswith('http'):
                        clean_value = value.split('/')[-1].replace('_', ' ')
                        print(f"{field.title()}: {clean_value}")
                        g.add((car_uri, URIRef(f"http://dbpedia.org/property/{prop}"),
                              URIRef(value)))
                    else:
                        print(f"{field.title()}: {value}")
                        if datatype:
                            try:
                                typed_value = Literal(value, datatype=datatype)
                                g.add((car_uri, URIRef(f"http://dbpedia.org/property/{prop}"),
                                      typed_value))
                            except ValueError:
                                # If conversion fails, add as string
                                g.add((car_uri, URIRef(f"http://dbpedia.org/property/{prop}"),
                                      Literal(value)))
                        else:
                            g.add((car_uri, URIRef(f"http://dbpedia.org/property/{prop}"),
                                  Literal(value)))

            print("-" * 50)

        # Save the graph in Turtle format
        g.serialize(destination="car_data.ttl", format="turtle")
        print("\nData has been exported to 'car_data.ttl' in Turtle format")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    query_car_info()