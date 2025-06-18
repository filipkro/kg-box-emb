# %%
import rdflib
import os
from rdflib.namespace import RDF, RDFS, OWL
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE)

KG = rdflib.Namespace('http://sgd-kg.project-genesis.io#')
OBOOWL = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')
SGD = rdflib.Namespace('http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#')
BC = rdflib.Namespace('http://biocyc.project-genesis.io#')
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')
# %%
kg = rdflib.Graph()
kg.bind('sgd', SGD)
kg.bind('kg', KG)
# %%
knowledge = rdflib.Graph()
knowledge.parse(os.path.join(BASE, 'graphs/proteins.ttl'))
knowledge.parse(os.path.join(BASE, 'graphs/catalyzedBy.ttl'))

# %%
def role_between_classes(a, b, r):
    bn = rdflib.BNode()
    trips = []
    trips.append((bn, RDF.type, OWL.Restriction))
    trips.append((bn, OWL.onProperty, r))
    trips.append((bn, OWL.someValuesFrom, b))
    trips.append((a, RDFS.subClassOf, bn))
    return trips
# %%
def find_path(r1, r2):
    q = f"""
PREFIX owl: <{str(OWL)}>
PREFIX rdfs: <{str(RDFS)}>
PREFIX sgd: <{str(SGD)}>
SELECT ?a ?b
WHERE
{{
    ?z owl:onProperty <{r1}> .
    ?a rdfs:subClassOf ?z .
    ?z owl:someValuesFrom ?c .

    ?x owl:onProperty <{r2}> .
    ?c rdfs:subClassOf ?x .
    ?x owl:someValuesFrom ?b .
}}"""
    
    return list(knowledge.query(q))

# %%
r1 = SGD.catalyzedBy
r2 = SGD.encodedBy
# %%
pairs = find_path(str(r1), str(r2))
print(pairs)
# %%
trips = []
for p in pairs:
    trips.extend(role_between_classes(p[0], p[1], SGD.catalyzedByGene))

for trip in trips:
    kg.add(trip)

kg.add((SGD.catalyzedByGene, RDF.type, OWL.ObjectProperty))
# %%
kg.serialize(os.path.join(BASE, 'graphs/catalyzedByGene.ttl'))
# %%
