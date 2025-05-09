# %%
import pickle, os
import rdflib
from rdflib.namespace import RDFS, RDF, OWL

OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')
SGD = rdflib.Namespace('http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#')
KG = rdflib.Namespace('http://sgd-kg.project-genesis.io#')
OBOOWL = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')

def add_list(terms):
    trips = []
    for x in terms:
        bn = rdflib.BNode()
        if len(trips) > 0:
            prev = trips[-1][0]
            trips.append((prev, RDF.rest, bn))
        trips.append((bn, RDF.type, RDF.List))
        trips.append((bn, RDF.first, x))
    trips.append((bn, RDF.rest, RDF.nil))

    return trips

def role_between_classes(a, b, r):
    bn = rdflib.BNode()
    trips = []
    trips.append((bn, RDF.type, OWL.Restriction))
    trips.append((bn, OWL.onProperty, r))
    trips.append((bn, OWL.someValuesFrom, b))
    trips.append((a, RDFS.subClassOf, bn))
    return trips

def sub_of_intersect(sub, sup):
    trips = []
    for parent in sup:
        trips.append((sub, RDFS.subClassOf, parent))
    return trips


def generate_go_kg(BASE, go, sgd):

    data_dir = os.path.join(BASE, 'data/')
    graph_dir = os.path.join(BASE, 'graphs/')
    ont = rdflib.Graph()

    ont += go
    ont += sgd


    def term_from_label(label):
        term = term = ont.value(predicate=RDFS.label,
                                object=rdflib.Literal(label, datatype=rdflib.URIRef('http://www.w3.org/2001/XMLSchema#string')))
        if term == None:
            term = ont.value(predicate=RDFS.label,
                            object=rdflib.Literal(label, lang='en'))
        if term == None:
            term = ont.value(predicate=RDFS.label,
                            object=rdflib.Literal(label))
        return term

    with open(data_dir + 'sgd-data-slim.pkl', 'rb') as fi:
        sgd_data = pickle.load(fi)

    remap_labels = {'regulation of cell aging': 'regulation of cellular senescence', 'regulation of response to DNA damage stimulus': 'cellular response to DNA damage stimulus', 'DNA damage response': 'cellular response to DNA damage stimulus', 'acts upstream of negative effect': 'acts upstream of, negative effect', 'acts upstream of positive effect': 'acts upstream of, positive effect', 'acts upstream of or within positive effect': 'acts upstream of or within, positive effect', 'null': 'null mutant', 'amino acid catabolic process': 'cellular amino acid catabolic process'}

    kg = rdflib.Graph()
    kg.bind('sgd', SGD)
    kg.bind('obo', OBO)
    kg.bind('kg', KG)
    kg.bind('oboInOwl', OBOOWL)
    NOT_go = []
    orfs = list(sgd_data.keys())
    c = 0
    for orf in sgd_data:
        for go in sgd_data[orf]['go_details']:
            goterm = OBO[go[0][1].replace(':', '_')]
            if go[0][0] == 'NOT':
                NOT_go.append((KG[orf], 'NOT', go[1]))
            # elif go[1] == 
            elif (goterm, OWL.deprecated, rdflib.Literal(True)) in ont:
                continue
            elif go[1] == 'manually curated' or go[1] == 'high-throughput':
                rel = remap_labels[go[0][0]] if go[0][0] in remap_labels \
                    else go[0][0]
                trips = role_between_classes(KG[orf], goterm, term_from_label(rel))
                for t in trips:
                    kg.add(t)
            else:
                c += 1
                
    print(c)
    print(len(kg))
    print(len(NOT_go))
    # %%
    kg.serialize(destination=graph_dir + "kg-go.ttl")
    print('kg-go.ttl saved')
    return kg

# %%

if __name__ == '__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(BASE)
    go = rdflib.Graph()
    go.parse(os.path.join(BASE, 'graphs/go-ext.ttl'))
    sgd = rdflib.Graph()
    sgd.parse(os.path.join(BASE, 'graphs/sgd_kg_fix.ttl'))

    generate_go_kg(BASE, go, sgd)


# %%
# ont.parse(graph_dir + 'go-ext.ttl')
# ont.parse(graph_dir + 'sgd_kg_fix.ttl')