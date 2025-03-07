# %%
import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import os

KG = rdflib.Namespace('http://sgd-kg.project-genesis.io#')
BC = rdflib.Namespace('http://biocyc.project-genesis.io#')
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')
SGD = rdflib.Namespace('http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#')
OBOOWL = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')
CHEBI = rdflib.Namespace('http://purl.obolibrary.org/obo/CHEBI_')
GO = rdflib.Namespace('http://purl.obolibrary.org/obo/GO_')


def generate_merged(BASE, bc, cco, chebi, go_ext, kg_nf, pathways,
                    proteins, reactions, sgd_kg_fix, go_kg=None, cat_kg=None):
    GRAPHS = os.path.join(BASE, 'graphs')
# %%
    g = rdflib.Graph()

    # g.bind('chebi', CHEBI)
    g.bind('obo', OBO)
    g.bind('bc', BC)
    g.bind('oboInOwl', OBOOWL)
    g.bind('rdfs', RDFS)
    g.bind('rdf', RDF)
    g.bind('owl', OWL)
    # g.bind('go', GO)
    g.bind('sgd', SGD)
    g.bind('kg', KG)

    g += bc
    del bc
    print(len(g))
    g += cco
    del cco
    print(len(g))
    g += chebi
    del chebi
    print(len(g))
    g += kg_nf
    del kg_nf
    print(len(g))
    g += pathways
    del pathways
    print(len(g))
    g += proteins
    del proteins
    print(len(g))
    g += reactions
    del reactions
    print(len(g))
    g += sgd_kg_fix
    del sgd_kg_fix
    print(len(g))
    g += go_ext
    del go_ext
    print(len(g))

    if go_kg != None:
        g += go_kg
        del go_kg
        print(len(g))

    if cat_kg != None:
        g += cat_kg
        del cat_kg
        print(len(g))


    g.serialize(os.path.join(GRAPHS, 'full-kg-no-int.ttl'))
    print('full-kg.ttl saved')

    pg = rdflib.Graph()
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        for p, o in g.predicate_objects(prop):
            pg.add((prop, p, o))

    pg.serialize(os.path.join(GRAPHS, 'role-graph-no-int.ttl'))
    print('role-graph-no-int.ttl saved')
# %%


if __name__ == '__main__':

    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    GRAPHS = os.path.join(BASE, 'graphs')
    bc = rdflib.Graph()
    bc.parse(os.path.join(GRAPHS, 'bc.ttl'))
    cco = rdflib.Graph()
    cco.parse(os.path.join(GRAPHS, 'cco.ttl'))
    chebi = rdflib.Graph()
    chebi.parse(os.path.join(GRAPHS, 'chebi.ttl'))
    go_ext = rdflib.Graph()
    go_ext.parse(os.path.join(GRAPHS, 'go-ext.ttl'))
    kg_nf = rdflib.Graph()
    kg_nf.parse(os.path.join(GRAPHS, 'kg-nf-no-int.ttl'))
    pathways = rdflib.Graph()
    pathways.parse(os.path.join(GRAPHS, 'pathways.ttl'))
    proteins = rdflib.Graph()
    proteins.parse(os.path.join(GRAPHS, 'proteins.ttl'))
    reactions = rdflib.Graph()
    reactions.parse(os.path.join(GRAPHS, 'reactions.ttl'))
    sgd_kg_fix = rdflib.Graph()
    sgd_kg_fix.parse(os.path.join(GRAPHS, 'sgd_kg_fix.ttl'))

    generate_merged(BASE, bc, cco, chebi, go_ext, kg_nf, pathways,
                    proteins, reactions, sgd_kg_fix)