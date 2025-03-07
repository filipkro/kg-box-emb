# %%
import rdflib
import os
from rdflib.plugins.stores import sparqlstore
from rdflib.namespace import RDFS, OWL, RDF
# %%
KG = rdflib.Namespace('http://sgd-kg.project-genesis.io#')
BC = rdflib.Namespace('http://biocyc.project-genesis.io#')
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')
SGD = rdflib.Namespace('http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#')
OBOOWL = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')
CHEBI = rdflib.Namespace('http://purl.obolibrary.org/obo/CHEBI_')
GO = rdflib.Namespace('http://purl.obolibrary.org/obo/GO_')

kg_endpoint = 'http://localhost:3030/kg'
sp_store = sparqlstore.SPARQLStore(kg_endpoint)
kg = rdflib.Graph(store=sp_store)
# %%
# genes sub bc:All-Genes
q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bc: <http://biocyc.project-genesis.io#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
SELECT ?a ?b ?al ?bl WHERE {
    ?a rdfs:subClassOf* bc:All-Genes .
    ?b rdfs:subClassOf ?a .
    OPTIONAL {?a rdfs:label ?al}
    OPTIONAL {?b rdfs:label ?bl}
}
"""

genes = kg.query(q)
gene_graph = rdflib.Graph()
gene_set = set()
for p in genes:
    # if p[0] not in gene_prod_set and p[1] not in gene_prod_set:
    gene_set.add(p[0])
    gene_set.add(p[1])
    gene_graph.add((p[1], RDFS.subClassOf, p[0]))
    if p[2]:
        gene_graph.add((p[0], RDFS.label, p[2]))
    if p[3]:
        gene_graph.add((p[1], RDFS.label, p[3]))
print(len(gene_set))
print(len(gene_graph))

# %%
# cellular component
q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bc: <http://biocyc.project-genesis.io#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
SELECT ?a ?b ?al ?bl WHERE {
    ?a rdfs:subClassOf* obo:GO_0005575 .
    ?b rdfs:subClassOf ?a .
    OPTIONAL {?a rdfs:label ?al}
    OPTIONAL {?b rdfs:label ?bl}
}
"""

cell_comp = kg.query(q)
cell_comp_graph = rdflib.Graph()
cell_comp_set = set()
for p in cell_comp:
    # if p[0] not in gene_prod_set and p[1] not in gene_prod_set:
    cell_comp_set.add(p[0])
    cell_comp_set.add(p[1])
    cell_comp_graph.add((p[1], RDFS.subClassOf, p[0]))
    if p[2]:
        cell_comp_set.add((p[0], RDFS.label, p[2]))
    if p[3]:
        cell_comp_set.add((p[1], RDFS.label, p[3]))
print(len(cell_comp_set))
print(len(cell_comp_graph))

# %%
q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bc: <http://biocyc.project-genesis.io#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
SELECT ?a ?b ?al ?bl WHERE {
    ?a rdfs:subClassOf* obo:BFO_0000040 .
    ?b rdfs:subClassOf ?a .
    OPTIONAL {?a rdfs:label ?al}
    OPTIONAL {?b rdfs:label ?bl}
}
"""

mat_ent = kg.query(q)
mat_graph = rdflib.Graph()
mat_set = set()
for p in mat_ent:
    if p[0] not in gene_set and p[1] not in gene_set and \
            p[0] not in cell_comp_set and p[1] not in cell_comp_set:
        mat_set.add(p[0])
        mat_set.add(p[1])
        mat_graph.add((p[1], RDFS.subClassOf, p[0]))
        if p[2]:
            mat_graph.add((p[0], RDFS.label, p[2]))
        if p[3]:
            mat_graph.add((p[1], RDFS.label, p[3]))
print(len(mat_ent))
print(len(mat_graph))
print(len(mat_set))

# %%
# all pathways and reactions
q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bc: <http://biocyc.project-genesis.io#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX sgd: <http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#>
SELECT ?a ?b ?al ?bl WHERE {
    ?a rdfs:subClassOf* bc:Generalized-Reactions .
    ?b rdfs:subClassOf ?a .
    OPTIONAL {?a rdfs:label ?al}
    OPTIONAL {?b rdfs:label ?bl}
}
"""

react = kg.query(q)
react_graph = rdflib.Graph()
react_set = set()
for p in react:
    react_set.add(p[0])
    react_set.add(p[1])
    react_graph.add((p[1], RDFS.subClassOf, p[0]))
    if p[2]:
        react_graph.add((p[0], RDFS.label, p[2]))
    if p[3]:
        react_graph.add((p[1], RDFS.label, p[3]))
print(len(react))
print(len(react_graph))
print(len(react_set))

# %%
# biological process
q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bc: <http://biocyc.project-genesis.io#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX sgd: <http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#>
SELECT ?a ?b ?al ?bl WHERE {
    ?a rdfs:subClassOf* obo:GO_0008150.
    ?b rdfs:subClassOf ?a .
    OPTIONAL {?a rdfs:label ?al}
    OPTIONAL {?b rdfs:label ?bl}
}
"""

bio_proc = kg.query(q)
bio_proc_graph = rdflib.Graph()
bio_proc_set = set()
for p in bio_proc:
    bio_proc_set.add(p[0])
    bio_proc_set.add(p[1])
    bio_proc_graph.add((p[1], RDFS.subClassOf, p[0]))
    if p[2]:
        bio_proc_graph.add((p[0], RDFS.label, p[2]))
    if p[3]:
        bio_proc_graph.add((p[1], RDFS.label, p[3]))
print(len(bio_proc))
print(len(bio_proc_graph))
print(len(bio_proc_set))


# %%
# interaction
q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bc: <http://biocyc.project-genesis.io#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX sgd: <http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#>
SELECT ?a ?b ?al ?bl WHERE {
    ?a rdfs:subClassOf* obo:INO_0000002.
    ?b rdfs:subClassOf ?a .
    OPTIONAL {?a rdfs:label ?al}
    OPTIONAL {?b rdfs:label ?bl}
}
"""

inter = kg.query(q)
inter_graph = rdflib.Graph()
inter_set = set()
for p in inter:
    inter_set.add(p[0])
    inter_set.add(p[1])
    inter_graph.add((p[1], RDFS.subClassOf, p[0]))
    if p[2]:
        inter_graph.add((p[0], RDFS.label, p[2]))
    if p[3]:
        inter_graph.add((p[1], RDFS.label, p[3]))
print(len(inter))
print(len(inter_graph))
print(len(inter_set))


# %%
# mol function
q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bc: <http://biocyc.project-genesis.io#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX sgd: <http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#>
SELECT ?a ?b ?al ?bl WHERE {
    ?a rdfs:subClassOf* obo:GO_0003674.
    ?b rdfs:subClassOf ?a .
    OPTIONAL {?a rdfs:label ?al}
    OPTIONAL {?b rdfs:label ?bl}
}
"""

mol = kg.query(q)
mol_graph = rdflib.Graph()
mol_set = set()
for p in mol:
    mol_set.add(p[0])
    mol_set.add(p[1])
    mol_graph.add((p[1], RDFS.subClassOf, p[0]))
    if p[2]:
        mol_graph.add((p[0], RDFS.label, p[2]))
    if p[3]:
        mol_graph.add((p[1], RDFS.label, p[3]))
print(len(mol))
print(len(mol_graph))
print(len(mol_set))
# %%
# process
q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bc: <http://biocyc.project-genesis.io#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX sgd: <http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#>
SELECT ?a ?b ?al ?bl WHERE {
    ?a rdfs:subClassOf* obo:BFO_0000015 .
    ?b rdfs:subClassOf ?a .
    OPTIONAL {?a rdfs:label ?al}
    OPTIONAL {?b rdfs:label ?bl}
}
"""

process = kg.query(q)
process_graph = rdflib.Graph()
process_set = set()
for p in process:
    if p[0] not in react_set and p[1] not in react_set and \
            p[0] not in bio_proc_set and p[1] not in bio_proc_set and \
                p[0] not in inter_set and p[1] not in inter_set and \
                    p[0] not in mol_set and p[1] not in mol_set:
        process_set.add(p[0])
        process_set.add(p[1])
        process_graph.add((p[1], RDFS.subClassOf, p[0]))
        if p[2]:
            process_graph.add((p[0], RDFS.label, p[2]))
        if p[3]:
            process_graph.add((p[1], RDFS.label, p[3]))
print(len(process))
print(len(process_graph))
print(len(process_set))

# %%
# quality BFO_0000019
q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bc: <http://biocyc.project-genesis.io#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX sgd: <http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#>
SELECT ?a ?b ?al ?bl WHERE {
    ?a rdfs:subClassOf* obo:BFO_0000019 .
    ?b rdfs:subClassOf ?a .
    OPTIONAL {?a rdfs:label ?al}
    OPTIONAL {?b rdfs:label ?bl}
}
"""

quality = kg.query(q)
quality_graph = rdflib.Graph()
quality_set = set()
for p in quality:
    quality_set.add(p[0])
    quality_set.add(p[1])
    quality_graph.add((p[1], RDFS.subClassOf, p[0]))
    if p[2]:
        quality_graph.add((p[0], RDFS.label, p[2]))
    if p[3]:
        quality_graph.add((p[1], RDFS.label, p[3]))
print(len(quality))
print(len(quality_graph))
print(len(quality_set))
# %%
# root NCBITaxon_1 , skip this?? just one-hot encode or something?? just some different strains
q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bc: <http://biocyc.project-genesis.io#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX sgd: <http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#>
SELECT ?a ?b ?al ?bl WHERE {
    ?a rdfs:subClassOf* obo:NCBITaxon_1 .
    ?b rdfs:subClassOf ?a .
    OPTIONAL {?a rdfs:label ?al}
    OPTIONAL {?b rdfs:label ?bl}
}
"""

root = kg.query(q)
root_graph = rdflib.Graph()
root_set = set()
for p in root:
    root_set.add(p[0])
    root_set.add(p[1])
    root_graph.add((p[1], RDFS.subClassOf, p[0]))
    if p[2]:
        root_graph.add((p[0], RDFS.label, p[2]))
    if p[3]:
        root_graph.add((p[1], RDFS.label, p[3]))
print(len(root))
print(len(root_graph))
print(len(root_set))
# %%
print(len(mat_graph))
print(len(cell_comp_graph))
print(len(quality_graph))
print(len(root_graph))
print(len(gene_set))

print(len(react_graph))
print(len(inter_graph))
print(len(bio_proc_graph))
print(len(mol_graph))
for s,o in kg.subject_objects(OWL.disjointWith):
    if s in mat_set and o in mat_set:
        mat_graph.add((s, OWL.disjointWith, o))
    elif s in cell_comp_set and o in cell_comp_set:
        cell_comp_graph.add((s, OWL.disjointWith, o))
    elif s in quality_set and o in quality_set:
        quality_graph.add((s, OWL.disjointWith, o))
    elif s in root_set and o in root_set:
        root_graph.add((s, OWL.disjointWith, o))
    elif s in gene_set and o in gene_set:
        gene_graph.add((s, OWL.disjointWith, o))
    elif s in react_set and o in react_set:
        react_graph.add((s, OWL.disjointWith, o))
    elif s in bio_proc_set and o in bio_proc_set:
        bio_proc_graph.add((s, OWL.disjointWith, o))
    elif s in inter_set and o in inter_set:
        inter_graph.add((s, OWL.disjointWith, o))
    elif s in mol_set and o in mol_set:
        mol_graph.add((s, OWL.disjointWith, o))
print()
print(len(mat_graph))
print(len(cell_comp_graph))
print(len(quality_graph))
print(len(root_graph))
print(len(gene_set))
print(len(react_graph))
print(len(inter_graph))
print(len(bio_proc_graph))
print(len(mol_graph))
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE)
# %%

graphs = [gene_graph, mat_graph, cell_comp_graph, quality_graph, root_graph,
          react_graph, inter_graph, mol_graph, bio_proc_graph]

for g in graphs:
    g.bind('obo', OBO)
    g.bind('bc', BC)
    g.bind('oboInOwl', OBOOWL)
    g.bind('rdfs', RDFS)
    g.bind('rdf', RDF)
    g.bind('owl', OWL)
    g.bind('sgd', SGD)
    g.bind('kg', KG)

gene_graph.serialize(destination=os.path.join(BASE, 'graphs/split_graphs/genes.ttl'))
mat_graph.serialize(destination=os.path.join(BASE, 'graphs/split_graphs/mat_ent.ttl'))
cell_comp_graph.serialize(destination=os.path.join(BASE, 'graphs/split_graphs/cell_comp.ttl'))
quality_graph.serialize(destination=os.path.join(BASE, 'graphs/split_graphs/quality.ttl'))
root_graph.serialize(destination=os.path.join(BASE, 'graphs/split_graphs/root.ttl'))

react_graph.serialize(destination=os.path.join(BASE, 'graphs/split_graphs/reactions.ttl'))
inter_graph.serialize(destination=os.path.join(BASE, 'graphs/split_graphs/reguls.ttl'))
bio_proc_graph.serialize(destination=os.path.join(BASE, 'graphs/split_graphs/bio_proc.ttl'))
mol_graph.serialize(destination=os.path.join(BASE, 'graphs/split_graphs/mol_func.ttl'))
# %%
