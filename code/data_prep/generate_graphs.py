import rdflib
import re, os

from bc import generate_bc
from cco import generate_cco
from go_kg import generate_go_kg
from gene_kg import generate_gene_kg
from pathways import generate_pathways
from proteins import generate_proteins
from reactions import generate_reactions
from merge_graphs import generate_merged

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGE_ALL = True
if not os.path.exists(os.path.join(BASE, 'data/sgd-data-ext.pkl')):
    print("Querying SGD for gene data...")
    import query_sgd
if not os.path.exists(os.path.join(BASE, 'data/sgd-data-slim.pkl')):
    print("Creating slim version of gene data...")
    import slim_data

print('generating pathways.ttl')
pathways = generate_pathways(BASE)


print('loading go...')
go = rdflib.Graph()
go.parse(os.path.join(BASE, 'graphs/go-ext.ttl'))

print('loading sgd ontology...')
sgd = rdflib.Graph()
sgd.parse(os.path.join(BASE, 'graphs/sgd_kg_fix.ttl'))


print('generating cco.ttl')
cco = generate_cco(BASE, go)
print('generating reactions.ttl')
reactions, cco, cat_kg = generate_reactions(BASE, cco)

print('generating kg-go.ttl')
go_kg = generate_go_kg(BASE, go, sgd)
print('generating kg-nf.ttl')
kg_nf = generate_gene_kg(BASE, go, sgd)
print('generating proteins.ttl')
proteins = generate_proteins(BASE, kg_nf)

print('loading chebi...')
chebi = rdflib.Graph()
chebi.parse(os.path.join(BASE, 'graphs/chebi.ttl'))
print('generating bc.ttl')
bc = generate_bc(BASE, chebi)

print('merging graphs...')
if MERGE_ALL:
    generate_merged(BASE, bc, cco, chebi, go, kg_nf, pathways, proteins,
                reactions, sgd, go_kg=go_kg, cat_kg=cat_kg)
else:    
    generate_merged(BASE, bc, cco, chebi, go, kg_nf, pathways, proteins,
                reactions, sgd)
print('all done, thank you for your patience')