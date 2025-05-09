# %%
import rdflib
from rdflib.namespace import RDF, RDFS, OWL
from SPARQLWrapper import SPARQLWrapper, JSON
import pickle, os
import requests

KG = rdflib.Namespace('http://sgd-kg.project-genesis.io#')
OBOOWL = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')
SGD = rdflib.Namespace('http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#')
BC = rdflib.Namespace('http://biocyc.project-genesis.io#')
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')

def role2class(a, b, r):
    bn = rdflib.BNode()
    trips = []
    trips.append((bn, RDF.type, OWL.Restriction))
    trips.append((bn, OWL.onProperty, r))
    trips.append((bn, OWL.someValuesFrom, b))
    trips.append((a, RDFS.subClassOf, bn))
    return trips

def generate_proteins(BASE, kg):

    sparql = SPARQLWrapper("https://sparql.uniprot.org/sparql")
    lines = open(os.path.join(BASE, 'data/proteins.dat'), 'r').read().splitlines()
    prefixes = """
    PREFIX up: <http://purl.uniprot.org/core/>
    PREFIX uniprotkb: <http://purl.uniprot.org/uniprot/>
    PREFIX udb: <http://purl.uniprot.org/database/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX emblcds: <http://purl.uniprot.org/embl-cds/>
    PREFIX sgd: <http://purl.uniprot.org/sgd/>
    SELECT DISTINCT ?link
    WHERE
    {
        """

    print('load sgd_dict')
    with open(os.path.join(BASE, 'data/sgd_id_dict.pkl'), 'rb') as fi:
        id_dict = pickle.load(fi)
    curr = None

    print('link db')
    for l in lines:
        if l[0] == '#':
            continue
        l = l.split(' - ')
        match l[0]:
            case 'UNIQUE-ID':
                curr = l[1]
            case 'GENE':
                continue
            case 'DBLINKS':
                if not 'UNIPROT' in l[1]:
                    continue
                if curr[:2] == 'CP':
                    continue
                if curr in id_dict and len(id_dict[curr]) > 0:
                    continue
                pid = l[1].split('"')[1]
                q = f"""uniprotkb:{pid} rdfs:seeAlso ?link .
                        ?link up:database udb:SGD .
                    }}"""
                query = prefixes + q
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)
                results = sparql.query().convert()
                if len(results['results']['bindings']) == 0:
                    q = f"""?pid rdfs:seeAlso emblcds:{pid} .
                        ?pid rdfs:seeAlso ?link .
                        ?link up:database udb:SGD .
                    }}"""
                    query = prefixes + q
                    sparql.setQuery(query)
                    sparql.setReturnFormat(JSON)
                    results = sparql.query().convert()
                id_dict[curr] = [s['link']['value'].split('/')[-1]
                                for s in results['results']['bindings']]

    # %%
    with open(os.path.join(BASE, 'data/sgd2sys_dict.pkl'), 'rb') as fi:
        sgd_id_dict = pickle.load(fi)
    gene_lines = open(os.path.join(BASE, 'data/genes.dat'), 'r').read().splitlines()
    print('find gene names')
    gene_prod_dict = {}
    gene_types = {}

    prot_kg = rdflib.Graph()
    # prot_subclass = rdflib.Graph()

    prot_kg.bind('kg', KG)
    prot_kg.bind('obo', OBO)
    prot_kg.bind('bc', BC)
    prot_kg.bind('oboInOwl', OBOOWL)
    prot_kg.bind('rdfs', RDFS)
    prot_kg.bind('owl', OWL)
    prot_kg.bind('rdf', RDF)
    prot_kg.bind('sgd', SGD)


    for l in gene_lines:
        if l[0] == '#':
            continue
        l = l.split(' - ')
        match l[0]:
            case 'UNIQUE-ID':
                curr = None
                types = []
            case 'TYPES':
                types.append(l[1])
            case 'ACCESSION-1':
                curr = l[1]
            case 'ACCESSION-2':
                if l[1] in sgd_id_dict:
                    curr = sgd_id_dict[l[1]]
                elif l[1][:2] == 'S0':
                    print(f'query sgd for {l[1]}')
                    url = f'https://yeastgenome.org/backend/locus/{l[1]}'
                    r = requests.get(url)
                    if r.ok:
                        d = r.json()
                        curr = d['format_name']
                        sgd_id_dict[l[1]] = curr
                    else:
                        print(l[1] +' missing')

            case 'PRODUCT':
                gene_prod_dict[l[1]] = curr
                # if l[1] not in id_dict and curr != None:
                #     id_dict[l[1]] = [curr]
            case '//':
                # gene_types[curr] = types
                for t in types:
                    prot_kg.add((KG[curr], RDFS.subClassOf, BC[t]))
                curr = None
    # %%
    with open(os.path.join(BASE, 'data/id_dict_ext.pkl'), 'wb') as fo:
        pickle.dump(id_dict, fo)
    with open(os.path.join(BASE, 'data/sgd2sys_dict.pkl'), 'wb') as fo:
        pickle.dump(sgd_id_dict, fo)

    sys_dict = {}
    print('find sys_dict')
    for k,v in id_dict.items():
        if len(v) > 1:
            q = f"""sgd:{v[0]} rdfs:comment ?link .
                    }}"""
            query = prefixes + q
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            p_group = results['results']['bindings'][0]['link']['value'][:3] + 'X'
            prot_kg.add((KG[p_group], RDFS.subClassOf, SGD.ORF))
            for g in v:
                res = list(kg.subjects(OBOOWL.id, rdflib.Literal(g)))[0]
                prot_kg.add((res, RDFS.subClassOf, KG[p_group]))
            sys_dict[k] = p_group
        else:
            res = list(kg.subjects(OBOOWL.id, rdflib.Literal(v[0])))
            if len(res) > 0:
                sys_name = str(res[0]).split('#')[-1]
                sys_dict[k] = sys_name
            else:
                print(f"{k}, {v} not found")

    for l in lines:
        if l[0] == '#':
            continue
        l = l.split(' - ')
        match l[0]:
            case 'UNIQUE-ID':
                curr = l[1]
            case 'GENE':
                if curr not in sys_dict:
                    if (KG[l[1]], None, None) in kg:
                        sys_dict[curr] = l[1]
                    elif l[1] in sys_dict:
                        sys_dict[curr] = sys_dict[l[1]]

    sys_dict['G3O-34413-RNA'] = 'YNCQ0027W'
    sys_dict['G3O-30361-RNA'] = 'YNCE0024W'

    sys_dict['G3O-29232-RNA'] = 'YNCQ0027W'
    sys_dict['G3O-30361-RNA'] = 'YNCE0024W'

    print('populate protein graph')
    for l in lines:
        if l[0] == '#':
            continue
        l = l.split(' - ')
        match l[0]:
            case 'UNIQUE-ID':
                curr = l[1]
                if curr in gene_prod_dict:
                    trips = role2class(KG[curr], KG[gene_prod_dict[curr]],
                                    SGD.encodedBy)
                    for t in trips:
                        prot_kg.add(t)
                else:
                    print(curr + ' missing??')
            case 'TYPES':
                prot_kg.add((KG[curr], RDFS.subClassOf, BC[l[1]]))
            case 'COMPONENTS':
                part = l[1]
                # check if part encodedBy gene exist, if not, add
                if part in gene_prod_dict:
                    trips = role2class(KG[part], KG[gene_prod_dict[part]],
                                    SGD.encodedBy)
                    for t in trips:
                        prot_kg.add(t)

                    if '-RNA' in part or '-rRNA' in part:
                        prot_kg.add((KG[part], RDFS.subClassOf, BC.Polypeptides))
                
                trips = role2class(KG[curr], KG[part], SGD.hasGeneProductPart)
                for t in trips:
                    prot_kg.add(t)

    # %%
    prot_kg.serialize(os.path.join(BASE, 'graphs/proteins.ttl'))
    # %%
    with open(os.path.join(BASE, 'data/sys_name_dict.pkl'), 'wb') as fo:
        pickle.dump(sys_dict, fo)
    # %%
    with open(os.path.join(BASE, 'data/sgd_id_dict.pkl'), 'wb') as fo:
        pickle.dump(id_dict, fo)
    print('proteins.ttl saved')
    return prot_kg

# %%
if __name__ == '__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(BASE)
    kg = rdflib.Graph()
    kg.parse(os.path.join(BASE, 'graphs/kg-nf-no-int.ttl'))

    generate_proteins(BASE, kg)
# %%
