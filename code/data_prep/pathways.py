# %%
import rdflib
from rdflib.namespace import RDFS, RDF, OWL
import os, json

KG = rdflib.Namespace('http://sgd-kg.project-genesis.io#')
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

def generate_pathways(BASE):
    def query4rel(a, r):
        q = f"""
    PREFIX owl: <{str(OWL)}>
    PREFIX rdfs: <{str(RDFS)}>
    SELECT DISTINCT ?b
    WHERE
    {{
        ?z owl:onProperty <{r}> .
        <{a}> rdfs:subClassOf ?z .
        ?z owl:someValuesFrom ?b .
    }}"""
        
        return list(kg.query(q))

    with open(os.path.join(BASE, 'data/pathways.dat'), 'r') as fi:
        lines = fi.read().splitlines()

    with open(os.path.join(BASE, 'data/cco_map.json'), 'r') as fi:
        cco_map = json.load(fi)

    with open(os.path.join(BASE, 'data/chebi_map.json'), 'r') as fi:
        chebi_map = json.load(fi)

    # %%
    kg = rdflib.Graph()

    kg.bind('kg', KG)
    kg.bind('obo', OBO)
    kg.bind('bc', BC)
    kg.bind('rdf', RDF)
    kg.bind('rdfs', RDFS)
    kg.bind('owl', OWL)

    super_pathways = set()
    used_chemicals = []
    for l in lines:
        if l[0] == '#':
            continue
        l = l.split(' - ')
        match l[0]:
            case 'UNIQUE-ID':
                curr = l[1]
                inputs = set()
                outputs = set()
            case 'TYPES':
                kg.add((KG[curr], RDFS.subClassOf, BC[l[1]]))
            case 'COMMON-NAME':
                kg.add((KG[curr], RDFS.label, rdflib.Literal(l[1])))
            case 'REACTION-LAYOUT':
                if (KG[curr], RDFS.subClassOf, BC['Super-Pathways']) in kg:
                    continue
                react = l[1].split(' ')[0][1:]
                # trips = role2class(KG[curr], KG[react], OBO.BFO_0000051)
                trips = role2class(KG[curr], KG[react], OBO.RO_0000057)
                for t in trips:
                    kg.add(t)
                left = l[1].split(':LEFT-PRIMARIES ')[1].split(')')[0].split(' ')
                right = l[1].split(':RIGHT-PRIMARIES ')[1].split(')')[0].split(' ')
                used_chemicals.extend(left)
                used_chemicals.extend(right)
                if 'L2R' in l[1]:
                    inputs.update(left)
                    outputs.update(right)
                elif 'R2L' in l[1]:
                    inputs.update(right)
                    outputs.update(left)
            case '//':
                if (KG[curr], RDFS.subClassOf, BC['Super-Pathways']) in kg:
                    super_pathways.add(curr)
                else:
                    start = inputs - outputs
                    end = outputs - inputs

                    trips = []
                    for c in start:
                        # trips.extend(role2class(KG[curr], BC[c], OBO.RO_0002233))
                        trips.extend(role2class(KG[curr], OBO[chebi_map[c]], OBO.RO_0002233))
                    for c in end:
                        # trips.extend(role2class(KG[curr], BC[c], OBO.RO_0002234))
                        trips.extend(role2class(KG[curr], OBO[chebi_map[c]], OBO.RO_0002234))

                    for t in trips:
                        kg.add(t)

    # %%
    skip = False
    while len(super_pathways) > 0:
        print(len(super_pathways))
        for l in lines:
            if l[0] == '#':
                continue
            l = l.split(' - ')
            match l[0]:
                case 'UNIQUE-ID':
                    curr = l[1]
                    inputs = set()
                    outputs = set()
                    skip = False
                case 'REACTION-LAYOUT':
                    if curr not in super_pathways or skip:
                        continue
                    react = l[1].split(' ')[0][1:]
                    if react in super_pathways:
                        skip = True
                        continue
                    trips = role2class(KG[curr], KG[react], OBO.RO_0000057)
                    for t in trips:
                        kg.add(t)
                    try:
                        left = l[1].split(':LEFT-PRIMARIES ')[1].split(')')[0].split(' ')
                        right = l[1].split(':RIGHT-PRIMARIES ')[1].split(')')[0].split(' ')
                        if 'L2R' in l[1]:
                            inputs.update(left)
                            outputs.update(right)
                        elif 'R2L' in l[1]:
                            inputs.update(right)
                            outputs.update(left)
                    except IndexError:
                        inp = query4rel(KG[react], OBO.RO_0002233)
                        for c in inp:
                            inputs.add(str(c[0]).split('#')[-1])
                        out = query4rel(KG[react], OBO.RO_0002234)
                        for c in out:
                            outputs.add(str(c[0]).split('#')[-1])

                case '//':
                    if curr not in super_pathways or skip:
                        continue
                    start = inputs - outputs
                    end = outputs - inputs

                    trips = []
                    for c in start:
                        if 'CHEBI' not in str(c):
                            c = OBO[chebi_map[c]]
                        else:
                            c = rdflib.URIRef(c)
                        # trips.extend(role2class(KG[curr], BC[c], OBO.RO_0002233))
                        trips.extend(role2class(KG[curr], c, OBO.RO_0002233))
                    for c in end:
                        if 'CHEBI' not in str(c):
                            c = OBO[chebi_map[c]]
                        else:
                            c = rdflib.URIRef(c)
                        # trips.extend(role2class(KG[curr], BC[c], OBO.RO_0002234))
                        trips.extend(role2class(KG[curr], c, OBO.RO_0002234))

                    for t in trips:
                        kg.add(t)

                    super_pathways.remove(curr)
    # %%
    print(f'number of chems used: {len(used_chemicals)}')
    used_chemicals = list(set(used_chemicals))
    print(f'number of unique chems used: {len(used_chemicals)}')
    # with open(os.path.join(BASE, 'data/chems_in_paths.pkl'), 'w') as fo:
    #     json.dump(used_chemicals, fo)
    kg.serialize(os.path.join(BASE, 'graphs/pathways.ttl'))
    print('pathways.ttl saved')
    return kg
# %%

if __name__ == '__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(BASE)
    generate_pathways(BASE)
# %%
