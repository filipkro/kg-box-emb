# %%
import rdflib
from rdflib.namespace import RDFS, RDF, OWL
import pickle, os
import requests
import browsercookie
cj = browsercookie.firefox()

KG = rdflib.Namespace('http://sgd-kg.project-genesis.io#')
BC = rdflib.Namespace('http://biocyc.project-genesis.io#')
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')
SGD = rdflib.Namespace('http://www.semanticweb.org/filipkro/ontologies/2023/10/sgd_kg#')

def role2class(a, b, r):
    bn = rdflib.BNode()
    trips = []
    trips.append((bn, RDF.type, OWL.Restriction))
    trips.append((bn, OWL.onProperty, r))
    trips.append((bn, OWL.someValuesFrom, b))
    trips.append((a, RDFS.subClassOf, bn))
    return trips
# %%
def generate_reactions(BASE, cco):

    with open(os.path.join(BASE, 'data/sys_name_dict.pkl'), 'rb') as fi:
        id_dict = pickle.load(fi)

    with open(os.path.join(BASE, 'data/enzrxns.dat'), 'r') as fi:
        lines = fi.read().splitlines()

    enzrxn_dict = {}
    for l in lines:
        if l[0] == '#':
            continue
        l = l.split(' - ')
        match l[0]:
            case 'UNIQUE-ID':
                curr = l[1]
            case 'ENZYME':
                enzrxn_dict[curr] = l[1]

# %%
    with open(os.path.join(BASE, 'data/enzrxn_dict.pkl'), 'wb') as fo:
        pickle.dump(enzrxn_dict, fo)

    with open(os.path.join(BASE, 'data/reactions.dat'), 'r') as fi:
        lines = fi.read().splitlines()

    react_kg = rdflib.Graph()
    react_kg.bind('kg', KG)
    react_kg.bind('obo', OBO)
    react_kg.bind('bc', BC)
    react_kg.bind('rdfs', RDFS)
    react_kg.bind('owl', OWL)
    react_kg.bind('rdf', RDF)
    react_kg.bind('sgd', SGD)

    cat_kg = rdflib.Graph()
    cat_kg.bind('kg', KG)
    cat_kg.bind('obo', OBO)
    cat_kg.bind('bc', BC)
    cat_kg.bind('rdfs', RDFS)
    cat_kg.bind('owl', OWL)
    cat_kg.bind('rdf', RDF)
    cat_kg.bind('sgd', SGD)
    print(f'populate graph, len: {len(lines)}')
    for l in lines:
        if l[0] == '#':
            continue
        l = l.split(' - ')
        match l[0]:
            case 'UNIQUE-ID':
                curr = l[1]
                r_dict = {'left': [], 'right': []}
            case 'TYPES':
                react_kg.add((KG[curr], RDFS.subClassOf, BC[l[1]]))
            case 'ENZYMATIC-REACTION':
                enz = enzrxn_dict[l[1]] if l[1] in enzrxn_dict else l[1]
                trips = role2class(KG[curr], KG[enz], SGD.catalyzedBy)
                for t in trips:
                    cat_kg.add(t)
            case 'RXN-LOCATIONS':
                if not (BC[l[1]], None, None) in cco:
                    locs = l[1].split('-CCO')
                    assert len(locs) > 1
                    for loc in locs:
                        if loc == 'CCI-GOLGI-LUM-FUN':
                            loc = 'CCO-GOLGI-LUM'
                        if 'CCO' not in loc:
                            loc = 'CCO' + loc
                        cco.add((BC[l[1]], RDFS.subClassOf, BC[loc]))
                trips = role2class(KG[curr], BC[l[1]], OBO.BFO_0000066)
                for t in trips:
                    react_kg.add(t)
            case 'COMMON-NAME':
                react_kg.add((KG[curr], RDFS.label, rdflib.Literal(l[1])))
            case 'LEFT':
                # add to dict
                r_dict['left'].append(l[1])
            case 'RIGHT':
                # add to dict
                r_dict['right'].append(l[1])
            case 'REACTION-DIRECTION':
                # add to dict
                if 'REVERSIBLE' in l[1]:
                    r_dict['dir'] = 'rev'
                elif l[1].find('LEFT') < l[1].find('RIGHT'):
                    r_dict['dir'] = 'left'
                elif l[1].find('LEFT') > l[1].find('RIGHT'):
                    r_dict['dir'] = 'right'
            case '//':
                try:
                    direction = r_dict['dir']
                except KeyError:
                    r = requests.get(f'https://biocyc.org/reaction?orgid=META&id={curr}',
                                    cookies=cj)
                    if r.ok:
                        res = r.text[r.text.find('Equation'):
                                    r.text.find('Enzymes and Genes')].split('\n\n\n')[0]
                        if '&rarr;' in res:
                            direction = 'left'
                        elif '&larr;' in res:
                            direction = 'right'
                        elif '&harr;' in res or '=' in res:
                            direction = 'rev'
                        else:
                            direction = 'rev'
                trips = []
                if direction == 'rev':
                    for p in r_dict['right']:
                        trips.extend(role2class(KG[curr], BC[p], SGD.hasRightParticipant))
                        # react_kg.add(())
                    for p in r_dict['left']:
                        trips.extend(role2class(KG[curr], BC[p], SGD.hasLeftParticipant))
                elif direction == 'left':
                    for p in r_dict['left']:
                        trips.extend(role2class(KG[curr], BC[p], OBO.RO_0002233))
                    for p in r_dict['right']:
                        trips.extend(role2class(KG[curr], BC[p], OBO.RO_0002234))
                elif direction == 'right':
                    for p in r_dict['right']:
                        trips.extend(role2class(KG[curr], BC[p], OBO.RO_0002233))
                        # react_kg.add(())
                    for p in r_dict['left']:
                        trips.extend(role2class(KG[curr], BC[p], OBO.RO_0002234))
                    
                for t in trips:
                    react_kg.add(t)

    # %%
    print('saving graphjs')
    react_kg.serialize(destination=os.path.join(BASE, 'graphs/reactions.ttl'))
    cat_kg.serialize(destination=os.path.join(BASE, 'graphs/catalyzedBy.ttl'))
    # %%
    cco.serialize(destination=os.path.join(BASE, 'graphs/cco.ttl'))

    print('reactions.ttl saved')
    print('catalyzedBy.ttl saved')
    print('cco.ttl potentially updated')

    return react_kg, cco, cat_kg


if __name__ == '__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cco = rdflib.Graph()
    cco.parse(os.path.join(BASE, 'graphs/cco.ttl'))
    print(f"cco len: {len(cco)}")
    generate_reactions(BASE, cco)