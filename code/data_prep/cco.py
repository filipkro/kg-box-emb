# %%
import rdflib
from rdflib.namespace import RDFS, OWL
import re, os

BC = rdflib.Namespace('http://biocyc.project-genesis.io#')
OBOOWL = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')
GO = rdflib.Namespace('http://purl.obolibrary.org/obo/GO_')
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')

def generate_cco(BASE, go):

    def find_label_query(label):
        r = list(go.subjects(RDFS.label, rdflib.Literal(label)))
        if r:
            r = [a for a in r if (a, OWL.deprecated, rdflib.Literal(True))
                not in go]
            return r
        r = list(go.subjects(RDFS.label, rdflib.Literal(label.lower())))
        if r:
            r = [a for a in r if (a, OWL.deprecated, rdflib.Literal(True))
                not in go]
            return r
        r = list(go.subjects(OBOOWL.hasExactSynonym, rdflib.Literal(label)))
        if r:
            r = [a for a in r if (a, OWL.deprecated, rdflib.Literal(True))
                not in go]
            return r
        r = list(go.subjects(OBOOWL.hasRelatedSynonym, rdflib.Literal(label)))
        if r:
            r = [a for a in r if (a, OWL.deprecated, rdflib.Literal(True))
                not in go]
            return r
        
    def find_label(label):
        r = find_label_query(label)
        if r:
            return r
        if '-' in label:
            r = find_label_query(label.replace('-', ' '))
            if r:
                return r
            r = find_label_query(label.replace('-', ''))
            if r:
                return r
        if '_' in label:
            r = find_label_query(label.replace('_', ' '))
            if r:
                return r
            r = find_label_query(label.replace('_', ''))
            if r:
                return r
        if label[:2] == 'a ':
            r = find_label_query(label[2:])
            if r:
                return r
        if label[:3] == 'an ':
            r = find_label_query(label[3:])
            if r:
                return r
        r = find_label_query('a ' + label.replace('-', ' ').lower())
        if r:
            return r
        r = find_label_query('a ' + label.replace('-', '').lower())
        if r:
            return r
        return []

# %%
    ont = rdflib.Graph()
    ont.bind('go', GO)
    ont.bind('obo', OBO)
    ont.bind('bc', BC)
    ont.bind('oboInOwl', OBOOWL)
    ont.bind('rdfs', RDFS)
    ont.bind('owl', OWL)

    lines = open(os.path.join(BASE, 'data/classes.dat'), 'r').read().splitlines()
    skip = -1
    for i, line in enumerate(lines):
        line = line.rstrip().split(' - ')
        text = re.sub('<\/?[ibu]>|<\/?su[bp]>', '', ' - '.join(line[1:]))
        match line[0]:
            case 'UNIQUE-ID':
                curr = BC[line[1]]
            case 'TYPES':
                if 'CCO-' not in str(curr):
                    continue
                if text == 'FRAMES':
                    continue
                sup = GO['0110165'] if text == 'CCO' else BC[text]
                ont.add((curr, RDFS.subClassOf, sup))
            case 'COMMON-NAME':
                if 'CCO-' not in str(curr):
                    continue
                ont.add((curr, OBOOWL.hasRelatedSynonym, rdflib.Literal(text)))
                res = find_label(text)
                if len(res) == 1:
                    ont.add((curr, RDFS.subClassOf, rdflib.URIRef(res[0])))
                elif len(res) > 1:
                    print(curr)
                    print(res)
                    print(text)
                    raise NotImplementedError()
            case 'SYNONYMS':
                if 'CCO-' not in str(curr):
                    continue
                ont.add((curr, OBOOWL.hasRelatedSynonym, rdflib.Literal(text)))
                res = find_label(text)
                if len(res) == 1:
                    ont.add((curr, RDFS.subClassOf, rdflib.URIRef(res[0])))
            case '//':
                curr = None

    ont.add((BC['CCO-IN'], RDFS.subClassOf, GO['0110165']))
    ont.add((BC['CCO-OUT'], RDFS.subClassOf, GO['0110165']))

    ont.serialize(destination=os.path.join(BASE, 'graphs/cco.ttl'))
    print("cco.ttl saved")
    return ont

# %%
if __name__ == '__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    go = rdflib.Graph()
    go.parse(os.path.join('graphs/go-ext.ttl'))
    generate_cco(BASE, go)