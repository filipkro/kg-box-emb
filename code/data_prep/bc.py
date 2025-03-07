# %%
import rdflib
from rdflib.namespace import RDFS, OWL, RDF
import re, os

BC = rdflib.Namespace('http://biocyc.project-genesis.io#')
OBOOWL = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')
CHEBI = rdflib.Namespace('http://purl.obolibrary.org/obo/CHEBI_')
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')

def get_children(ont, term):
    found = []
    children = list(ont.subjects(RDFS.subClassOf, term))
    found.extend(children)
    for c in children:
        found.extend(get_children(ont, c))
    return found

def in_role(chebi, term):
    q = f"""PREFIX obo: <{str(OBO)}>
    PREFIX rdfs: <{str(RDFS)}>
    ASK {{
        <{term}> rdfs:subClassOf* obo:CHEBI_50906
    }}"""
    return chebi.query(q)


def generate_bc(BASE, chebi):

    def find_label_query(label):
        r = list(chebi.subjects(RDFS.label, rdflib.Literal(label)))
        if r:
            return r
        r = list(chebi.subjects(RDFS.label, rdflib.Literal(label.lower())))
        if r:
            return r
        r = list(chebi.subjects(OBOOWL.hasExactSynonym, rdflib.Literal(label)))
        if r:
            return r
        r = list(chebi.subjects(OBOOWL.hasRelatedSynonym, rdflib.Literal(label)))
        if r:
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


    ont = rdflib.Graph()
    ont.bind('chebi', CHEBI)
    ont.bind('obo', OBO)
    ont.bind('bc', BC)
    ont.bind('oboInOwl', OBOOWL)
    ont.bind('rdfs', RDFS)
    ont.bind('rdf', RDF)
    ont.bind('owl', OWL)

    regulation_dict = {'Regulation-of-Enzyme-Activity': OBO['INO_0000055'], 'Regulation-of-Gene-Products': OBO['INO_0000157'], 'Regulation-of-Reactions': OBO['INO_0000157'], 'Regulation-of-Transcription': OBO['INO_0000032'], 'Regulation-of-Translation': OBO['INO_0000034']}

    lines = open(os.path.join(BASE, 'data/classes.dat'), 'r').readlines()
    skip = -1
    chem_entities = [BC.Chemicals, BC['Polymer-Segments']]
    ont.add((BC['Generalized-Reactions'], RDFS.subClassOf, OBO.BFO_0000015))
    for i, line in enumerate(lines):
        if i <= skip:
            continue
        line = line.rstrip().split(' - ')
        text = re.sub('<\/?[ibu]>|<\/?su[bp]>', '', ' - '.join(line[1:]))
        match line[0]:
            case 'UNIQUE-ID':
                curr = BC[line[1]]
            case 'TYPES':
                if 'GO:' in str(curr) or 'CCO' in str(curr):
                    continue
                if curr in chem_entities:
                    ont.add((curr, RDFS.subClassOf, CHEBI['24431']))
                elif str(curr).split('#')[-1] in regulation_dict:
                    ont.add((curr, RDFS.subClassOf,
                            regulation_dict[str(curr).split('#')[-1]]))
                elif line[1] == 'Regulation':
                    ont.add((curr, RDFS.subClassOf, OBO['INO_0000157']))
                elif line[1] != 'FRAMES':
                    ont.add((curr, RDFS.subClassOf, BC[line[1]]))
            case 'COMMON-NAME':
                if 'GO:' in str(curr) or 'CCO' in str(curr):
                    continue
                ont.add((curr, RDFS.label, rdflib.Literal(text)))
            case 'COMMENT':
                if 'GO:' in str(curr) or 'CCO' in str(curr):
                    continue
                comment = text
                while lines[i+1][0] == '/' and lines[i+1][:2] != '//':
                    i += 1   
                    comment = comment + '\n' + re.sub('<\/?[ibu]>|<\/?su[bp]>',
                                                    '', lines[i][1:])
                ont.add((curr, RDFS.comment, rdflib.Literal(comment)))
                skip = i
            case 'SYNONYMS':
                if 'GO:' in str(curr) or 'CCO' in str(curr):
                    continue
                ont.add((curr, OBOOWL.hasRelatedSynonym, rdflib.Literal(text)))
            case 'ABBREV-NAME':
                if 'GO:' in str(curr) or 'CCO' in str(curr):
                    continue
                ont.add((curr, OBOOWL.shorthand, rdflib.Literal(text)))
            case '//':
                curr = None

# %%
    in_chebi = {}
    lines = open(os.path.join(BASE, 'data/compounds.dat'), 'r').readlines()
    skip = -1
    curr = None
    print(len(ont))
    for i, line in enumerate(lines):
        line = line.rstrip().split(' - ')
        text = re.sub('<\/?[ibu]>|<\/?su[bp]>', '', ' - '.join(line[1:]))
        match line[0]:
            case 'UNIQUE-ID':
                curr = BC[line[1]]
            case 'TYPES':
                if 'GO:' in str(curr):
                    continue
                if curr in chem_entities:
                    ont.add((curr, RDFS.subClassOf, CHEBI['24431']))
                elif line[1] != 'FRAMES':
                    ont.add((curr, RDFS.subClassOf, BC[line[1]]))
            case 'SYNONYMS':
                if 'GO:' in str(curr):
                    continue
                ont.add((curr, OBOOWL.hasRelatedSynonym, rdflib.Literal(text)))
            case 'SYSTEMATIC-NAME':
                if 'GO:' in str(curr):
                    continue
                ont.add((curr, OBOOWL.hasExactSynonym, rdflib.Literal(text)))
            case 'INCHI':
                if 'GO:' in str(curr):
                    continue
                ont.add((curr, OBO['chebi/inchi'], rdflib.Literal(text)))
            case 'INCHI-KEY':
                if 'GO:' in str(curr):
                    continue
                ont.add((curr, OBO['chebi/inchi'],
                        rdflib.Literal('='.join(text.split('=')[1:]))))
            case 'DBLINKS':
                if 'GO:' in str(curr):
                    continue
                if not (curr, None, None) in ont:
                    print(curr)
                    raise NotImplementedError()
                if 'CHEBI' in line[1]:
                    ch = CHEBI[line[1].split('"')[1]]
                    if not (ch, OWL.deprecated, rdflib.Literal(True)) in chebi:
                        if str(curr) in in_chebi:
                            in_chebi[str(curr)].append(ch)
                        else:
                            in_chebi[str(curr)] = [ch]
            case '//':
                curr = None
    print(len(ont))
# %%
    all_chemicals = get_children(ont, CHEBI['24431'])
    all_chemicals.extend(get_children(ont, BC['Polymer-Segments']))
    print(len(all_chemicals))
# %%
    for c in all_chemicals:
        if str(c) in in_chebi:
            continue
        if c == BC.PECTATE:
            in_chebi[str(c)] = [CHEBI['68837']]
            continue

        mc = 'MetaCyc:' + str(c).split('#')[-1]
        ch = list(chebi.subjects(OBOOWL.hasDbXref, rdflib.Literal(mc)))
        if ch:
            in_chebi[str(c)] = ch
            continue
    print(len(in_chebi))

# %%
    for b in all_chemicals:
        b = str(b)
        if b in in_chebi:
            continue

        label = b.split('#')[-1]
        r = find_label(label)
        
        if r:
            in_chebi[b] = r
            continue

        inchi = list(ont.objects(rdflib.URIRef(b), OBO['chebi/inchi']))
        r = []
        for i in inchi:
            r.extend(list(chebi.subjects(OBO['chebi/inchi'], rdflib.Literal(i))))
        if r:
            in_chebi[b] = list(set(r))
            continue
        inchi = list(ont.objects(rdflib.URIRef(b), OBO['chebi/inchikey']))
        r = []
        for i in inchi:
            r.extend(list(chebi.subjects(OBO['chebi/inchikey'],
                                        rdflib.Literal(i))))
        if r:
            in_chebi[b] = list(set(r))
            continue

        labels = list(ont.objects(rdflib.URIRef(b), RDFS.label))
        if labels:
            r = []
            for label in labels:
                r.extend(find_label(label))
            if r:
                in_chebi[b] = list(set(r))
                continue

        labels = list(ont.objects(rdflib.URIRef(b), OBOOWL.hasExactSynonym))
        if not labels:
            r = []
            for label in labels:
                r.extend(find_label(label))
            if r:
                in_chebi[b] = list(set(r))
                continue

        labels = list(ont.objects(rdflib.URIRef(b), OBOOWL.hasRelatedSynonym))
        if not labels:
            r = []
            for label in labels:
                r.extend(find_label(label))
            if r:
                in_chebi[b] = list(set(r))
                continue

        labels = list(ont.objects(rdflib.URIRef(b), OBOOWL.shorthand))
        if labels:
            r = []
            for label in labels:
                r.extend(find_label(label))
            if r:
                in_chebi[b] = list(set(r))
                continue

    print(len(in_chebi))
# %%
    simplified_chebi = {}
    for k in in_chebi:
        v = in_chebi[k]
        if len(v) == 1:
            if not in_role(chebi, str(v[0])):
                simplified_chebi[k] = str(v[0])
            continue

        q = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?a
    WHERE {
    """
        for ch in v:
            if not in_role(chebi, str(ch)):
                cid = str(ch).split('_')[-1]
                q = q + f"\tchebi:{cid} rdfs:subClassOf* ?a .\n"
        q = q + "} LIMIT 1"
        try:
            res = list(chebi.query(q))
            res = str(res[0][0])
        except IndexError:
            print(res)
            print(k)
            print(in_chebi[k])
            print(v)
            print(len(v))
            raise IndexError()
        if res != str(CHEBI['24431']):
            simplified_chebi[k] = res
    len(simplified_chebi)

# %%
    for k, v in simplified_chebi.items():
        ont.add((rdflib.URIRef(k), RDFS.subClassOf, rdflib.URIRef(v)))
# %%
# missing compounds
    ont.add((BC['CPD-20741'], RDFS.subClassOf,
            BC['Adjacent-pyrimidine-dimer-in-DNA']))
    ont.add((BC['CPD-20742'], RDFS.subClassOf,
            BC['Adjacent-pyrimidine-dimer-in-DNA']))
    ont.add((BC['CPD-20743'], RDFS.subClassOf,
            BC['Adjacent-pyrimidine-dimer-in-DNA']))
    ont.add((BC['CPD-20744'], RDFS.subClassOf,
            BC['Adjacent-pyrimidine-dimer-in-DNA']))
    ont.add((BC['E-'], RDFS.subClassOf, CHEBI['10545']))
# %%
    ont.serialize(destination=os.path.join(BASE, 'graphs/bc-fix.ttl'))
    print('bc.ttl saved')
    return ont



if __name__ == '__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(BASE)
    chebi = rdflib.Graph()
    chebi.parse(os.path.join(BASE, 'graphs/chebi.ttl'))
    generate_bc(BASE, chebi)
