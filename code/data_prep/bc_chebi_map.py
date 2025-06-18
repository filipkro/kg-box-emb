# %%
import rdflib
from rdflib.namespace import RDFS, OWL, RDF
import openai
import re, os, json
import requests
import xml.etree.ElementTree as ET
from OPENAI_KEY import API_KEY
# %%
BC = rdflib.Namespace('http://biocyc.project-genesis.io#')
OBOOWL = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')
CHEBI = rdflib.Namespace('http://purl.obolibrary.org/obo/CHEBI_')
OBO = rdflib.Namespace('http://purl.obolibrary.org/obo/')
client = openai.OpenAI(api_key=API_KEY)
# %%
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

def find_label_query(label):
    r = list(chebi.subjects(RDFS.label, rdflib.Literal(label)))
    if r:
        return r
    r = list(chebi.subjects(RDFS.label, rdflib.Literal(label.lower())))
    if r:
        return r
    r = list(chebi.subjects(RDFS.label, rdflib.Literal(label.capitalize())))
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
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE)
chebi = rdflib.Graph()
# chebi.parse(os.path.join(BASE, 'graphs/chebi.ttl'))
with open(os.path.join(BASE, 'data/chems_in_reacts.json'), 'r') as fi:
    chems_of_interest = json.load(fi)
# generate_bc(BASE, chebi, chems_of_interest)



# def generate_bc(BASE, chebi, chems_of_interest):

    


ont = rdflib.Graph()
ont.bind('chebi', CHEBI)
ont.bind('obo', OBO)
ont.bind('bc', BC)
ont.bind('oboInOwl', OBOOWL)
ont.bind('rdfs', RDFS)
ont.bind('rdf', RDF)
ont.bind('owl', OWL)

regulation_dict = {'Regulation-of-Enzyme-Activity': OBO['INO_0000055'], 'Regulation-of-Gene-Products': OBO['INO_0000157'], 'Regulation-of-Reactions': OBO['INO_0000157'], 'Regulation-of-Transcription': OBO['INO_0000032'], 'Regulation-of-Translation': OBO['INO_0000034']}
# %%
lines = open(os.path.join(BASE, 'data/classes.dat'), 'r').readlines()
skip = -1
chem_entities = [BC.Chemicals, BC['Polymer-Segments']]
ont.add((BC['Generalized-Reactions'], RDFS.subClassOf, OBO.BFO_0000015))
counter = 0
in_classes = []
classes_data = {}
oi = False
label = None
comment = None
synonym = None
for i, line in enumerate(lines):
    if i <= skip:
        continue
    line = line.rstrip().split(' - ')
    text = re.sub('<\/?[ibu]>|<\/?su[bp]>', '', ' - '.join(line[1:]))
    
    match line[0]:
        case 'UNIQUE-ID':
            curr = line[1]
            counter += int(line[1] in chems_of_interest)
            if curr in chems_of_interest:
                if curr not in classes_data:
                    classes_data[curr] = []
                    for _ in range(8):
                        classes_data[curr].append([])
                in_classes.append(line[1])
            
        case 'COMMON-NAME':
            if curr in chems_of_interest:
                classes_data[curr][3].append(text)
        case 'SYSTEMATIC-NAME':
            if curr in chems_of_interest:
                classes_data[curr][5].append(text)
            
        case 'COMMENT':
            if 'GO:' in str(curr) or 'CCO' in str(curr):
                continue
            comment = text
            while lines[i+1][0] == '/' and lines[i+1][:2] != '//':
                i += 1   
                comment = comment + '\n' + re.sub('<\/?[ibu]>|<\/?su[bp]>',
                                                '', lines[i][1:])
            if curr in chems_of_interest:
                classes_data[curr][7].append(text)
        case 'SYNONYMS':
            if curr in chems_of_interest:
                classes_data[curr][4].append(text)
        case 'ABBREV-NAME':
            if curr in chems_of_interest:
                classes_data[curr][6].append(text)
        case '//':
            curr = None
print(counter)

# %%
in_chebi = {}
lines = open(os.path.join(BASE, 'data/compounds.dat'), 'r').readlines()
skip = -1
curr = None
print(len(ont))
in_compounds = []
oi = False
oi_w_chebi = []
oi_w_inchi = []
for i, line in enumerate(lines):
    line = line.rstrip().split(' - ')
    text = re.sub('<\/?[ibu]>|<\/?su[bp]>', '', ' - '.join(line[1:]))
    match line[0]:
        case 'UNIQUE-ID':
            curr = line[1]
            if curr in chems_of_interest:
                in_compounds.append(line[1])
                if curr not in classes_data:
                    classes_data[curr] = []
                    for _ in range(8):
                        classes_data[curr].append([])

        case 'COMMON-NAME':
            if curr in chems_of_interest:
                classes_data[curr][3].append(text)
            
        case 'COMMENT':
            if 'GO:' in str(curr) or 'CCO' in str(curr):
                continue
            comment = text
            while lines[i+1][0] == '/' and lines[i+1][:2] != '//':
                i += 1   
                comment = comment + '\n' + re.sub('<\/?[ibu]>|<\/?su[bp]>',
                                                '', lines[i][1:])
            if curr in chems_of_interest:
                classes_data[curr][7].append(text)
        case 'SYSTEMATIC-NAME':
            if curr in chems_of_interest:
                classes_data[curr][5].append(text)
        case 'SYNONYMS':
            if curr in chems_of_interest:
                classes_data[curr][4].append(text)
        case 'ABBREV-NAME':
            if curr in chems_of_interest:
                classes_data[curr][6].append(text)
        case 'INCHI':
            if curr in chems_of_interest:
                classes_data[curr][1].append(text)
        case 'INCHI-KEY':
            if curr in chems_of_interest:
                classes_data[curr][2].append('='.join(text.split('=')[1:]))
        case 'DBLINKS':
            if 'CHEBI' in line[1] and curr in chems_of_interest:
                classes_data[curr][0].append('CHEBI_' + line[1].split('"')[1])
        case '//':
            curr = None
# print(len(ont))
# %%
namespaces = {
    'S': 'http://schemas.xmlsoap.org/soap/envelope/',
    'chebi': 'https://www.ebi.ac.uk/webservices/chebi'
}
def search_chebi_candidates(description, maximum_results=20, include_star=False):
    url = "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getLiteEntity"
    params = {
        'search': description,
        'maximumResults': maximum_results,
        'searchCategory': 'ALL',
        'starsCategory': 'ALL'
    }
    response = requests.get(url, params=params)
    if response.ok:
        root = ET.fromstring(response.text)
        elements = root.findall('.//chebi:ListElement', namespaces)
        results = []
        for el in elements:
            chebi_id = el.find('chebi:chebiId', namespaces).text
            name = el.find('chebi:chebiAsciiName', namespaces).text
            score = el.find('chebi:searchScore', namespaces).text
            star = el.find('chebi:entityStar', namespaces).text
            results.append({
                'chebiId': chebi_id,
                'chebiAsciiName': name,
                'searchScore': float(score)
            })
            if include_star:
                results[-1]['entityStar'] = int(star)
    else:
        print(response)
        results = None

    return results

def find_in_chebi(description):
    # print(description)
    a = search_chebi_candidates(description, maximum_results=10)
    # print(a)

    response = client.responses.create(
        model="gpt-4o",
        instructions=f"From the list of chebi entries, which includes scores which are a measure of string matching, but not necessarily the most semantically correct match, pick the best match to this description: {description}. Return the dictionary with the best match or None if nothing matches.",
        input=str(a),
    )
    # print(response.output_text)
    # print()
    if "chebiId': '" in response.output_text:
        ch = response.output_text.split("chebiId': '")[-1].split("',")[0]
    else:
        ch = None
    return ch
# %%
chebi = rdflib.Graph()
chebi.parse(os.path.join(BASE, 'graphs/chebi.ttl'))
# %%
chebi_map = {}
counter = 0
for ch, entries in classes_data.items():
    counter += 1
    print(f"{counter} / {len(classes_data)}", end='\r')
    # chebi
    if len(entries[0]) > 0:
        if len(entries[0]) > 1:
            
            print('Double: ', ch, entries[0])
            # raise ValueError((ch, entries[0]))
        elif not (OBO[entries[0][0]], OWL.deprecated, rdflib.Literal(True)) in chebi:
            chebi_map[ch] = entries[0][0]
            continue
        else:
            print('Deprecated: ', ch, entries[0])
            
    # continue
    if len(entries[1]) > 0:
        if len(entries[1]) > 1:
            raise ValueError((ch, entries[1]))
        inchi_chebi = list(chebi.subjects(OBO['chebi/inchi'], rdflib.Literal(entries[1][0])))
        if len(inchi_chebi) == 1:
            if not (inchi_chebi[0], OWL.deprecated, rdflib.Literal(True)) in chebi:
                chebi_map[ch] = str(inchi_chebi[0]).split('/')[-1]
                continue
            else:
                print('Deprecated inchi: ', ch, inchi_chebi)
        elif len(inchi_chebi) > 1:
            print('Double inchi: ', ch, inchi_chebi)

    if len(entries[2]) > 0 and False:
        if len(entries[2]) > 1:
            raise ValueError((ch, entries[2]))
        inchi_chebi = list(chebi.subjects(OBO['chebi/inchikey'], rdflib.Literal(entries[2][0])))
        if len(inchi_chebi) == 1:
            if not (inchi_chebi[0], OWL.deprecated, rdflib.Literal(True)) in chebi:
                chebi_map[ch] = str(inchi_chebi[0]).split('/')[-1]
                continue
            print('Deprecated inchikey: ', ch, inchi_chebi)
        elif len(inchi_chebi) > 1:
            print('Double inchikey: ', ch, inchi_chebi)

    # metacyc
    mc = 'MetaCyc:' + str(ch)
    metacyc_chebi = list(chebi.subjects(OBOOWL.hasDbXref, rdflib.Literal(mc)))
    if len(metacyc_chebi) == 1:
        if not (metacyc_chebi[0], OWL.deprecated, rdflib.Literal(True)) in chebi:
            chebi_map[ch] = str(metacyc_chebi[0]).split('/')[-1]
            continue
        else:
            print('Deprecated metacyc: ', ch, metacyc_chebi)
    elif len(metacyc_chebi) > 1:
        print('Double metacyc: ', ch, metacyc_chebi)

    if len(entries[3]) > 0:
        labels = list(set(entries[3]))
        if len(labels) > 1:
            raise ValueError((ch, labels))
        label_chebi = find_label(labels[0])
        if len(label_chebi) == 1:
            if not (label_chebi[0], OWL.deprecated, rdflib.Literal(True)) in chebi:
                chebi_map[ch] = str(label_chebi[0]).split('/')[-1]
                continue
            else:
                print('Deprecated label: ', ch, label_chebi)
        elif len(label_chebi) > 1:
            print('Double label: ', ch, label_chebi)

    if len(entries[5]) > 0:
        labels = list(set(entries[5]))
        if len(labels) > 1:
            raise ValueError((ch, labels))
        label_chebi = find_label(labels[0])
        if len(label_chebi) == 1:
            if not (label_chebi[0], OWL.deprecated, rdflib.Literal(True)) in chebi:
                chebi_map[ch] = str(label_chebi[0]).split('/')[-1]
                continue
            else:
                print('Deprecated systematic name: ', ch, label_chebi)
        elif len(label_chebi) > 1:
            print('Double systematic name: ', ch, label_chebi)

    if len(entries[4]) > 0:
        labels = list(set(entries[4]))
        for lab in labels:
            label_chebi = find_label(lab)
            if len(label_chebi) == 1:
                if not (label_chebi[0], OWL.deprecated, rdflib.Literal(True)) in chebi:
                    chebi_map[ch] = str(label_chebi[0]).split('/')[-1]
                    continue
                else:
                    print('Deprecated synonym name: ', ch, label_chebi)
            elif len(label_chebi) > 1:
                print('Double synonym name: ', ch, label_chebi)
        # if len(labels) == 1:
        #     # raise ValueError((ch, labels))
        #     label_chebi = find_label(labels[0])
        #     if len(label_chebi) == 1:
        #         if not (label_chebi[0], OWL.deprecated, rdflib.Literal(True)) in chebi:
        #             chebi_map[ch] = str(label_chebi[0]).split('/')[-1]
        #             continue
        #         else:
        #             print('Deprecated synonym name: ', ch, label_chebi)
        #     elif len(label_chebi) > 1:
        #         print('Double synonym name: ', ch, label_chebi)
    if not ch in chebi_map:
        desc = ', '.join(list(set(entries[3])) + list(set(entries[4])))
        candidate = find_in_chebi(desc)
        if candidate != None and not (OBO[candidate.replace(':', '_')], OWL.deprecated, rdflib.Literal(True)) in chebi:
            chebi_map[ch] = candidate.replace(':', '_')
print(len(chebi_map))

# %%
chebi_map['CPD-5169'] = 'CHEBI_132521'
chebi_map['CPD-5167'] = 'CHEBI_132519'
chebi_map['CPD-17438'] = 'CHEBI_17515'
# %%
missing = {}
for k,v in classes_data.items():
    if k not in chebi_map:
        missing[k] = v
print(len(missing))
# %%
# manual

from collections import Counter

nbr_matches = 0
matches = []
no_matches = 0
ill_format = 0
reps = 5

for i, (k,v) in enumerate(missing.items()):
    print(f"{i} / {len(missing)}", end='\r')
    for _ in range(reps):
        response = client.responses.create(
                model="gpt-4o",
                tools=[{'type': 'web_search_preview'}],
                tool_choice={'type': 'web_search_preview'},
                instructions="""Anserws should be on the form:
                'Match: <CHEBI-entity>' or 'No match'
                
                followed by a line break and your reasoning steps and source for the information""",
                input=f"""Give me the closest match in chebi, it does not have to be the perfect match, for the compound below. If nothing matches return None. Only give the closest match and only return the chebi code
                
                {missing[k][3][0]}""",
            )
        if 'CHEBI:' in response.output_text:
            nbr_matches += 1

            comp = 'CHEBI:' + response.output_text.split('CHEBI:')[1].split('\n')[0].split(' ')[0].split(',')[0].split('.')[0].split('*')[0]
            matches.append(comp)

    match = Counter(matches).most_common()[0]
    if match[1] > reps/2:
        chebi_map[k] = match[0]
    else:
        print(k)
        print(Counter(matches))

        
# print(response.output_text)
# %%
for k in chems_of_interest:
    if k not in chebi_map:
        print(k)
# %%
# manual
chebi_map['CPD-20741'] = 'CHEBI_116736'
chebi_map['CPD-20743'] = 'CHEBI_165813'
chebi_map['CPD-20744'] = 'CHEBI_165817'
chebi_map['CPD-20742'] = 'CHEBI_75710'
chebi_map['E-'] = 'CHEBI_10545'

# %%
with open(os.path.join(BASE, 'data/chebi_map.json'), 'w') as fo:
    json.dump(chebi_map, fo)
# %%
