# %%
import pickle, os
import rdflib
from rdflib.namespace import RDFS, RDF, OWL

SKIP_GO = True

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

def generate_gene_kg(BASE, go, sgd):
    data_dir = os.path.join(BASE, 'data/')
    graph_dir = os.path.join(BASE, 'graphs/')
    ont = rdflib.Graph()
    # ont += go
    ont += sgd

    def term_from_label(label):
        term = ont.value(predicate=RDFS.label,
                         object=rdflib.Literal(label, datatype=rdflib.URIRef('http://www.w3.org/2001/XMLSchema#string')))
        if term == None:
            term = ont.value(predicate=RDFS.label,
                            object=rdflib.Literal(label, lang='en'))
        if term == None:
            term = ont.value(predicate=RDFS.label,
                            object=rdflib.Literal(label))
        return term

    def term_from_label_go(label):
        term = go.value(predicate=RDFS.label,
                                object=rdflib.Literal(label, datatype=rdflib.URIRef('http://www.w3.org/2001/XMLSchema#string')))
        if term == None:
            term = go.value(predicate=RDFS.label,
                            object=rdflib.Literal(label, lang='en'))
        if term == None:
            term = go.value(predicate=RDFS.label,
                            object=rdflib.Literal(label))
        return term

    with open(data_dir + 'sgd-data-slim.pkl', 'rb') as fi:
        sgd_data = pickle.load(fi)

    # %%
    # regulation, go relations, mutants, happens during
    remap_labels = {'regulation of cell aging': 'regulation of cellular senescence', 'regulation of response to DNA damage stimulus': 'cellular response to DNA damage stimulus', 'DNA damage response': 'cellular response to DNA damage stimulus', 'acts upstream of negative effect': 'acts upstream of, negative effect', 'acts upstream of positive effect': 'acts upstream of, positive effect', 'acts upstream of or within positive effect': 'acts upstream of or within, positive effect', 'null': 'null mutant', 'amino acid catabolic process': 'cellular amino acid catabolic process'}
    remap_terms = {'NTR:1': 'CHEBI:82641', 'NTR:23602': 'CHEBI:14321', 'NTR:19552': 'CHEBI:156524', 'NTR:3117': 'CHEBI:52643', 'NTR:21951': 'CHEBI:156510', 'NTR:17300': 'CHEBI:156516', 'NTR:19385': 'CHEBI:83634', 'NTR:19547': 'CHEBI:156525', 'NTR:21956': 'CHEBI:156509', 'NTR:11022': None, 'NTR:23707': None}
    regulation_map = {'RNA activity': 'regulation of RNA activity', 'RNA stability': 'regulation of RNA stability', 'protein activity': 'regulation of protein activity', 'protein stability': 'regulation of protein stability', 'transcription': 'regulation of transcription', 'translation': 'regulation of translation'}

    # %%
    pheno_dict = {}
    full_list = []
    for orf in sgd_data:
        for pheno in sgd_data[orf]['phenotype_details']:
            key = term_from_label(pheno[0]).split('/')[-1]
            # if key == 'APO_0000271':
            #     key = 'GO_0009451'
            if pheno[1] != None:
                key = key + '--' + term_from_label(pheno[1]).split('/')[-1]
            chem = remap_terms[pheno[4]] if pheno[4] in remap_terms else pheno[4]
            if chem != None:
                key = key + '-' + chem.replace(':', '_')
            full_list.append(key)
            if key in pheno_dict:
                pheno_dict[key].append(orf)
            else:
                pheno_dict[key] = [orf]

    interactions = {}
    for interact in set([(a[1], a[2], a[3], a[5], a[6]) for orf in sgd_data
                        for a in sgd_data[orf]['interaction_details']]):
        key = [interact[0], interact[1]]
        key.sort()
        try:
            key = '--'.join(key) + '--' + interact[2]
        except TypeError:
            print(interact)
            raise KeyError
        if interact[3]:
            try:
                interactions[key].add((interact[3], interact[4]))
            except KeyError:
                interactions[key] = set([(interact[3], interact[4])])
        elif not key in interactions:
                interactions[key] = set()

    reguls = {}
    for reg in set([a for orf in sgd_data
                    for a in sgd_data[orf]['regulation_details']]):
        key = f'{reg[1]}--{reg[2]}--{reg[0]}'
        if reg[4]:
            key = key + '--' + reg[4]
        if reg[3]:
            try:
                reguls[key].add(reg[3])
            except KeyError:
                reguls[key] = set([reg[3]])
        elif key not in reguls:
            reguls[key] = set()


    # %%
    kg = rdflib.Graph()
    kg.bind('sgd', SGD)
    kg.bind('obo', OBO)
    kg.bind('kg', KG)
    kg.bind('oboInOwl', OBOOWL)
    NOT_go = []
    orfs = list(sgd_data.keys())
    for orf in sgd_data:
        locus_type = SGD.locus if sgd_data[orf]['locus'] == '' \
            else SGD[sgd_data[orf]['locus'].replace(' ', '_')]
        kg.add((KG[orf], RDFS.subClassOf, locus_type))
        kg.add((KG[orf], OBOOWL.id, rdflib.Literal(sgd_data[orf]['id'])))
        if SKIP_GO:
            continue
            # probably just remove this once I have fixed a GO graph
        for go in sgd_data[orf]['go_details']:
            print('WTFWTFWTFWTFWTF')
            if go[0] == 'NOT':
                NOT_go.append((KG[orf], 'NOT', go[1]))
            else:
                rel = remap_labels[go[0]] if go[0] in remap_labels else go[0]
                trips = role_between_classes(KG[orf], OBO[go[1].replace(':', '_')],
                                            term_from_label_go(rel))
                for t in trips:
                    kg.add(t)

    print(len(kg))
# %%
    parents = {'APO_0000017': 'hasChemObservable',
            'APO_0000023': 'hasChemDevelopment',
            'APO_0000217': 'hasChemEssentiality',
            'APO_0000216': 'hasChemFitness',
            'APO_0000094': 'hasChemMetAndGrowth',
            'APO_0000049': 'hasChemMorphology',
            'APO_0000066': 'hasChemCellularProc'}

    children = {'APO_0000024': 'hasChemBudding',
                'APO_0000027': 'hasChemFilamentousGrowth',
                'APO_0000030': 'hasChemLifespan',
                'APO_0000031': 'hasChemSexualCycle',
                'APO_0000112': 'hasChemInviable',
                'APO_0000113': 'hasChemViable',
                'APO_0000110': 'hasChemCompetitiveGrowth',
                'APO_0000152': 'hasChemHaploinsufficient',
                'APO_0000111': 'hasChemViability',
                'APO_0000095': 'hasChemChemCompAcc',
                'APO_0000222': 'hasChemChemCompExc',
                'APO_0000096': 'hasChemNutrientUtilization',
                'APO_0000022': 'hasChemProteinActivity',
                'APO_0000149': 'hasChemProteinAcc',
                'APO_0000209': 'hasChemProteinDistribution',
                'APO_0000131': 'hasChemProteinMod',
                'APO_0000218': 'hasChemRedoxState',
                'APO_0000224': 'hasChemRNAacc',
                'APO_0000106': 'hasChemVegGrowth',
                'APO_0000050': 'hasChemCellMorph',
                'APO_0000158': 'hasChemCultureApp',
                'APO_0000312': 'hasChemCellDeath',
                'APO_0000143': 'hasChemChromMaint',
                'APO_0000073': 'hasChemIntracellTransp',
                'APO_0000072': 'hasChemMitoticCellCycle',
                'APO_0000274': 'hasChemPrionState',
                'APO_0000080': 'hasChemStressResistance',
                'GO_0009451': 'hasChemRNAmod'}

    for pheno in pheno_dict:
        ps = pheno.replace('--', '-')
        ps = ps.split('-')
        if len(ps) == 1:
            phenotype = OBO[pheno]
        else:
            phenotype = KG['-'.join(ps)]
            trips = sub_of_intersect(phenotype, [OBO[p] for p in ps
                                                 if 'CHEBI' not in p])
            for t in trips:
                kg.add(t)
        chem_rel = None
        if 'CHEBI' in pheno:
            obs = ps[0]
            chem = ps[-1]

            trips = role_between_classes(phenotype, OBO[chem], SGD.aboutChemical)
            for t in trips:
                kg.add(t)

            if obs in parents:
                chem_rel = (parents[obs], chem)
                continue
            if obs in children:
                chem_rel = (children[obs], chem)
                continue

            for o in children:
                q = f"""PREFIX obo: <http://purl.obolibrary.org/obo/>
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        SELECT (count(?mid) as ?distance) WHERE {{
                            obo:{obs} rdfs:subClassOf* ?mid .
                            ?mid rdfs:subClassOf+ obo:{o} .
                        }}"""
                if int(list(ont.query(q))[0][0]) > 0:
                    chem_rel = (children[o], chem)
                    break

        for orf in pheno_dict[pheno]:
            trips = role_between_classes(KG[orf], phenotype, OBO.RO_0002200)
            if chem_rel != None:
                trips.extend(role_between_classes(KG[orf], OBO[chem_rel[1]], SGD[chem_rel[0]]))
            for t in trips:
                kg.add(t)
    print(len(kg))

    # %%
    for inter in interactions:
        isplit = inter.split('--')
        interaction = SGD.genetically_interacts_with if isplit[2] == 'Genetic' \
            else SGD.physically_interacts_with
        trips = role_between_classes(KG[isplit[0]], KG[isplit[1]], interaction)
        # to make the relation symmetric - not necessary if method supports symmetric relations:
        # trips.extend(role_between_classes(KG[isplit[1]], KG[isplit[0]], interaction))

        if len(interactions[inter]) > 0:
            itype = OBO.MI_0208 if isplit[2] == 'Genetic' else OBO.INO_0000311
            trips.append((KG[inter], RDFS.subClassOf, itype))
            irel = SGD.has_genetic_interaction if isplit[2] == 'Genetic' \
                else SGD.has_physical_interaction
            trips.extend(role_between_classes(KG[isplit[0]], KG[inter], irel))
            trips.extend(role_between_classes(KG[isplit[1]], KG[inter], irel))

            for pheno in interactions[inter]:
                if pheno[1] != None:
                    terms = term_from_label(pheno[0]).split('/')[-1] + '-' + \
                        term_from_label(pheno[1]).split('/')[-1]

                    phenotype = KG[terms]
                    if terms not in pheno_dict:
                        trips.extend(sub_of_intersect(phenotype,
                                                    [OBO[p] for p in
                                                    terms.split('-')]))
                else:
                    phenotype = term_from_label(pheno[0])
                trips.extend(role_between_classes(KG[inter], phenotype,
                                                SGD.resultsInPhenotype))
            
        for t in trips:
            kg.add(t)

    print(len(kg))

    # %%
    full_ont = rdflib.Graph()
    full_ont += go
    full_ont += ont
    def term_from_label_full(label):
        term = full_ont.value(predicate=RDFS.label,
                         object=rdflib.Literal(label, datatype=rdflib.URIRef('http://www.w3.org/2001/XMLSchema#string')))
        if term == None:
            term = full_ont.value(predicate=RDFS.label,
                            object=rdflib.Literal(label, lang='en'))
        if term == None:
            term = full_ont.value(predicate=RDFS.label,
                            object=rdflib.Literal(label))
        return term

    for reg in reguls:
        rsplit = reg.split('--')
        if len(rsplit) == 4:
            rel_label = 'negatively ' if rsplit[3] == 'negative' else 'positively '
        else:
            rel_label = ''
        rel_label = rel_label + 'regulating ' + rsplit[2]
        rel = SGD[rel_label.replace(' ', '_')]
        
        trips = role_between_classes(KG[rsplit[0]], KG[rsplit[1]], rel)

        if len(reguls[reg]) > 0:
            reg_label = 'regulation of gene transcription' if \
                rsplit[2] == 'transcription' and len(rsplit) == 4 else \
                    f'regulation of {rsplit[2]}'
            reg_rel = 'regulator of'
            if len(rsplit) == 4:
                reg_label = f'{rsplit[3]} {reg_label}'
                reg_rel = f'{rsplit[3]} {reg_rel}'
            reg_node = term_from_label_full(reg_label)

            reg_id = reg.replace(' ', '_')
            reg_id = reg_id.replace('--', '-')
            trips.append((KG[reg_id], RDFS.subClassOf, reg_node))
            rel = term_from_label_full(reg_rel)
            trips.extend(role_between_classes(KG[rsplit[0]], KG[reg_id], rel))
            trips.extend(role_between_classes(KG[reg_id], KG[rsplit[1]],
                                            SGD.regulating_gene))
            for ev in reguls[reg]:
                event = remap_labels[ev] if ev in remap_labels else ev

                trips.extend(role_between_classes(KG[reg_id],
                                                term_from_label_go(event),
                                                OBO.RO_0002092)
                                                )

        for t in trips:
            kg.add(t)

    print(len(kg))

    # %%
    kg.serialize(destination=graph_dir + "kg-nf.ttl")
    print('kg-nf.ttl saved')
    return kg
# %%

if __name__ =='__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sgd = rdflib.Graph()
    sgd.parse(os.path.join(BASE, 'graphs/sgd_kg_fix.ttl'))
    go = rdflib.Graph()
    go.parse(os.path.join(BASE, 'graphs/go-ext.ttl'))

    generate_gene_kg(BASE, go, sgd)