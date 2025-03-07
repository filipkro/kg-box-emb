# %%
import pickle, os
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(BASE, 'data/')
print(data_dir)
# %%
with open(data_dir + 'sgd-data-ext.pkl', 'rb') as fi:
    data = pickle.load(fi)

names = dict([(data[key]['locus']['display_name'], key) for key in data])
names['YMR31'] = 'YFR049W'

print(len(names))
print(len(data))
# %%
entries = ['go_details', 'phenotype_details', 'interaction_details',
           'regulation_details', 'locus', 'id']

chem_synonyms = {'fijianolide B': 'CHEBI:69134'}

# %%
slim_data = {}
for orf in data:
    slim_data[orf] = {}

    gog = [((a['qualifier'], a['go']['go_id']), a['annotation_type'])
           for a in data[orf][entries[0]]]
    godict = {}
    for g in gog:
        if g[0] not in godict or g[1] == 'manually curated':
            godict[g[0]] = g[1]
        elif g[0] in godict and godict[g[0]] == 'computational':
            godict[g[0]] = g[1]
        # high-throughput
    slim_data[orf][entries[0]] = list(godict.items())

    terms = []
    for a in data[orf][entries[1]]:
        pheno = a['phenotype']['display_name'].split(': ')
        try:
            quantifier = pheno[1] if len(pheno) > 1 else None
        except IndexError:
            print(pheno)
            raise IndexError
        chem = None
        for b in a['properties']:
            if b['class_type'] == 'CHEMICAL':
                try:
                    chem = b['bioitem']['link'].split('/')[-1] if \
                        b['bioitem']['link'] is not None else \
                            chem_synonyms[b['bioitem']['display_name']]
                except KeyError:
                    print(b)
                    raise KeyError
        terms.append((pheno[0], quantifier, a['strain']['display_name'],
                      a['mutant_type'], chem))
    
    slim_data[orf][entries[1]] = list(set(terms))

    terms = []
    for a in data[orf][entries[2]]:
        if a['locus1']['display_name'] in names and \
                a['locus2']['display_name'] in names:
            pheno = a['phenotype']
            quantifier = None
            if pheno is not None:
                pheno = pheno['display_name'].split(': ')
                quantifier = pheno[1] if len(pheno) > 1 else None
                pheno = pheno[0]
            terms.append((a['bait_hit'], names[a['locus1']['display_name']],
                          names[a['locus2']['display_name']],
                          a['interaction_type'], a['mutant_type'],
                          pheno, quantifier))
    slim_data[orf][entries[2]] = list(set(terms))

    terms = set([(a['regulation_of'], names[a['locus1']['display_name']],
                  names[a['locus2']['display_name']], a['happens_during'],
                  a['direction']) for a in data[orf]['regulation_details']
                  if a['locus1']['display_name'] in names and
                  a['locus2']['display_name'] in names])
    
    slim_data[orf][entries[3]] = list(terms)

    slim_data[orf][entries[4]] = data[orf][entries[4]]['locus_type']
    slim_data[orf][entries[5]] = data[orf][entries[4]]['sgdid']

print(len(slim_data))

# %%
with open(data_dir + 'sgd-data-slim.pkl', 'wb') as fo:
    pickle.dump(slim_data, fo)
# %%

