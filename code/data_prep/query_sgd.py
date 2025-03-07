# %%
import os, pickle
import requests
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE, 'data/')
BASE_URL = 'https://www.yeastgenome.org/backend/locus/'
end_points = ['go_details', 'phenotype_details', 'interaction_details',
              'regulation_details']

with open(data_dir + 'orf-names-ext.pkl', 'rb') as fi:
    all_orfs = pickle.load(fi)

try:
    data = pickle.load(open(data_dir + 'sgd-data-ext.pkl', 'rb'))
except FileNotFoundError:
    data = {}

# %%
for c, orf in enumerate(all_orfs):
    orf_data = {}
    if ' ' in orf or orf in data:
        print(f"\nSkipping {orf}...")
        continue

    # if orf not in data.keys():
    url = BASE_URL + orf
    r = requests.get(url)
    if r.ok:
        locus_data = r.json()
        orf = locus_data['format_name']
        data[orf] = {'locus': locus_data}
        for ep in end_points:
            url = BASE_URL + orf + '/' + ep
            r = requests.get(url)
            if r.ok:
                orf_data[ep] = r.json()

    data[orf] = orf_data
    
    if c > 0 and c % 1000 == 0:
        with open(data_dir + 'sgd-data-ext.pkl', 'wb') as fo:
            pickle.dump(data, fo)

    print(f'{c+1} of {len(all_orfs)} done...', end='\r')

with open(data_dir + 'sgd-data-ext.pkl', 'wb') as fo:
    pickle.dump(data, fo)

print()
print('done')
# %%
