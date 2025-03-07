# %%
import os, pickle
from utils import dataset_utils
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
role_fp = os.path.join(BASE, 'graphs/role-graph.ttl')
full_fp = os.path.join(BASE, 'graphs/all_graphs-no-int.ttl')
split_dir = os.path.join(BASE, 'datasets/split_datasets')
LOAD_NORMALIZED_DATA = False
# %%
if LOAD_NORMALIZED_DATA:
    print('Loading normalized datasets')
    with open(os.path.join(BASE, 'datasets/normalized_base_all_graph.pkl'), 'rb') as fi:
        data = pickle.load(fi)
        full_data = data['data']
        full_index = data['index']
else:
    full_data, full_index = dataset_utils.get_normalized_dataset(full_fp,
                                                                role_fp=role_fp)

    print('saving normalized and reindexed data')
    with open(os.path.join(BASE, 'datasets/normalized_base_all_graph.pkl'), 'wb') as fo:
        pickle.dump({'data': full_data, 'index': full_index}, fo)
# %%
# load the dicts for split datasets
id_dicts = {}
for f in os.listdir(split_dir):
    if f not in ['mat_ent.pkl', 'cell_comp.pkl', 'quality.pkl', 'root.pkl', 'genes.pkl', 'reactions.pkl', 'reguls.pkl', 'mol_func.pkl', 'bio_proc.pkl']:
        continue
    key = f.split('.')[0]
    print(f)
    print(key)
    with open(os.path.join(split_dir, f), 'rb') as fi:
        id_dicts[key] = pickle.load(fi)['index']['class_index']

# %%
orig_data = full_data['gci2']
rev_rel_dict = {v:k for k,v in full_index['property_index'].items()}
rev_class_dict = {v:k for k,v in full_index['class_index'].items()}
data = {}
dom_range = {}
for t in orig_data:
    dom = None
    ran = None
    rel = rev_rel_dict[t[1].item()]
    A = rev_class_dict[t[0].item()]
    B = rev_class_dict[t[2].item()]
    for k,v in id_dicts.items():
        if not dom and A in v:
            dom = k
        if not ran and B in v:
            ran = k
    assert dom and ran
    if rel in data:
        if (dom, ran) in data[rel]:
            data[rel][(dom, ran)].append([id_dicts[dom][A],
                                          id_dicts[ran][B]])
        else:
            data[rel][(dom, ran)] = [[id_dicts[dom][A],
                                          id_dicts[ran][B]]]
    else:
        data[rel] = {(dom, ran): [[id_dicts[dom][A],
                                          id_dicts[ran][B]]]}


# %%
orig_data_sym = full_data['gci2_sym']
rev_sym_rel_dict = {v:k for k,v in full_index['sym_property_index'].items()}
sym_data = {}
dom_range = {}

for t in orig_data_sym:
    dom = None
    ran = None
    rel = rev_sym_rel_dict[t[1].item()]
    A = rev_class_dict[t[0].item()]
    B = rev_class_dict[t[2].item()]
    for k,v in id_dicts.items():
        if not dom and A in v:
            dom = k
        if not ran and B in v:
            ran = k
    assert dom and ran
    if rel in sym_data:
        if (dom, ran) in sym_data[rel]:
            sym_data[rel][(dom, ran)].append([id_dicts[dom][A],
                                          id_dicts[ran][B]])
        else:
            sym_data[rel][(dom, ran)] = [[id_dicts[dom][A],
                                          id_dicts[ran][B]]]
    else:
        sym_data[rel] = {(dom, ran): [[id_dicts[dom][A],
                                          id_dicts[ran][B]]]}
        
# %%
with open(os.path.join(split_dir, 'split_gci2.pkl'), 'wb') as fo:
    pickle.dump({'gci2': data, 'gci2_sym': sym_data}, fo)
