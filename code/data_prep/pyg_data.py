# %%
import os, pickle, sys
import torch as th
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embeddings')))

from embeddings.boxGumbel_gci0 import BoxGumbelModel
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(BASE, 'datasets/split_datasets/split_gci2.pkl'), 'rb') as fi:
    dataset = pickle.load(fi)

# %%
# set which dataset you want to create
INTERACT_DATA = 'interactions_DMA30'
# zZ, c, box
FEATURE_REPRESENTATION = 'box'

# %%
with open(os.path.join(BASE, f'datasets/split_datasets/{INTERACT_DATA}.pkl'),
          'rb') as fi:
    interactions = pickle.load(fi)

# %%
# model_dir = os.path.join(BASE, 'large_files')
model_dir = os.path.join(BASE, 'trained_models')
box_dirs = {
    'mat_ent': '20250617-141221-mat_ent',
    'genes': '20250617-141028-genes',
    # 'root': None,
    'quality': '20250617-142123-quality',
    'reactions': '20250617-142351-reactions',
    'cell_comp': '20250617-141851-cell_comp',
    'reguls': '20250617-142314-reguls',
    'mol_func': '20250617-141725-mol_func--20250617-144156',
    'bio_proc': '20250617-141633-bio_proc--20250617-151352'}

box_models = {}

for k,v in box_dirs.items():
    box_models[k] = BoxGumbelModel(None, from_file=os.path.join(model_dir, v), init_for_train=False)

# %%
data = HeteroData()
for k,v in box_models.items():
    data[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    match FEATURE_REPRESENTATION:
        case 'zZ':
            data[k].x = th.cat([v.module.class_boxes.all_boxes.z,
                                v.module.class_boxes.all_boxes.Z], dim=-1)
        case 'c':
            data[k].x = th.cat([v.module.class_boxes.all_boxes.centre,
                                (v.module.class_boxes.all_boxes.Z -
                                 v.module.class_boxes.all_boxes.z).abs()],
                                 dim=-1)
        case 'box':
            boxes = v.module.class_boxes.all_boxes.data
            data[k].x = boxes.reshape((len(boxes), -1))
        case _:
            raise NotImplementedError('Only implemented for zZ, c, and box. '
                                      f'Not for {FEATURE_REPRESENTATION}')
        
# %%
regulating = []
pos = []
neg = []
reg = []
for r,v in dataset['gci2'].items():
    r = r.split('/')[-1].split('#')[-1]
    for nodes, rels in v.items():
        if 'root' in nodes:
            continue
        if nodes[0] == 'genes' and nodes[1] == 'genes' and 'regul' in r:
            # print((nodes, r))
            regulating.extend(rels)
            if 'pos' in r:
                pos.extend(rels)
            elif 'neg' in r:
                neg.extend(rels)
            else:
                reg.extend(rels)
        data[nodes[0], r, nodes[1]].edge_index = th.tensor(rels,
                                                           dtype=th.int64).T
        
data['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating,
                                                            dtype=th.int64).T

data['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos,
                                                            dtype=th.int64).T

data['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg,
                                                            dtype=th.int64).T

data['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg,
                                                            dtype=th.int64).T

data['genes', 'interacts', 'genes'].edge_index = interactions[:,:2].T.long()
data['genes', 'interacts', 'genes'].edge_label = interactions[:, 3]

data = T.ToUndirected(merge=False)(data)
del data['genes', 'rev_interacts', 'genes']

# %%
with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_'
                       f'{FEATURE_REPRESENTATION}_{INTERACT_DATA}.pkl'),
                       'wb') as fo:
    pickle.dump(data, fo)

# %%
