# %%
import os, pickle, sys
import torch as th
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embeddings.boxGumbel_gci0 import BoxGumbelModel
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
model_dir = os.path.join(BASE, 'large_files')
box_dirs = {
    'mat_ent': '20250110-192405-mat_ent--20250110-221512--20250110-232045',
    'genes': '20250110-192529-genes--20250110-194714--20250110-200406--20250110-201617--20250110-202609--20250110-203226',
    'root': '20250110-192626-root--20250110-193340',
    'quality': '20250110-192713-quality--20250110-200051--20250110-203129--20250110-204549--20250110-205647',
    'reactions': '20250110-192751-reactions--20250110-194606--20250110-195357--20250110-200211',
    'cell_comp': '20250110-192823-cell_comp--20250110-195704--20250110-201413--20250110-202426',
    'reguls': '20250110-192928-reguls--20250110-201108--20250110-203357',
    'mol_func': '20250110-192957-mol_func--20250110-195819--20250110-201814--20250110-203037',
    'bio_proc': '20250110-193049-bio_proc--20250110-212952--20250110-221236'}

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
