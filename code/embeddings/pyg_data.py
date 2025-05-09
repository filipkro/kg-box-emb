# %%
import os, pickle
import torch as th
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from boxGumbel_gci0 import BoxGumbelModel
import torch.nn.functional as F
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(BASE, 'datasets/split_datasets/split_gci2.pkl'), 'rb') as fi:
    dataset = pickle.load(fi)
# %%
with open(os.path.join(BASE, 'datasets/split_datasets/interactions.pkl'), 'rb') as fi:
    interactions = pickle.load(fi)
with open(os.path.join(BASE, 'datasets/split_datasets/refined_interactions.pkl'), 'rb') as fi:
    refined_interactions = pickle.load(fi)
with open(os.path.join(BASE, 'datasets/split_datasets/interactions_DMA30.pkl'), 'rb') as fi:
    interactions_DMA30 = pickle.load(fi)
with open(os.path.join(BASE, 'datasets/split_datasets/refined_interactions_DMA30.pkl'), 'rb') as fi:
    refined_interactions_DMA30 = pickle.load(fi)

# %%
model_dir = os.path.join(BASE, 'trained_models')
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
    box_models[k] = BoxGumbelModel(None, from_file=os.path.join(model_dir, v))


# %%
data = HeteroData()
data_c = HeteroData()
data_one_hot = HeteroData()
data_no_x = HeteroData()

data_DMA30_c = HeteroData()

data_refined = HeteroData()
data_c_refined = HeteroData()
data_one_hot_refined = HeteroData()
data_no_x_refined = HeteroData()

data_refined_DMA30_c = HeteroData()
for k,v in box_models.items():
    data[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    data[k].x = th.cat([v.module.class_boxes.all_boxes.z,
                        v.module.class_boxes.all_boxes.Z], dim=-1)
    data_c[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    data_c[k].x = th.cat([v.module.class_boxes.all_boxes.centre,
                          (v.module.class_boxes.all_boxes.Z -
                           v.module.class_boxes.all_boxes.z).abs()], dim=-1)
    
    data_one_hot[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    data_no_x[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    if k in ['genes', 'root']:
        # one hot for genes and root??
        data_one_hot[k].x = th.eye(v.module.nbr_classes, dtype=th.float32)
    else:
        data_one_hot[k].x = th.cat([v.module.class_boxes.all_boxes.z,
                        v.module.class_boxes.all_boxes.Z], dim=-1)
        data_no_x[k].x = th.cat([v.module.class_boxes.all_boxes.z,
                        v.module.class_boxes.all_boxes.Z], dim=-1)
        

    data_refined[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    data_refined[k].x = th.cat([v.module.class_boxes.all_boxes.z,
                        v.module.class_boxes.all_boxes.Z], dim=-1)
    data_c_refined[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    data_c_refined[k].x = th.cat([v.module.class_boxes.all_boxes.centre,
                          (v.module.class_boxes.all_boxes.Z -
                           v.module.class_boxes.all_boxes.z).abs()], dim=-1)
    
    data_one_hot_refined[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    data_no_x_refined[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    if k in ['genes', 'root']:
        # one hot for genes and root??
        data_one_hot_refined[k].x = th.eye(v.module.nbr_classes, dtype=th.float32)
    else:
        data_one_hot_refined[k].x = th.cat([v.module.class_boxes.all_boxes.z,
                        v.module.class_boxes.all_boxes.Z], dim=-1)
        data_no_x_refined[k].x = th.cat([v.module.class_boxes.all_boxes.z,
                        v.module.class_boxes.all_boxes.Z], dim=-1)
        
    data_DMA30_c[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    data_DMA30_c[k].x = th.cat([v.module.class_boxes.all_boxes.centre,
                          (v.module.class_boxes.all_boxes.Z -
                           v.module.class_boxes.all_boxes.z).abs()], dim=-1)
    
    data_refined_DMA30_c[k].node_id = th.arange(v.module.nbr_classes, dtype=th.int64)
    data_refined_DMA30_c[k].x = th.cat([v.module.class_boxes.all_boxes.centre,
                          (v.module.class_boxes.all_boxes.Z -
                           v.module.class_boxes.all_boxes.z).abs()], dim=-1)


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
        data[nodes[0], r, nodes[1]].edge_index = th.tensor(rels, dtype=th.int64).T
        data_c[nodes[0], r, nodes[1]].edge_index = th.tensor(rels, dtype=th.int64).T
        data_one_hot[nodes[0], r, nodes[1]].edge_index = th.tensor(rels, dtype=th.int64).T
        data_no_x[nodes[0], r, nodes[1]].edge_index = th.tensor(rels, dtype=th.int64).T

        data_refined[nodes[0], r, nodes[1]].edge_index = th.tensor(rels, dtype=th.int64).T
        data_c_refined[nodes[0], r, nodes[1]].edge_index = th.tensor(rels, dtype=th.int64).T
        data_one_hot_refined[nodes[0], r, nodes[1]].edge_index = th.tensor(rels, dtype=th.int64).T
        data_no_x_refined[nodes[0], r, nodes[1]].edge_index = th.tensor(rels, dtype=th.int64).T

        data_DMA30_c[nodes[0], r, nodes[1]].edge_index = th.tensor(rels, dtype=th.int64).T
        data_refined_DMA30_c[nodes[0], r, nodes[1]].edge_index = th.tensor(rels, dtype=th.int64).T
data['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating, dtype=th.int64).T
data_c['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating, dtype=th.int64).T
data_one_hot['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating, dtype=th.int64).T
data_no_x['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating, dtype=th.int64).T

data_refined['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating, dtype=th.int64).T
data_c_refined['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating, dtype=th.int64).T
data_one_hot_refined['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating, dtype=th.int64).T
data_no_x_refined['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating, dtype=th.int64).T

data_DMA30_c['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating, dtype=th.int64).T
data_refined_DMA30_c['genes', 'regulating', 'genes'].edge_index = th.tensor(regulating, dtype=th.int64).T

# %%
data['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos, dtype=th.int64).T
data_c['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos, dtype=th.int64).T
data_one_hot['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos, dtype=th.int64).T
data_no_x['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos, dtype=th.int64).T

data['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg, dtype=th.int64).T
data_c['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg, dtype=th.int64).T
data_one_hot['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg, dtype=th.int64).T
data_no_x['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg, dtype=th.int64).T

data['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg, dtype=th.int64).T
data_c['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg, dtype=th.int64).T
data_one_hot['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg, dtype=th.int64).T
data_no_x['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg, dtype=th.int64).T


data_refined['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos, dtype=th.int64).T
data_c_refined['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos, dtype=th.int64).T
data_one_hot_refined['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos, dtype=th.int64).T
data_no_x_refined['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos, dtype=th.int64).T

data_refined['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg, dtype=th.int64).T
data_c_refined['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg, dtype=th.int64).T
data_one_hot_refined['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg, dtype=th.int64).T
data_no_x_refined['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg, dtype=th.int64).T

data_refined['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg, dtype=th.int64).T
data_c_refined['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg, dtype=th.int64).T
data_one_hot_refined['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg, dtype=th.int64).T
data_no_x_refined['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg, dtype=th.int64).T


data_DMA30_c['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos, dtype=th.int64).T
data_refined_DMA30_c['genes', 'pos_regulating', 'genes'].edge_index = th.tensor(pos, dtype=th.int64).T

data_DMA30_c['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg, dtype=th.int64).T
data_refined_DMA30_c['genes', 'neg_regulating', 'genes'].edge_index = th.tensor(neg, dtype=th.int64).T

data_DMA30_c['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg, dtype=th.int64).T
data_refined_DMA30_c['genes', 'unspec_regulating', 'genes'].edge_index = th.tensor(reg, dtype=th.int64).T
# %%
data['genes', 'interacts', 'genes'].edge_index = interactions[:,:2].T.long()
data['genes', 'interacts', 'genes'].edge_label = interactions[:,2]
data_c['genes', 'interacts', 'genes'].edge_index = interactions[:,:2].T.long()
data_c['genes', 'interacts', 'genes'].edge_label = interactions[:,2]
data_one_hot['genes', 'interacts', 'genes'].edge_index = interactions[:,:2].T.long()
data_one_hot['genes', 'interacts', 'genes'].edge_label = interactions[:,2]
data_no_x['genes', 'interacts', 'genes'].edge_index = interactions[:,:2].T.long()
data_no_x['genes', 'interacts', 'genes'].edge_label = interactions[:,2]

data_refined['genes', 'interacts', 'genes'].edge_index = refined_interactions[:,:2].T.long()
data_refined['genes', 'interacts', 'genes'].edge_label = refined_interactions[:,2]
data_c_refined['genes', 'interacts', 'genes'].edge_index = refined_interactions[:,:2].T.long()
data_c_refined['genes', 'interacts', 'genes'].edge_label = refined_interactions[:,2]
data_one_hot_refined['genes', 'interacts', 'genes'].edge_index = refined_interactions[:,:2].T.long()
data_one_hot_refined['genes', 'interacts', 'genes'].edge_label = refined_interactions[:,2]
data_no_x_refined['genes', 'interacts', 'genes'].edge_index = refined_interactions[:,:2].T.long()
data_no_x_refined['genes', 'interacts', 'genes'].edge_label = refined_interactions[:,2]

data_DMA30_c['genes', 'interacts', 'genes'].edge_index = interactions_DMA30[:,:2].T.long()
data_DMA30_c['genes', 'interacts', 'genes'].edge_label = interactions_DMA30[:,2]
data_refined_DMA30_c['genes', 'interacts', 'genes'].edge_index = refined_interactions_DMA30[:,:2].T.long()
data_refined_DMA30_c['genes', 'interacts', 'genes'].edge_label = refined_interactions_DMA30[:,2]
# %%
data = T.ToUndirected(merge=False)(data)
del data['genes', 'rev_interacts', 'genes']

data_c = T.ToUndirected(merge=False)(data_c)
del data_c['genes', 'rev_interacts', 'genes']

data_one_hot = T.ToUndirected(merge=False)(data_one_hot)
del data_one_hot['genes', 'rev_interacts', 'genes']

data_no_x = T.ToUndirected(merge=False)(data_no_x)
del data_no_x['genes', 'rev_interacts', 'genes']


data_refined = T.ToUndirected(merge=False)(data_refined)
del data_refined['genes', 'rev_interacts', 'genes']

data_c_refined = T.ToUndirected(merge=False)(data_c_refined)
del data_c_refined['genes', 'rev_interacts', 'genes']

data_one_hot_refined = T.ToUndirected(merge=False)(data_one_hot_refined)
del data_one_hot_refined['genes', 'rev_interacts', 'genes']

data_no_x_refined = T.ToUndirected(merge=False)(data_no_x_refined)
del data_no_x_refined['genes', 'rev_interacts', 'genes']

data_DMA30_c = T.ToUndirected(merge=False)(data_DMA30_c)
del data_DMA30_c['genes', 'rev_interacts', 'genes']
data_refined_DMA30_c = T.ToUndirected(merge=False)(data_refined_DMA30_c)
del data_refined_DMA30_c['genes', 'rev_interacts', 'genes']
# %%
with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph.pkl'), 'wb') as fo:
    pickle.dump(data, fo)

with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_c.pkl'), 'wb') as fo:
    pickle.dump(data_c, fo)

with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_one_hot.pkl'), 'wb') as fo:
    pickle.dump(data_one_hot, fo)

with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_no_x.pkl'), 'wb') as fo:
    pickle.dump(data_no_x, fo)


with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_refined.pkl'), 'wb') as fo:
    pickle.dump(data_refined, fo)

with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_c_refined.pkl'), 'wb') as fo:
    pickle.dump(data_c_refined, fo)

with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_one_hot_refined.pkl'), 'wb') as fo:
    pickle.dump(data_one_hot_refined, fo)

with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_no_x_refined.pkl'), 'wb') as fo:
    pickle.dump(data_no_x_refined, fo)

# %%
data_refined_class = data_refined.clone()

cut_off = 0.05
data_refined_class['genes', 'interacts', 'genes'].edge_label[data_refined_class['genes', 'interacts', 'genes'].edge_label <= -cut_off] = -1 * th.ones((data_refined_class['genes', 'interacts', 'genes'].edge_label <= -cut_off).sum())

data_refined_class['genes', 'interacts', 'genes'].edge_label[data_refined_class['genes', 'interacts', 'genes'].edge_label >= cut_off] = th.ones((data_refined_class['genes', 'interacts', 'genes'].edge_label >= cut_off).sum())

data_refined_class['genes', 'interacts', 'genes'].edge_label[(data_refined_class['genes', 'interacts', 'genes'].edge_label > -cut_off) & (data_refined_class['genes', 'interacts', 'genes'].edge_label < cut_off)] = th.zeros(((data_refined_class['genes', 'interacts', 'genes'].edge_label > -cut_off) & (data_refined_class['genes', 'interacts', 'genes'].edge_label < cut_off)).sum())

# %%
data_refined_binary = data_refined.clone()

cut_off = 0.05
data_refined_binary['genes', 'interacts', 'genes'].edge_label[data_refined_binary['genes', 'interacts', 'genes'].edge_label.abs() <= cut_off] = th.zeros((data_refined_binary['genes', 'interacts', 'genes'].edge_label.abs() <= cut_off).sum())

data_refined_binary['genes', 'interacts', 'genes'].edge_label[data_refined_binary['genes', 'interacts', 'genes'].edge_label.abs() > cut_off] = th.ones((data_refined_binary['genes', 'interacts', 'genes'].edge_label.abs() > cut_off).sum())

# %%
data_class = data.clone()

cut_off = 0.05
data_class['genes', 'interacts', 'genes'].edge_label[data_class['genes', 'interacts', 'genes'].edge_label <= -cut_off] = -1 * th.ones((data_class['genes', 'interacts', 'genes'].edge_label <= -cut_off).sum())

data_class['genes', 'interacts', 'genes'].edge_label[data_class['genes', 'interacts', 'genes'].edge_label >= cut_off] = th.ones((data_class['genes', 'interacts', 'genes'].edge_label >= cut_off).sum())

data_class['genes', 'interacts', 'genes'].edge_label[(data_class['genes', 'interacts', 'genes'].edge_label > -cut_off) & (data_class['genes', 'interacts', 'genes'].edge_label < cut_off)] = th.zeros(((data_class['genes', 'interacts', 'genes'].edge_label > -cut_off) & (data_class['genes', 'interacts', 'genes'].edge_label < cut_off)).sum())

# %%
data_refined_class['genes', 'interacts', 'genes'].edge_label = F.one_hot(data_refined_class['genes', 'interacts', 'genes'].edge_label.long() + 1).float()

data_class['genes', 'interacts', 'genes'].edge_label = F.one_hot(data_class['genes', 'interacts', 'genes'].edge_label.long() + 1).float()
# %%
# with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_class_refined.pkl'), 'wb') as fo:
#     pickle.dump(data_refined_class, fo)

# with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_binary_refined.pkl'), 'wb') as fo:
#     pickle.dump(data_refined_binary, fo)

# with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_class.pkl'), 'wb') as fo:
#     pickle.dump(data_class, fo)
# %%
data_fitness = data.clone()
data_refined_fitness = data_refined.clone()
data_fitness['genes', 'interacts', 'genes'].edge_label = interactions[:,3]
data_refined_fitness['genes', 'interacts', 'genes'].edge_label = refined_interactions[:,3]

data_c_refined_fitness = data_c_refined.clone()
data_c_refined_fitness['genes', 'interacts', 'genes'].edge_label = refined_interactions[:,3]

data_c_fitness = data_c.clone()
data_c_fitness['genes', 'interacts', 'genes'].edge_label = interactions[:,3]

data_DMA30_c_fitness = data_DMA30_c.clone()
data_DMA30_c_fitness['genes', 'interacts', 'genes'].edge_label = interactions_DMA30[:,3]

data_refined_DMA30_c_fitness = data_refined_DMA30_c.clone()
data_refined_DMA30_c_fitness['genes', 'interacts', 'genes'].edge_label = refined_interactions_DMA30[:,3]
# %%
# with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_fitness.pkl'), 'wb') as fo:
#     pickle.dump(data_fitness, fo)

# with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_refined_fitness.pkl'), 'wb') as fo:
#     pickle.dump(data_refined_fitness, fo)

# with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_c_fitness.pkl'), 'wb') as fo:
#     pickle.dump(data_c_fitness, fo)

# with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_c_refined_fitness.pkl'), 'wb') as fo:
#     pickle.dump(data_c_refined_fitness, fo)

with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_c_DMA30_fitness.pkl'), 'wb') as fo:
    pickle.dump(data_DMA30_c_fitness, fo)

# with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_c_refined_DMA30_fitness.pkl'), 'wb') as fo:
#     pickle.dump(data_refined_DMA30_c_fitness, fo)
# %%
