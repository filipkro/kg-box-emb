# %%
import os, pickle
from sklearn.model_selection import KFold
from comparison_model import InstantiationModelLight
import torch as th
import torch_geometric.transforms as T
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# %%
def get_data_from_idx(data, idx, transform):
    new_data = data.clone()
    new_data['genes', 'interacts', 'genes'].edge_index = \
        new_data['genes', 'interacts', 'genes'].edge_index[:, idx]
    new_data['genes', 'interacts', 'genes'].edge_label = \
        new_data['genes', 'interacts', 'genes'].edge_label[idx]
    new_data, _, _ = transform(new_data)

    return new_data

def node_split(data, split_transform, v_idx, t_idx=None, device='cpu'):
    mask = (data['genes', 'interacts', 'genes'].edge_index.unsqueeze(2) ==
            th.tensor(v_idx, device=device)).any(dim=2)
    v_mask = mask.all(dim=0)
    t_mask = (~mask).all(dim=0)
    val_data = get_data_from_idx(data, v_mask, split_transform)
    train_data = get_data_from_idx(data, t_mask, split_transform)
    return train_data, val_data
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = 'cpu'
# %%
with open(os.path.join(BASE, 'datasets/split_datasets/pyg_graph_c_DMA30_fitness.pkl'), 'rb') as fi:
    data = pickle.load(fi).contiguous()
# %%
kf = KFold(n_splits=10, shuffle=True, random_state=42)
split_transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=("genes", "interacts", "genes")
    )
data_to_split = data['genes'].node_id
split_data = node_split

# %%
feats = data['genes', 'hasChemStressResistance', 'mat_ent']['edge_index']
feats = data['genes', 'hasChemCellMorph', 'mat_ent']['edge_index']
feats = data['genes', 'RO_0002200', 'quality']['edge_index']
# feat
with open(os.path.join(BASE, 'datasets/split_datasets/quality.pkl'), 'rb') as fi:
    quality_index = pickle.load(fi)['index']['class_index']
rev_quality = {v: k for k,v in quality_index.items()}
# %%
vscores = []
tscores = []
models = []
for i, (t_idx, v_idx) in enumerate(kf.split(data_to_split)):
    print(f"Fold: {i}")
    train_data, val_data = split_data(data=data, t_idx=t_idx, v_idx=v_idx,
                                      split_transform=split_transform,
                                      device=device)
    train_x = train_data['genes', 'interacts', 'genes']['edge_label_index'].numpy()
    # train_x = train_data['genes', 'interacts', 'genes']['edge_label_index'].numpy()[:,:1000000]
    train_y = train_data['genes', 'interacts', 'genes']['edge_label'].numpy()
    # train_y = train_data['genes', 'interacts', 'genes']['edge_label'].numpy()[:1000000]
    val_x = val_data['genes', 'interacts', 'genes']['edge_label_index'].numpy()
    val_y = val_data['genes', 'interacts', 'genes']['edge_label'].numpy()

    model = InstantiationModelLight(feats, n_jobs=16)

    print('model created')
    model.fit(train_x, train_y)

    train_preds = model.predict(train_x)
    val_preds = model.predict(val_x)
    tscore = r2_score(train_y, train_preds)
    vscore = r2_score(val_y, val_preds)
    vscores.append(vscore)
    tscores.append(tscore)
    # plt.figure()
    # plt.scatter(val_y, val_preds, s=0.1)
    # plt.ylim((-0.05,1.5))
    # plt.xlim((-0.05,1.5))
    print(f"train r2: {tscore}")
    print(f"val r2: {vscore}")
    models.append({'model': model, 'train_score': tscore, 'val_score': vscore})
    
    # break

# plt.show()
print(f"train: {np.mean(tscores)} +- {np.std(tscores)}")
print(f"val: {np.mean(vscores)} +- {np.std(vscores)}")
# %%