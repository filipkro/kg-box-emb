# %%
from model import Regressor
from sklearn.metrics import r2_score
from torch.nn.functional import mse_loss
from sklearn.model_selection import KFold
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
import torch as th
from itertools import chain
import numpy as np
import os, pickle, time
from bayes_opt import BayesianOptimization
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
    cpu_data = data['genes', 'interacts', 'genes'].edge_index.to('cpu')
    idx_tensor = th.tensor(v_idx, device='cpu')
    mask = (cpu_data.unsqueeze(2) == idx_tensor).any(dim=2)
    v_mask = mask.all(dim=0).to(device)
    t_mask = (~mask).all(dim=0).to(device)
    val_data = get_data_from_idx(data, v_mask, split_transform)
    train_data = get_data_from_idx(data, t_mask, split_transform)
    return train_data, val_data
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = 'pyg_graph_c_DMA30_fitness'
device = 'cuda'

with open(os.path.join(BASE, f'datasets/split_datasets/{DATASET}.pkl'),
          'rb') as fi:
    data = pickle.load(fi).contiguous()
data.to(device)
# %%
epochs = 900
PARAMS = {'GNN_depth': (1,4), 'GNN_channels': (1,7), 'lin_depth': (1,4),
          'lin_width': (1,7), 'lr': (1,6), 'reg': (1,6), 'batch_size': (14,18)}


kf = KFold(n_splits=5, shuffle=True, random_state=0)
split_transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=("genes", "interacts", "genes")
    )
data_to_split = data['genes'].node_id
t_idx, v_idx = next(kf.split(data_to_split))
# %%
train_data, val_data = node_split(data=data, t_idx=t_idx, v_idx=v_idx,
                                  split_transform=split_transform,
                                  device=device)
# %%
skip_edge = [e for e in data.edge_types if data[e].edge_index.shape[1] < 1000]
skip_edge.append(('genes', 'interacts', 'genes'))
for e in data.edge_types:
    if ('reg' in e[1] and e[1] not in [
                    'pos_regulating', 'neg_regulating', 'unspec_regulating',
                    'rev_pos_regulating', 'rev_neg_regulating',
                    'rev_unspec_regulating'
                ]) or e[1] in ['catalyzedBy', 'encodedBy',
                           'rev_catalyzedBy', 'rev_encodedBy']:
        skip_edge.append(e)
edge_types = [e for e in data.edge_types if e not in skip_edge]
# %%
def eval_params(GNN_depth, GNN_channels, lin_depth, lin_width, lr, reg,
                batch_size):
    GNN_depth = int(GNN_depth)
    GNN_channels = int(GNN_channels)
    lin_depth = int(lin_depth)
    lin_width = int(lin_width)
    batch_size = int(batch_size)
    reg = int(reg)
    lr = int(lr)
    print(GNN_depth)
    print(GNN_channels)
    print(lin_depth)
    print(lin_width)
    print(batch_size)
    print(reg)
    print(lr)

    gnn_channels = []
    for i in range(GNN_depth):
        gnn_channels.append(2**GNN_channels)

    nn_channels = []
    for i in range(lin_depth):
        if i > 0 and i == lin_depth-1:
            nn_channels.append(2**(lin_width // 2))
        else:
            nn_channels.append(2**lin_width)

    model_kwargs = {'gnn_channels': gnn_channels, 'nn_channels': nn_channels,
                'meta_data': data.metadata(), 'embeddings': data.x_dict}
    
    
    model = Regressor(edge_types=edge_types, **model_kwargs)
    model.to(device)
    model.node_embeddings['genes'].requires_grad_(False)

    sample_depth = len(model.gnn.layers)
    neighbor_samples = [-1] * sample_depth
    
    neighbors = {t: [0] * sample_depth if t in skip_edge
                    else neighbor_samples for t in train_data.edge_types}
    val_neighbors = {t: [0] * sample_depth if t in skip_edge else
                        [-1] * sample_depth for t in train_data.edge_types}
    
    model.set_neighbors_to_sample(neighbors, val_neighbors)

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=neighbors,
        edge_label_index=(('genes', 'interacts', 'genes'),
                          train_data['genes', 'interacts',
                                     'genes'].edge_label_index),
        edge_label=train_data['genes', 'interacts', 'genes'].edge_label,
        batch_size=2**batch_size,
        shuffle=True,
    )

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=val_neighbors,
        edge_label_index=(('genes', 'interacts', 'genes'),
                          val_data['genes', 'interacts',
                                   'genes'].edge_label_index),
        edge_label=val_data['genes', 'interacts', 'genes'].edge_label,
        batch_size=2**batch_size,
    )
    optimizer = th.optim.Adam([
            {'params': chain(model.gnn.parameters(), model.lin4.parameters(),
                             model.lin_layers.parameters())}
                              ], lr=10**(-lr), weight_decay=10**(-reg))

    since_improved = 0
    best_metric = -np.inf
    model.node_embeddings['genes'].requires_grad_(False)
    model.node_embeddings.requires_grad_(False)
    for epoch in range(1, epochs+1):
        total_loss = total_examples = 0
        all_labels = []
        all_preds = []
        for sampled_data in train_loader:
            sampled_data.to(device)
            optimizer.zero_grad()
            preds = model(sampled_data)
            loss = mse_loss(preds, sampled_data['genes', 'interacts',
                                                     'genes'].edge_label,
                                 reduction='sum')
            
            total_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
            total_examples += preds.numel()
            targets = sampled_data['genes','interacts',
                          'genes'].edge_label.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            all_labels = np.concatenate((all_labels, targets))
            all_preds = np.concatenate((all_preds, preds))

        tm = r2_score(all_labels, all_preds)
        with th.no_grad():
            total_val_loss = val_examples = 0
            val_labels = []
            val_preds = []
        
            for sampled_data in val_loader:
                sampled_data.to(device)
                preds = model(sampled_data)
                total_val_loss += mse_loss(
                        preds,
                        sampled_data['genes', 'interacts', 'genes'].edge_label,
                        reduction='sum'
                ).item()

                val_examples += preds.numel()
                targets = sampled_data['genes','interacts',
                          'genes'].edge_label.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()
                val_labels = np.concatenate((val_labels, targets))
                val_preds = np.concatenate((val_preds, preds))
            vm = r2_score(val_labels, val_preds)

        if vm > best_metric:
            since_improved = 0
            best_metric = vm
        else:
            since_improved += 1
        if since_improved >20:
            break
    return best_metric

# %%
optimizer = BayesianOptimization(
    f=eval_params,
    pbounds=PARAMS,
    verbose=2,
    random_state=1,
)
# %%
optimizer.maximize(
    init_points=4,
    n_iter=20,
)
# %%
print(optimizer.max)
file_name = time.strftime("%Y%m%d-%H%M%S") + 'optimizer.pkl'
with open(os.path.join(BASE, 'optims', file_name), 'wb') as fo:
    pickle.dump(optimizer, fo)
