import os, pickle, sys
from copy import copy
from parameters import (DATASET, LR, GNN_CHANNELS, NN_CHANNELS, LR, EPOCHS)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.cuda import is_available
from train_loop import node_split, custom_train_loop

from model import Regressor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embeddings')))


if is_available():
    device = 'cuda'
else:
    device = 'cpu'

with open(os.path.join(BASE, 'code/prediction_models/cross_val_indices.pkl'), 'rb') as fi:
    indices = pickle.load(fi)

for i, (t_idx, v_idx) in enumerate(indices):
    print(f"Fold: {i}")
    if i == 3:
        break
    with open(os.path.join(BASE, f'datasets/split_datasets/{DATASET}.pkl'),
              'rb') as fi:
        data = pickle.load(fi).contiguous()

    emb = copy({k: v.detach().clone() for k,v in data.x_dict.items()})
    # model_kwargs = {'gnn_channels': GNN_CHANNELS, 'nn_channels': NN_CHANNELS,
    #             'meta_data': data.metadata(), 'embeddings': emb}
    
    train_data, val_data = node_split(data=data, t_idx=t_idx, v_idx=v_idx,
                                      device='cpu')
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    skip_edge = [e for e in train_data.edge_types if
                 train_data[e].edge_index.shape[1] < 1000]
    skip_edge.append(('genes', 'interacts', 'genes'))
    for e in train_data.edge_types:
        if ('reg' in e[1] and e[1] not in [
                    'pos_regulating', 'neg_regulating', 'unspec_regulating',
                    'rev_pos_regulating', 'rev_neg_regulating',
                    'rev_unspec_regulating'
                ]) or e[1] in ['catalyzedBy', 'encodedBy',
                           'rev_catalyzedBy', 'rev_encodedBy']:
            skip_edge.append(e)
    edge_types = {e: v.shape[1] for e, v in train_data.edge_index_dict.items()
                  if e not in skip_edge}

    model = Regressor(gnn_channels=GNN_CHANNELS, nn_channels=NN_CHANNELS, meta_data=data.metadata(), embeddings=emb, edge_types=edge_types)
    model.to(device)

    if model.gnn:
        sample_depth = len(model.gnn.layers)
        neighbor_samples = [-1] * sample_depth
        neighbors = {t: [0] * sample_depth if t in skip_edge
                     else neighbor_samples for t in train_data.edge_types}
        val_neighbors = {t: [0] * sample_depth if t in skip_edge else
                         [-1] * sample_depth for t in train_data.edge_types}
    else:
        neighbors = val_neighbors =  [0]
        
    model.set_neighbors_to_sample(neighbors, val_neighbors)

    custom_train_loop(model=model, train_data=train_data, val_data=val_data, lr=LR, epochs=EPOCHS, device=device)    

   
