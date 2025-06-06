import tqdm
import numpy as np
import torch as th
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
import copy
from itertools import chain

from sklearn.model_selection import KFold
from parameters import (LR_DECAY, SCHEDULE_RATE, TRAIN_EMBEDDING_EPOCH,
                        TRAIN_GENES, BOX_WEIGHT, REGULARIZATION, DATASET)

seed_everything(42)

def add_reverse_edges(data):
    data['genes','interacts','genes'].edge_index = \
        th.cat([data['genes', 'interacts', 'genes'].edge_index,
                th.flip(data['genes', 'interacts', 'genes'].edge_index,
                        dims=(0,))], dim=1)
    data['genes','interacts','genes'].edge_label_index = \
        th.cat([data['genes', 'interacts', 'genes'].edge_label_index,
                th.flip(data['genes', 'interacts', 'genes'].edge_label_index,
                        dims=(0,))], dim=1)
    data['genes','interacts','genes'].edge_label = \
        th.cat([data['genes', 'interacts', 'genes'].edge_label,
                data['genes', 'interacts', 'genes'].edge_label], dim=0)
    
    return data

def get_targets_preds(sampled_data, preds):
    labels = sampled_data['genes','interacts',
                          'genes'].edge_label.detach().cpu().numpy()
    labels = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
    
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1) if len(preds.shape) > 1 else preds

    return labels, preds

def get_data_from_idx(data, idx, transform):
    new_data = data.clone()
    new_data['genes', 'interacts', 'genes'].edge_index = \
        new_data['genes', 'interacts', 'genes'].edge_index[:, idx]
    new_data['genes', 'interacts', 'genes'].edge_label = \
        new_data['genes', 'interacts', 'genes'].edge_label[idx]
    new_data, _, _ = transform(new_data)

    new_data = add_reverse_edges(new_data)
    return new_data

def link_split(data, split_transform, v_idx, t_idx=None, device='cpu'):
    train_data = get_data_from_idx(data, t_idx, split_transform)
    val_data = get_data_from_idx(data, v_idx, split_transform)
    return train_data, val_data

def node_split(data, split_transform, v_idx, t_idx=None, device='cpu'):
    cpu_data = data['genes', 'interacts', 'genes'].edge_index.to('cpu')
    idx_tensor = th.tensor(v_idx, device='cpu')
    mask = (cpu_data.unsqueeze(2) == idx_tensor).any(dim=2)
    v_mask = mask.all(dim=0).to(device)
    t_mask = (~mask).all(dim=0).to(device)
    val_data = get_data_from_idx(data, v_mask, split_transform)
    train_data = get_data_from_idx(data, t_mask, split_transform)
    return train_data, val_data

def val_model_params(model_type, model_kwargs, data, epochs, loss_function, metric,
              device, folds=10, lr=0.001, split='nodes', gci0_data=None):
    print(f"Splitting {split}")
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    metrics = []
    best_models = []
    split_transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=("genes", "interacts", "genes")
    )
    best_metrics = []
    if split == 'nodes':
        data_to_split = data['genes'].node_id
        split_data = node_split
    elif split == 'links':
        data_to_split = data['genes', 'interacts', 'genes'].edge_index.T
        split_data = link_split
    else:
        raise NotImplementedError(f"split for {split} is not implemented."
                                  "Use nodes or links")
    
    for gnn, nn in zip([[16,32], [64,64], [32,32]], [[64], [64,8], [64, 16]]):
        for i, (t_idx, v_idx) in enumerate(kf.split(data_to_split)):
            print(f"Fold: {i}")
            model_kwargs['gnn_channels'] = gnn
            model_kwargs['nn_channels'] = nn
            print(f"GNN channels: {gnn}")
            print(f"NN : {nn}")
            train_data, val_data = split_data(data=data, t_idx=t_idx,
                                              v_idx=v_idx,
                                              split_transform=split_transform,
                                              device=device)


            fold_metrics, fold_model = train_loop(model_type, train_data,
                                                  val_data, epochs,
                                                  loss_function, metric, device,
                                                  model_kwargs, lr, gci0_data)
            metrics.append(fold_metrics)
            best_models.append(fold_model.cpu())
            best_metrics.append(fold_metrics['best_metric'])
            break

    print(f"Average best metric over folds: {np.mean(best_metrics)}")
    print(f"Std best metric over folds: {np.std(best_metrics)}")

    return metrics, best_models

def cross_val(model_type, model_kwargs, data, epochs, loss_function, metric,
              device, folds=10, lr=0.001, split='nodes', gci0_data=None):
    print(f"Splitting {split}")
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    metrics = []
    best_models = []
    split_transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=("genes", "interacts", "genes")
    )
    best_metrics = []
    if split == 'nodes':
        data_to_split = data['genes'].node_id
        split_data = node_split
    elif split == 'links':
        data_to_split = data['genes', 'interacts', 'genes'].edge_index.T
        split_data = link_split
    else:
        raise NotImplementedError(f"split for {split} is not implemented."
                                  "Use nodes or links")

    data_splits = []
    for i, (t_idx, v_idx) in enumerate(kf.split(data_to_split)):
        print(f"Fold: {i}")
        train_data, val_data = split_data(data=data, t_idx=t_idx, v_idx=v_idx,
                                          split_transform=split_transform,
                                          device=device)
        
        fold_metrics, fold_model = train_loop(model_type, train_data, val_data,
                                              epochs, loss_function, metric,
                                              device, model_kwargs, lr,
                                              gci0_data)
        metrics.append(fold_metrics)
        best_models.append(fold_model.cpu())
        best_metrics.append(fold_metrics['best_metric'])
        data_splits.append((train_data.to('cpu'), val_data.to('cpu')))
        #break

    print(f"Average best metric over folds: {np.mean(best_metrics)}")
    print(f"Std best metric over folds: {np.std(best_metrics)}")

    return metrics, best_models, data_splits

def train_loop(model_type, train_data, val_data, epochs, loss_function, metric,
               device, model_kwargs, lr=0.001, gci0_data=None):
    
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

    model = model_type(edge_types=edge_types, **model_kwargs)
    model.to(device)
    model.node_embeddings['genes'].requires_grad_(TRAIN_GENES)
    
    since_improved = 0
    
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
    
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=neighbors,
        edge_label_index=(('genes', 'interacts', 'genes'),
                          train_data['genes', 'interacts',
                                     'genes'].edge_label_index),
        edge_label=train_data['genes', 'interacts', 'genes'].edge_label,
        batch_size=2**19,
        shuffle=True,
    )

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=val_neighbors,
        edge_label_index=(('genes', 'interacts', 'genes'),
                          val_data['genes', 'interacts',
                                   'genes'].edge_label_index),
        edge_label=val_data['genes', 'interacts', 'genes'].edge_label,
        batch_size=2**20,
    )

    metrics = {'train_losses': [], 'train_metrics': [], 'val_losses': [],
               'val_metrics': [],
               'box_losses': {k: [] for k in model.node_embeddings.keys()}}
    optimizer = th.optim.Adam([
            {'params': model.node_embeddings.parameters(), 'weight_decay': 0},
            {'params': chain(model.gnn.parameters(), model.lin4.parameters(),
                             model.lin_layers.parameters())}
                              ], lr=lr, weight_decay=REGULARIZATION)
    scheduler = th.optim.lr_scheduler.MultiplicativeLR(optimizer,
                                                       lambda epoch: 0.1)
    decreased = False
    best_metric = -np.inf
    model.node_embeddings['genes'].requires_grad_(False)
    model.node_embeddings.requires_grad_(False)
    for epoch in range(1, epochs+1):
        if epoch > TRAIN_EMBEDDING_EPOCH:
            model.node_embeddings.requires_grad_(True)
            model.node_embeddings['genes'].requires_grad_(TRAIN_GENES)
        total_loss = total_examples = 0
        all_labels = []
        all_preds = []
        box_loss_epoch = {k: [] for k in model.node_embeddings.keys()}
        for sampled_data in tqdm.tqdm(train_loader):
            sampled_data.to(device)
            optimizer.zero_grad()
            preds = model(sampled_data)
            loss = loss_function(preds, sampled_data['genes', 'interacts',
                                                     'genes'].edge_label,
                                 reduction='sum')
            
            total_loss += loss.detach().item()
            if gci0_data:
                # evaluate box losses as well
                for node, gci0 in gci0_data.items():
                    dims = int(model.node_embeddings[node].embedding_dim / 2)
   
                    r = model.node_embeddings[node](gci0)
                    
                    if '_c_' in DATASET:
                        rc, ro = r[..., :dims], r[..., dims:]
                        subl = rc[:, 0, :] - ro[:, 0, :]
                        subh = rc[:, 0, :] + ro[:, 0, :]
                        supl = rc[:, 1, :] - ro[:, 1, :]
                        suph = rc[:, 1, :] + ro[:, 1, :]
                    else:
                        rl, rh = r[..., :dims], r[..., dims:]
                        subl = rl[:, 0, :]
                        subh = rh[:, 0, :]
                        supl = rl[:, 1, :]
                        suph = rh[:, 1, :]
                    box_l = (F.relu(subh - suph) + F.relu(supl - subl) +
                             F.relu(supl - suph) + F.relu(subl - subh)).norm()
                    if epoch > TRAIN_EMBEDDING_EPOCH:
                        loss += BOX_WEIGHT * box_l
                    box_loss_epoch[node].append(box_l.detach().item())

            loss.backward()
            optimizer.step()
            
            total_examples += preds.numel()

            targets, preds = get_targets_preds(sampled_data, preds)
            all_labels = np.concatenate((all_labels, targets))
            all_preds = np.concatenate((all_preds, preds))

        tm = metric(all_labels, all_preds)
        print(f"Epoch: {epoch:04d}")
        print(f"train loss: {total_loss / total_examples}")
        print(f"train metric: {tm}")

        with th.no_grad():
            total_val_loss = val_examples = 0
            val_labels = []
            val_preds = []
        
            for sampled_data in val_loader:
                sampled_data.to(device)
                preds = model(sampled_data)
                total_val_loss += loss_function(
                        preds,
                        sampled_data['genes', 'interacts', 'genes'].edge_label,
                        reduction='sum'
                ).item()

                val_examples += preds.numel()
                targets, preds = get_targets_preds(sampled_data, preds)
                val_labels = np.concatenate((val_labels, targets))
                val_preds = np.concatenate((val_preds, preds))
            vm = metric(val_labels, val_preds)
            print(f"val loss: {total_val_loss / val_examples}")
            print(f"val metric: {vm}")

        metrics['train_losses'].append(total_loss / total_examples)
        metrics['train_metrics'].append(tm)
        metrics['val_losses'].append(total_val_loss / val_examples)
        metrics['val_metrics'].append(vm)
        if gci0_data:
            for k,v in box_loss_epoch.items():
                metrics['box_losses'][k].append(np.mean(v).item())

        # if vm > 0 and not decreased:
        #     scheduler.step()
        #     decreased = True
        if vm > best_metric:
            since_improved = 0
            best_metric = vm
            print('copying model...')
            best_model = copy.deepcopy(model)
        else:
            since_improved += 1

        if since_improved >20:
            print('Model has not improved in 20 epochs, stopping training...')
            break

    metrics['best_metric'] = best_metric
    th.cuda.empty_cache()
    print(f'BEST METRIC FOR FOLD: {best_metric}')
    return metrics, best_model

def continue_final_training(model, data, epochs, loss_function, metric, device,
                            lr=0.001):
    model.to(device)
    neighbors = model._neighbors_to_sample['neighbors']

    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=neighbors,
        edge_label_index=(('genes', 'interacts', 'genes'),
                          data['genes', 'interacts', 'genes'].edge_index),
        edge_label=data['genes', 'interacts', 'genes'].edge_label,
        batch_size=2**20,
        shuffle=True,
    )
    epochs = 200
    return final_train_loop(model, train_loader, epochs, lr, loss_function,
                            metric, device)

def train_final_model(model_type, model_kwargs, data, epochs, loss_function,
                      metric, device, lr=0.001, gci0_data=None):
    skip_edge = [e for e in data.edge_types
                 if data[e].edge_index.shape[1] < 1000]
    skip_edge.append(('genes', 'interacts', 'genes'))
    for e in data.edge_types:
         if ('reg' in e[1] and e[1] not in [
                    'pos_regulating', 'neg_regulating', 'unspec_regulating',
                    'rev_pos_regulating', 'rev_neg_regulating',
                    'rev_unspec_regulating'
                ]) or e[1] in ['catalyzedBy', 'encodedBy',
                           'rev_catalyzedBy', 'rev_encodedBy']:
            skip_edge.append(e)
    edge_types = {e: v.shape[1] for e, v in data.edge_index_dict.items()
                  if e not in skip_edge}

    model = model_type(edge_types=edge_types, **model_kwargs)
    model.to(device)
    
    if model.gnn:
        sample_depth = len(model.gnn.layers)
        neighbor_samples = [-1] * sample_depth
        neighbors = {t: [0] * sample_depth if t in skip_edge
                     else neighbor_samples for t in data.edge_types}
    else:
        neighbors = [0]
        
    model.set_neighbors_to_sample(neighbors)
    
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=neighbors,
        edge_label_index=(('genes', 'interacts', 'genes'),
                          data['genes', 'interacts', 'genes'].edge_index),
        edge_label=data['genes', 'interacts', 'genes'].edge_label,
        batch_size=2**19,
        shuffle=True,
    )

    return final_train_loop(model, train_loader, epochs, lr, loss_function,
                            metric, device)


def final_train_loop(model, train_loader, epochs, lr, loss_function, metric,
                     device, gci0_data=None):
    metrics = {'train_losses': [], 'train_metrics': [], 'val_losses': [],
               'val_metrics': [],
               'box_losses': {k: [] for k in model.node_embeddings.keys()}}

    optimizer = th.optim.Adam([
            {'params': model.node_embeddings.parameters(), 'weight_decay': 0},
            {'params': chain(model.gnn.parameters(), model.lin4.parameters(),
                             model.lin_layers.parameters())}
                              ], lr=lr, weight_decay=REGULARIZATION)
    scheduler = th.optim.lr_scheduler.MultiplicativeLR(optimizer,
                                                       lambda epoch: LR_DECAY)
    best_metric = -np.inf
    model.node_embeddings['genes'].requires_grad_(False)
    model.node_embeddings.requires_grad_(False)
    for epoch in range(1, epochs+1):
        if gci0_data and epoch > TRAIN_EMBEDDING_EPOCH:
            model.node_embeddings.requires_grad_(True)
            model.node_embeddings['genes'].requires_grad_(TRAIN_GENES)
        total_loss = total_examples = 0
        all_labels = []
        all_preds = []
        for sampled_data in tqdm.tqdm(train_loader):
            sampled_data.to(device)
            optimizer.zero_grad()
            preds = model(sampled_data)
            loss = loss_function(preds, sampled_data['genes', 'interacts',
                                                     'genes'].edge_label,
                                 reduction='sum')
            
            total_loss += loss.detach().item()

            loss.backward()
            optimizer.step()
            
            total_examples += preds.numel()

            targets, preds = get_targets_preds(sampled_data, preds)
            all_labels = np.concatenate((all_labels, targets))
            all_preds = np.concatenate((all_preds, preds))

        tm = metric(all_labels, all_preds)
        print(f"Epoch: {epoch:04d}")
        print(f"train loss: {total_loss / total_examples}")
        print(f"train metric: {tm}")

        metrics['train_losses'].append(total_loss / total_examples)
        metrics['train_metrics'].append(tm)

        if epoch % SCHEDULE_RATE == 0:
            scheduler.step()

    metrics['best_metric'] = best_metric
    th.cuda.empty_cache()
    return metrics, model
