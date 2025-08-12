import tqdm
import numpy as np
import torch as th
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import sort_edge_index, degree
import torch_geometric.transforms as T
import copy
from itertools import chain

from torch.nn.functional import mse_loss
from sklearn.metrics import r2_score

from box_embeddings.parameterizations import MinDeltaBoxTensor, SigmoidBoxTensor#TanhBoxTensor
from box_embeddings.modules.intersection import GumbelIntersection
from box_embeddings.modules.volume import BesselApproxVolume

from sklearn.model_selection import KFold
from sklearn.utils import gen_batches
from parameters import (LR_DECAY, SCHEDULE_RATE, TRAIN_EMBEDDING_EPOCH,
                        TRAIN_GENES, BOX_WEIGHT, REGULARIZATION, DATASET,
                        MIN_NBR_EDGES, SEMANTIC_WEIGHT, NEG_WEIGHT)

import os, pickle
seed_everything(42)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copy import deepcopy, copy

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
    if transform != None:
        new_data, _, _ = transform(new_data)
    else:
        new_data['genes', 'interacts', 'genes'].edge_label_index = new_data['genes', 'interacts', 'genes'].edge_index

    new_data = add_reverse_edges(new_data)
    return new_data

def link_split(data, split_transform, v_idx, t_idx=None, device='cpu'):
    train_data = get_data_from_idx(data, t_idx, split_transform)
    val_data = get_data_from_idx(data, v_idx, split_transform)
    return train_data, val_data

def node_split(data, v_idx, t_idx=None, split_transform=None, device='cpu'):
    cpu_data = data['genes', 'interacts', 'genes'].edge_index.to('cpu')
    idx_tensor = th.tensor(v_idx, device='cpu')
    mask = (cpu_data.unsqueeze(2) == idx_tensor).any(dim=2)
    v_mask = mask.all(dim=0).to(device)
    t_mask = (~mask).all(dim=0).to(device)
    val_data = get_data_from_idx(data, v_mask, split_transform)
    train_data = get_data_from_idx(data, t_mask, split_transform)
    return train_data, val_data


def cross_val(model_type, model_kwargs, data, epochs, loss_function, metric,
              device, folds=10, lr=0.001, split='nodes', gci0_data=None, num_batches=2):
    print(f"Splitting {split}")
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    metrics = []
    best_models = []
    # split_transform = T.RandomLinkSplit(
    #     num_val=0.0,
    #     num_test=0.0,
    #     neg_sampling_ratio=0.0,
    #     add_negative_train_samples=False,
    #     edge_types=("genes", "interacts", "genes")
    # )
    split_transform = None
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
        print(f"Fold: {i}", flush=True)
        #if i > 2:
        #    break
        train_data, val_data = split_data(data=data.clone(), t_idx=t_idx, v_idx=v_idx,
                                          split_transform=split_transform,
                                          device=device)
        
        fold_metrics, fold_model = train_loop(model_type, train_data, val_data,
                                              epochs, loss_function, metric,
                                              device, model_kwargs, lr,
                                              gci0_data, num_batches)
        metrics.append(fold_metrics)
        best_models.append(fold_model.cpu())
        best_metrics.append(fold_metrics['best_metric'])
        data_splits.append((train_data.to('cpu'), val_data.to('cpu')))
        #break

    print(f"Average best metric over folds: {np.mean(best_metrics)}")
    print(f"Std best metric over folds: {np.std(best_metrics)}")

    return metrics, best_models, data_splits

def box_loss(embeddings, gci0, loss_type='inclusion', box_transform='mindelta',
             inter='gumbel', inter_temp=0.1, vol='bessel', vol_temp=0.1,
             gamma=0.0, neg=False, return_layer_loss=False, **kwargs):
    match box_transform:
        case 'mindelta':
            box = MinDeltaBoxTensor
        case 'sigmoid':
            box = SigmoidBoxTensor
        case _:
            raise NotImplementedError()
    if loss_type == 'inclusion':
        return box_loss_inclusion(embeddings, gci0, box=box, inter=inter,
                                  inter_temp=inter_temp, vol=vol,
                                  vol_temp=vol_temp, neg=neg,
                                  return_layer_loss=return_layer_loss)
    if loss_type == 'distance':
        return box_loss_distance(embeddings, gci0, box=box, gamma=gamma,
                                 neg=neg, return_layer_loss=return_layer_loss)
    pass

def box_loss_inclusion(embeddings, gci0, box=MinDeltaBoxTensor, inter='gumbel',
             inter_temp=0.1, vol='bessel', vol_temp=0.1, neg=False,
             return_layer_loss=return_layer_loss, **kwargs):
    if neg:
        raise NotImplementedError("Negative loss not yet implemented for inclusion loss")
    match inter:
        case 'gumbel':
            intersect = GumbelIntersection(intersection_temperature=inter_temp)
        case _:
            raise NotImplementedError()
        
    match vol:
        case 'bessel':
            volume = BesselApproxVolume(intersection_temperature=inter_temp,
                                        volume_temperature=vol_temp, log_scale=False)
    loss = 0
    layer_losses = []
    for x_dict in embeddings:
        layer_loss = 0
        for k, emb in x_dict.items():
            
            if k == 'genes':
                continue
            box_emb = box.from_vector(emb)
            
            subclasses = box_emb[gci0[k][:,0], ...]
            supclasses = box_emb[gci0[k][:,1], ...]

            loss -= (volume(intersect(subclasses, supclasses)) / volume(subclasses)).clamp(min=1e-9, max=1).log().sum()
            layer_loss -= (volume(intersect(subclasses, supclasses)) / volume(subclasses)).clamp(min=1e-9, max=1).log().sum().detach().item()
        layer_losses.append(layer_loss)
    if return_layer_loss:
        return loss, 0, layer_losses
    return loss, 0


def box_loss_distance(embeddings, gci0, box=MinDeltaBoxTensor, gamma=0.0,
                      neg=False, return_layer_loss=return_layer_loss):

    def dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False):
        n = -1 if neg else 1
        return th.relu(n*(th.abs(sub_c - sup_c) + sub_o - sup_o -
                          gamma)).norm(dim=-1).sum()
    loss = 0
    neg_loss = 0
    layer_losses = []
    for x_dict in embeddings:
        layer_loss = 0
        for k, emb in x_dict.items():
            if k == 'genes':
                continue
            box_emb = box.from_vector(emb)
            
            subclasses = box_emb[gci0[k][:,0], ...]
            sub_c, sub_o = subclasses.centre, subclasses.Z - subclasses.centre
            supclasses = box_emb[gci0[k][:,1], ...]
            sup_c, sup_o = supclasses.centre, supclasses.Z - supclasses.centre

            loss += dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False)
            layer_loss += dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False).detach().item()
            
            # th.relu(th.abs(sub_c - sup_c) + sub_o - sup_o -
            #                 gamma).norm(dim=-1).sum()
            
            if neg:
                max_i = len(emb)

                rand_classes = th.randint(low=0, high=max_i, size=(len(gci0[k]),), device=gci0[k].device)
                nsub = box_emb[rand_classes, ...]
                nsub_c, nsub_o = nsub.centre, nsub.Z - nsub.centre
                neg_loss += dist_inclusion(nsub_c, nsub_o, sup_c, sup_o, neg=True)

                rand_classes = th.randint(low=0, high=max_i, size=(len(gci0[k]),), device=gci0[k].device)
                nsup = box_emb[rand_classes, ...]
                nsup_c, nsup_o = nsup.centre, nsup.Z - nsup.centre
                neg_loss += dist_inclusion(sub_c, sub_o, nsup_c, nsup_o, neg=True)

                rand_classes = th.randint(low=0, high=max_i, size=(len(gci0[k]),2), device=gci0[k].device)
                nsub = box_emb[rand_classes[:,0], ...]
                nsub_c, nsub_o = nsub.centre, nsub.Z - nsub.centre
                nsup = box_emb[rand_classes[:,1], ...]
                nsup_c, nsup_o = nsup.centre, nsup.Z - nsup.centre
                neg_loss += dist_inclusion(nsub_c, nsub_o, nsup_c, nsup_o, neg=True)

                


            # loss -= (volume(intersect(subclasses, supclasses)) / volume(subclasses)).clamp(min=1e-9, max=1).log().sum()
        layer_losses.append(layer_loss)

    if return_layer_loss:
        return loss, neg_loss, layer_losses
    return loss, neg_loss

def train_loop(model_type, train_data, val_data, epochs, loss_function, metric,
               device, model_kwargs, lr=0.001, gci0_data=None, num_batches=2):
    
    skip_edge = [e for e in train_data.edge_types if
                 train_data[e].edge_index.shape[1] < MIN_NBR_EDGES]
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
    
    # edge_index_max = {e: int(degree(train_data[e].edge_index[1]).max().item())
    #                   for e in edge_types}

    model = model_type(edge_types=edge_types, inter_temp=0.1, aggr='attn', **model_kwargs)
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
    
    # train_loader = LinkNeighborLoader(
    #     data=train_data,
    #     num_neighbors=neighbors,
    #     edge_label_index=(('genes', 'interacts', 'genes'),
    #                       train_data['genes', 'interacts',
    #                                  'genes'].edge_label_index),
    #     edge_label=train_data['genes', 'interacts', 'genes'].edge_label,
    #     batch_size=2**25,
    #     shuffle=True,
    # )

    #val_loader = LinkNeighborLoader(
    #    data=val_data,
    #    num_neighbors=val_neighbors,
    #    edge_label_index=(('genes', 'interacts', 'genes'),
    #                      val_data['genes', 'interacts',
    #                               'genes'].edge_label_index),
    #    edge_label=val_data['genes', 'interacts', 'genes'].edge_label,
    #    batch_size=2**20,
    #)

    metrics = {'train_losses': [], 'train_metrics': [], 'val_losses': [],
               'val_metrics': [],
               'box_losses': {k: [] for k in model.node_embeddings.keys()}}
    optimizer = th.optim.Adam([
            {'params': model.node_embeddings.parameters(), 'weight_decay': 0},
            {'params': model.gnn.parameters(), 'weight_decay': REGULARIZATION},
            {'params': chain(model.lin4.parameters(), model.lin_layers.parameters())}
                        ], lr=lr, weight_decay=REGULARIZATION)
    # scheduler = th.optim.lr_scheduler.MultiplicativeLR(optimizer,
    #                                                    lambda epoch: 0.1)
    # decreased = False
    #print('inclusion')
    print('distance')
    best_metric = -np.inf
    model.node_embeddings.requires_grad_(True)
    model.node_embeddings['genes'].requires_grad_(False)
    for e in edge_types:
        train_data[e].edge_index = sort_edge_index(train_data[e].edge_index, sort_by_row=False)
        val_data[e].edge_index = sort_edge_index(val_data[e].edge_index, sort_by_row=False)
    train_data.to(device)
    val_data.to(device)
    edge_indices = train_data['genes', 'interacts', 'genes'].edge_label_index
    edge_labels = train_data['genes', 'interacts', 'genes'].edge_label
    # if gci0_data:
    #     gci0_da
    # train_data.cuda()
    num_data = train_data['genes', 'interacts', 'genes'].edge_label_index.shape[1]
    #num_batches = 5
    increased_lr = False
    epoch_layer_losses = []
    print(f"Trainable parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    for epoch in range(1, epochs+1):
        # if epoch > TRAIN_EMBEDDING_EPOCH:
        td = train_data.clone()
        perm = np.random.permutation(range(num_data))
        #td['genes', 'interacts', 'genes'].edge_label_index = td['genes', 'interacts', 'genes'].edge_label_index[:, perm]
        #td['genes', 'interacts', 'genes'].edge_index = td['genes', 'interacts', 'genes'].edge_index[:, perm]
        #td['genes', 'interacts', 'genes'].edge_label = td['genes', 'interacts', 'genes'].edge_label[perm]
        td.to(device)
        # model.node_embeddings.requires_grad_(False)
        # model.node_embeddings['genes'].requires_grad_(TRAIN_GENES)
        total_loss = total_examples = 0
        total_sem_loss = total_neg_sem_loss = 0
        # all_labels = []
        # all_preds = []
        sem_loss = neg_sem_loss = 0
        # box_loss_epoch = {k: [] for k in model.node_embeddings.keys()}
        # for sampled_data in tqdm.tqdm(train_loader):
        all_targets = []
        all_preds = []
        for s in gen_batches(num_data, num_data // num_batches):
            batch_indices = edge_indices[:, perm][:, s]
            batch_labels = edge_labels[perm][s]
            td['genes', 'interacts', 'genes'].edge_label_index = batch_indices
            optimizer.zero_grad()
            if gci0_data:
                preds, x_dicts = model(td, return_embs=True)
                sem_loss, neg_sem_loss, layer_losses = box_loss(x_dicts, gci0_data,
                                                loss_type='distance', neg=False, return_layer_loss=True)
                total_sem_loss += sem_loss.detach().item()
                epoch_layer_losses.append(layer_losses)
                if isinstance(neg_sem_loss, th.Tensor):
                    total_neg_sem_loss += neg_sem_loss.detach().item()
                
            else:
                preds = model(td)

            # loss = loss_function(preds, td['genes', 'interacts',
            #                                         'genes'].edge_label,
            #                     reduction='sum') 
            loss = loss_function(preds, batch_labels,
                                reduction='sum') 
            
            total_loss += loss.detach().item()

            combined_loss = loss + SEMANTIC_WEIGHT * (sem_loss + NEG_WEIGHT * neg_sem_loss) / num_batches
            combined_loss.backward()
            optimizer.step()
            
            total_examples += preds.numel()

            all_targets.extend(batch_labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())

        # all_targets = np.array(all_targets).flatten()
        # all_preds = np.array(all_preds).flatten()
        tm = metric(all_targets, all_preds)
        #if tm > -0.1:
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = lr
                
        print(f"Epoch: {epoch:04d}")
        print(f"train loss: {total_loss / total_examples}")
        print(f"semantic loss: {sem_loss}")
        print(f"neg semantic loss: {neg_sem_loss}")
        print(f"train metric: {tm}")

        with th.no_grad():
            total_val_loss = val_examples = 0
        
            preds = model(val_data)
            total_val_loss += loss_function(
                    preds,
                    val_data['genes', 'interacts', 'genes'].edge_label,
                    reduction='sum'
            ).item()

            val_examples += preds.numel()
            targets = val_data['genes','interacts',
                          'genes'].edge_label.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            
            vm = metric(targets, preds)
            print(f"val loss: {total_val_loss / val_examples}")
            print(f"val metric: {vm}", flush=True)

        metrics['train_losses'].append(total_loss / total_examples)
        metrics['train_metrics'].append(tm)
        metrics['val_losses'].append(total_val_loss / val_examples)
        metrics['val_metrics'].append(vm)
        #if gci0_data:
        #    for k,v in box_loss_epoch.items():
        #        metrics['box_losses'][k].append(np.mean(v).item())

        # if vm > 0 and not decreased:
        #     scheduler.step()
        #     decreased = True
        if increased_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if vm > best_metric:
            since_improved = 0
            best_metric = vm
            print('copying model...', flush=True)
            best_model = deepcopy(model)
        else:
            since_improved += 1
        if since_improved == 10:
            print("increasing lr for one epoch")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 10*lr
            increased_lr = True
        if since_improved > 50:
            print('Model has not improved in 20 epochs, stopping training...', flush=True)
            if best_metric < 0.20:
                print("Restarting training for this fold", flush=True)
                return train_loop(model_type=model_type, train_data=train_data,
                                  val_data=val_data, epochs=epochs,
                                  loss_function=loss_function, metric=metric,
                                  device=device, model_kwargs=model_kwargs,
                                  lr=lr/1.5, gci0_data=gci0_data, num_batches=num_batches+1)
            break

    metrics['best_metric'] = best_metric
    th.cuda.empty_cache()
    print(f'BEST METRIC FOR FOLD: {best_metric}', flush=True)
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
                 if data[e].edge_index.shape[1] < MIN_NBR_EDGES]
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
