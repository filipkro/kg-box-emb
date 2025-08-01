# %%
from memory_profiler import profile
import numpy as np
from box_embeddings.modules.volume import BesselApproxVolume
from box_embeddings.modules.intersection import GumbelIntersection
from box_embeddings.parameterizations import MinDeltaBoxTensor, SigmoidBoxTensor
from box_embeddings.modules.regularization import L2SideBoxRegularizer
from model import HeteroGNNGAT, HeteroGNNSAGE
import pickle
import os
import sys
from pprint import pprint
from model import HeteroGNNGAT, HeteroGNNSAGE, OntologyGNN
import torch
from torch_geometric import seed_everything

import logging
from time import time

from itertools import product
from tqdm.auto import tqdm

from box_forward import get_boxes_from_model_and_graph
sys.path.append(os.path.join("/", "workspaces",
                "kg-box-emb", "code", "presentation"))
from boxplot2d import plot_box_2d, animate_boxes, animate_boxes_with_blitting


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)
torch.manual_seed(42)
# %%

# %%
GNN_CHANNELS = [2*2]
LR = 5e-2
LR_DECAY = 0
REGULARIZATION = 0
# BOX_REGULARIZATION = 0
# BOX_REGULARIZATION = 1e-7
# BOX_REGULARIZATION = 1e-5
BOX_REGULARIZATION = 0.01
# BOX_REGULARIZATION = 1000
EPOCHS = 201
NEG_WEIGHT = 0.1
LOSS_TYPE = 'inclusion'
SCALE_LOSSES = False


class TrainingLogger:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.training_logs = []

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        logging.info(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time() - self.epoch_start_time
        logging.info(
            f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs['epoch_time'] = elapsed_time  # Add epoch time to logs
        self.training_logs.append(logs)  # Collect training logs


def box_loss(embeddings, gci0, loss_type='distance', box_transform='mindelta',
             inter='gumbel', inter_temp=0.1, vol='bessel', vol_temp=0.1,
             gamma=0.0, neg_data=None, neg=False, **kwargs):
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
                                  vol_temp=vol_temp, neg_data=neg_data, neg=neg)
    if loss_type == 'distance':
        return box_loss_distance(embeddings, gci0, box=box, gamma=gamma,
                                 neg_data=neg_data, neg=neg)
    pass


def box_loss_inclusion(embeddings, gci0, box=MinDeltaBoxTensor, inter='gumbel',
                       inter_temp=0.1, vol='bessel', vol_temp=0.1, neg_data=None,
                       neg=False, **kwargs):
    def neg_loss_func(A, B, volume, intersect):
        return (1 - (volume(intersect(A, B)) /
                     torch.minimum(volume(A), volume(B)))).clamp(min=1e-9,
                                                                 max=1).log().sum()

    # if neg or neg_data:
    #     raise NotImplementedError("Negative loss not yet implemented "
    #                               "for inclusion loss")
    match inter:
        case 'gumbel':
            intersect = GumbelIntersection(intersection_temperature=inter_temp)
        case _:
            raise NotImplementedError()

    match vol:
        case 'bessel':
            volume = BesselApproxVolume(intersection_temperature=inter_temp,
                                        volume_temperature=vol_temp,
                                        log_scale=False)
        case _:
            raise NotImplementedError()

    loss = 0
    neg_loss = 0
    for x_dict in embeddings:
        for k, emb in x_dict.items():

            if k == 'genes':
                continue
            box_emb = box.from_vector(emb)

            subclasses = box_emb[gci0[k][:, 0], ...]
            supclasses = box_emb[gci0[k][:, 1], ...]

            loss -= (volume(intersect(subclasses, supclasses)) /
                     volume(subclasses)).clamp(min=1e-9, max=1).log().sum()
            # print(volume(intersect(subclasses, supclasses)))
            # print(volume(subclasses))
            # print((volume(intersect(subclasses, supclasses)) /
            #          volume(subclasses)))
            # print(volume(subclasses))
            # print(volume(supclasses))
            # print(torch.minimum(volume(subclasses), volume(supclasses)))

            # print(((volume(subclasses), volume(subclasses)).min()))
            # print()

            if neg:
                max_i = len(emb)
                rand_classes = torch.randint(low=0, high=max_i,
                                             size=(len(gci0[k]),),
                                             device=gci0[k].device)
                A = box_emb[rand_classes, ...]
                neg_loss -= neg_loss_func(A, supclasses, volume, intersect)

                rand_classes = torch.randint(low=0, high=max_i,
                                             size=(len(gci0[k]),),
                                             device=gci0[k].device)
                A = box_emb[rand_classes, ...]
                neg_loss -= neg_loss_func(A, subclasses, volume, intersect)

                rand_classes = torch.randint(low=0, high=max_i,
                                             size=(len(gci0[k]), 2),
                                             device=gci0[k].device)
                A = box_emb[rand_classes[:, 0], ...]
                B = box_emb[rand_classes[:, 1], ...]
                neg_loss -= neg_loss_func(A, B, volume, intersect)

            if neg_data:
                A = box_emb[neg_data[k][:, 0], ...]
                B = box_emb[neg_data[k][:, 1], ...]

                neg_loss -= neg_loss_func(A, B, volume, intersect)

    return loss, neg_loss


def box_loss_distance(embeddings, gci0, box=MinDeltaBoxTensor, gamma=0.0,
                      neg_data=None, neg=False):

    def dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False):
        n = -1 if neg else 1
        return torch.relu(n*(torch.abs(sub_c - sup_c) + sub_o - sup_o -
                          gamma)).norm(dim=-1).sum()
    loss = 0
    neg_loss = 0
    for x_dict in embeddings:
        for k, emb in x_dict.items():
            if k == 'genes':
                continue
            box_emb = box.from_vector(emb)

            subclasses = box_emb[gci0[k][:, 0], ...]
            sub_c, sub_o = subclasses.centre, subclasses.Z - subclasses.centre
            supclasses = box_emb[gci0[k][:, 1], ...]
            sup_c, sup_o = supclasses.centre, supclasses.Z - supclasses.centre

            loss += dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False)

            if neg:
                max_i = len(emb)

                rand_classes = torch.randint(low=0, high=max_i,
                                             size=(len(gci0[k]),),
                                             device=gci0[k].device)
                nsub = box_emb[rand_classes, ...]
                nsub_c, nsub_o = nsub.centre, nsub.Z - nsub.centre
                neg_loss += dist_inclusion(nsub_c, nsub_o, sup_c, sup_o,
                                           neg=True)

                rand_classes = torch.randint(low=0, high=max_i,
                                             size=(len(gci0[k]),),
                                             device=gci0[k].device)
                nsup = box_emb[rand_classes, ...]
                nsup_c, nsup_o = nsup.centre, nsup.Z - nsup.centre
                neg_loss += dist_inclusion(sub_c, sub_o, nsup_c, nsup_o,
                                           neg=True)

                rand_classes = torch.randint(low=0, high=max_i,
                                             size=(len(gci0[k]), 2),
                                             device=gci0[k].device)
                nsub = box_emb[rand_classes[:, 0], ...]
                nsub_c, nsub_o = nsub.centre, nsub.Z - nsub.centre
                nsup = box_emb[rand_classes[:, 1], ...]
                nsup_c, nsup_o = nsup.centre, nsup.Z - nsup.centre
                neg_loss += dist_inclusion(nsub_c, nsub_o, nsup_c, nsup_o,
                                           neg=True)

            if neg_data:
                subclasses = box_emb[neg_data[k][:, 0], ...]
                sub_c = subclasses.centre
                sub_o = subclasses.Z - subclasses.centre
                supclasses = box_emb[neg_data[k][:, 1], ...]
                sup_c = supclasses.centre
                sup_o = supclasses.Z - supclasses.centre

                neg_loss += dist_inclusion(sub_c,
                                           sub_o, sup_c, sup_o, neg=True)

    return loss, neg_loss


box_regularizer = L2SideBoxRegularizer(weight=1.0, log_scale=False)
box = MinDeltaBoxTensor


def regularize_box(embeddings):
    reg_loss = 0
    for x_dict in embeddings:
        for k, emb in x_dict.items():
            box_emb = box.from_vector(emb)
            reg_loss -= box_regularizer(box_emb)
    return reg_loss


def small_box_penalty(embeddings):
    loss = 0
    for x_dict in embeddings:
        for emb in x_dict.values():
            box_emb = box.from_vector(emb)
            box_sizes = torch.norm(box_emb.Z - box_emb.z, dim=-1)
            # print(box_sizes)
            loss += torch.relu(1/box_sizes - 1).sum()
    return loss


# @profile
def train_boxes_OntologyGNN(
    graph,
    gci,
    gnn_channels=GNN_CHANNELS,
    lr=LR,
    lr_decay=LR_DECAY,
    epochs=EPOCHS,
    loss_type=LOSS_TYPE,
    regularization=REGULARIZATION,
    box_regularization=BOX_REGULARIZATION,
    neg_weight=NEG_WEIGHT,
    scale_losses=SCALE_LOSSES,
    save_weights=False,
):
    print(f"""

GNN_CHANNELS: {gnn_channels}
LR: {lr}
LR_DECAY: {lr_decay}
EPOCHS: {epochs}
LOSS_TYPE: {loss_type}
REGULARIZATION: {regularization}
BOX_REGULARIZATION: {box_regularization}
NEG_WEIGHT: {neg_weight}
SCALE_LOSSES: {scale_losses}"""
        )
    # model = HeteroGNNGAT(GNN_CHANNELS, graph.edge_types, graph.x_dict)
    # model = HeteroGNNSAGE(GNN_CHANNELS, graph.edge_types, graph.x_dict)
    model = OntologyGNN(gnn_channels, graph.edge_types, graph.x_dict)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=regularization)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # %%
    model.requires_grad_(True)

    boxes = []
    weights = [] if save_weights else None
    last_epoch = 0
    try:
        for epoch in range(epochs):
            optimizer.zero_grad()

            # x_dicts = model(graph.x_dict, graph.edge_index_dict, return_embs=True)
            x_dicts = [model(graph, return_embs=False)]

            pos_loss, neg_loss = box_loss(
                x_dicts, gci['gci0'], loss_type=loss_type, neg_data=gci['gci1_bot'], neg=False)
            reg_loss = small_box_penalty(x_dicts)
            pos_ratio = torch.exp(-pos_loss / len(gci['gci0']['classes']))
            neg_ratio = 1 - \
                torch.exp(-neg_loss /
                          (3*len(gci['gci0']['classes']) + len(gci['gci1_bot']['classes'])))
            if scale_losses:
                pos_loss_scaled = pos_loss / len(gci['gci0']['classes'])
                neg_loss_scaled = neg_loss / \
                    (3*len(gci['gci0']['classes']) +
                     len(gci['gci1_bot']['classes']))
                loss = pos_loss_scaled + neg_weight * \
                    neg_loss_scaled + box_regularization * reg_loss
            else:
                loss = pos_loss + neg_weight * neg_loss + box_regularization * reg_loss
            # loss = neg_loss
            # loss = 1 - pos_ratio + neg_weight * neg_ratio + box_regularization * reg_loss
            total_loss = loss.detach().item()

            if loss_type == 'distance':
                print(
                    f"Epoch: {epoch}, total loss: {total_loss:.4f}, pos loss: {pos_ratio:.6f}, neg loss: {neg_ratio:.6f}, reg: {reg_loss:.3f}")
            else:
                print(
                    f"Epoch: {epoch}, total loss: {total_loss:.4f}, pos ratio: {pos_ratio:.6f}, neg ratio: {neg_ratio:.6f}, reg: {reg_loss:.8f}")

            # Backpropagate loss gradients
            loss.backward()
            optimizer.step()
        #
            if epoch % 1 == 0:
                boxes.append((
                    get_boxes_from_model_and_graph(
                        model, graph).data.detach().numpy(),
                    total_loss,
                    pos_ratio.detach().item(),
                    neg_ratio.detach().item(),
                    reg_loss.detach().item()
                ))
                if save_weights:
                    weights.append(model.state_dict())
            last_epoch = epoch

            # decay LR
            lr = lr * (1 - lr_decay)

    except KeyboardInterrupt:
        print(f"\nTraining stopped by user during Epoch {epoch}")
        boxes = boxes[:last_epoch]
    # print(MinDeltaBoxTensor.from_vector(x_dicts[-1]['classes']).Z)
    # %%
    return model, boxes, last_epoch, weights


if __name__ == "__main__":

#     box_regs = [0, 1e-4, 1]
#     neg_weights = [1e-2, 1e-1, 1]
#     scales = [True, False]
#     gnns = [[2*2], [16, 2*2]]

#     for br, neg, scale, g in tqdm(product(box_regs, neg_weights, scales, gnns)):
#         BOX_REGULARIZATION = br
#         NEG_WEIGHT = neg
#         SCALE_LOSSES = scale
#         GNN_CHANNELS = g

#         print(f"""

# GNN_CHANNELS: {GNN_CHANNELS}
# LR: {LR}
# LR_DECAY: {LR_DECAY}
# EPOCHS: {EPOCHS}
# LOSS_TYPE: {LOSS_TYPE}
# REGULARIZATION: {REGULARIZATION}
# BOX_REGULARIZATION: {BOX_REGULARIZATION}
# NEG_WEIGHT: {NEG_WEIGHT}
# SCALE_LOSSES: {SCALE_LOSSES}"""
#         )

    # %%
    BASE = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(BASE, 'datasets/box_graph.pkl'), 'rb') as fi:
        data = pickle.load(fi)
    graph = data['graph'].to(device)
    rev_class_dict = data['rev_class_dict']
    rev_rel_dict = data['rev_rel_dict']
    pprint(graph.edge_types)
    gci = data['gci']
    gci = {k: {kk: vv.to(device) for kk, vv in v.items()}
        for k, v in gci.items()}
    # graph['classes'].node_id = torch.arange(len(graph['classes'].x))
    # %%

    # Create an output directory if it doesn't exist
    # with current date and time in the directory name
    import datetime
    now = datetime.datetime.now()
    output_dir = os.path.join(
        BASE, 'trained_models', f"{LOSS_TYPE}_loss_{now.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    # Save the hyperparameters and training information to a text file
    with open(os.path.join(output_dir, 'training_info.txt'), 'w') as fo:
        fo.write(f"Ontology source: {data['source_ontology']}\n")
        fo.write(f"Loss type: {LOSS_TYPE}\n")
        fo.write(f"Epochs: {EPOCHS}\n")
        fo.write(f"Learning rate: {LR}\n")
        fo.write(f"Learning rate decay: {LR_DECAY}\n")
        fo.write(f"Regularization: {REGULARIZATION}\n")
        fo.write(f"Box regularization: {BOX_REGULARIZATION}\n")
        fo.write(f"Negative weight: {NEG_WEIGHT}\n")
        fo.write(f"Losses scaled (pos. and neg.): {SCALE_LOSSES}\n")
        fo.write(f"Model channels: {GNN_CHANNELS}\n")
        fo.write(f"Training started at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")

    model, boxes, stop_epoch, weights = train_boxes_OntologyGNN(
        graph, gci, save_weights=True,
        # gnn_channels=g,
        # box_regularization=br,
        # neg_weight=neg,
        # scale_losses=scale,
    )

    model.to('cpu')

    with open(os.path.join(output_dir, 'training_info.txt'), 'a') as fo:
        fo.write(f"Model: {model.__class__.__name__}\n")
        fo.write(
            f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
        fo.write(
            f"Epochs completed: {stop_epoch + 1} of planned {EPOCHS} epochs")

    # Save the model to a file
    print(f"Saving model to {output_dir}")

    # Save the model and graph to a pickle file
    with open(os.path.join(output_dir, 'box_model.pkl'), 'wb') as fo:
        pickle.dump(model, fo)
    # %%

    # Get boxes and losses
    boxes_epochs = np.stack([b[0] for b in boxes])
    be_dict = {i: boxes_epochs[:, i, :, :]
            for i in range(boxes_epochs.shape[1])}
    # print([[t for t in b[1:]] for b in boxes])
    losses = np.array([b[1:] for b in boxes])

    true_classes = set(gci["gci0"]["classes"][:, 1].detach().numpy())
    pprint({k: v for k, v in rev_class_dict.items() if k in true_classes})

    plot_classes = set(c for c in gci["gci0"]["classes"][:, 1].detach().numpy() if (lambda k: any([
        k.startswith("http://purl.obolibrary.org/obo/APO"),
        k.startswith("http://purl.obolibrary.org/obo/CHEBI"),
        k == 'http://hypo.project-genesis.io#organismState'
    ]))(rev_class_dict[c]))

    animate_boxes_with_blitting(be_dict, losses, save=True, fp=os.path.join(
        output_dir, 'training.mp4'),
        box_filter=lambda k: k in true_classes,
        # box_filter=lambda k: k in plot_classes,
        # box_filter=lambda k: True,
        box_filter_type='omit',
        box_labels=rev_class_dict,
        box_label_filter=lambda k: k in true_classes
        # box_label_filter=lambda k: k in plot_classes
    )
