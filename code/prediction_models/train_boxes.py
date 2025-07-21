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

from box_forward import get_boxes_from_model_and_graph
sys.path.append(os.path.join("/", "workspaces",
                "kg-box-emb", "code", "presentation"))
from boxplot2d import plot_box_2d, animate_boxes


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)
torch.manual_seed(42)
# %%

# %%
GNN_CHANNELS = [2*2]
LR = 1e-2
REGULARIZATION = 0
BOX_REGULARIZATION = 1e-1
EPOCHS = 300
NEG_WEIGHT = 0.5*1e0


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
    epochs=EPOCHS,
    regularization=REGULARIZATION,
    box_regularization=BOX_REGULARIZATION,
    neg_weight=NEG_WEIGHT,
):

    # model = HeteroGNNGAT(GNN_CHANNELS, graph.edge_types, graph.x_dict)
    # model = HeteroGNNSAGE(GNN_CHANNELS, graph.edge_types, graph.x_dict)
    model = OntologyGNN(GNN_CHANNELS, graph.edge_types, graph.x_dict)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=REGULARIZATION)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # %%
    model.requires_grad_(True)

    boxes = []

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # x_dicts = model(graph.x_dict, graph.edge_index_dict, return_embs=True)
        x_dicts = [model(graph, return_embs=False)]

        loss_type = 'inclusion'
        pos_loss, neg_loss = box_loss(
            x_dicts, gci['gci0'], loss_type=loss_type, neg_data=gci['gci1_bot'], neg=True)
        reg_loss = small_box_penalty(x_dicts)
        loss = pos_loss + NEG_WEIGHT * neg_loss + BOX_REGULARIZATION * reg_loss
        # loss = neg_loss
        loss.backward()
        optimizer.step()
        total_loss = loss.detach().item()
        pos_ratio = torch.exp(-pos_loss / len(gci['gci0']['classes']))
        neg_ratio = 1 - \
            torch.exp(-neg_loss /
                      (3*len(gci['gci0']['classes']) + len(gci['gci1_bot']['classes'])))

        if loss_type == 'distance':
            print(
                f"Epoch: {epoch}, total loss: {total_loss:.4f}, pos loss: {pos_ratio:.6f}, neg loss: {neg_ratio:.6f}, reg: {reg_loss:.3f}")
        else:
            print(
                f"Epoch: {epoch}, total loss: {total_loss:.4f}, pos ratio: {pos_ratio:.6f}, neg ratio: {neg_ratio:.6f}, reg: {reg_loss:.8f}")
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
    # print(MinDeltaBoxTensor.from_vector(x_dicts[-1]['classes']).Z)
    # %%
    return model, boxes


if __name__ == "__main__":

    # %%
    BASE = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(BASE, 'datasets/box_graph.pkl'), 'rb') as fi:
        data = pickle.load(fi)
    graph = data['graph'].to(device)
    pprint(graph.edge_types)
    gci = data['gci']
    gci = {k: {kk: vv.to(device) for kk, vv in v.items()}
           for k, v in gci.items()}
    # graph['classes'].node_id = torch.arange(len(graph['classes'].x))
    # %%

    model, boxes = train_boxes_OntologyGNN(graph, gci)

    model.to('cpu')
    with open('box_model.pkl', 'wb') as fo:
        pickle.dump(model, fo)
    # %%

    boxes_epochs = np.stack([b[0] for b in boxes])
    be_dict = {i: boxes_epochs[:, i, :, :] for i in range(boxes_epochs.shape[1])}
    # print([[t for t in b[1:]] for b in boxes])
    losses = np.array([b[1:] for b in boxes])

    animate_boxes(be_dict, losses, save=True)
