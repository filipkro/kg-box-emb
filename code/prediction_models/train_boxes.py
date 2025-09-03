# %%
from memory_profiler import profile
import numpy as np
from box_embeddings.modules.volume import BesselApproxVolume, HardVolume
from box_embeddings.modules.intersection import GumbelIntersection, HardIntersection
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
from matplotlib import pyplot as plt

import rdflib
from rdflib.namespace import RDF

import logging
import traceback
from time import time

from itertools import product
from tqdm.auto import tqdm

from box_forward import get_boxes_from_model_and_graph, get_initial_boxes_from_model

sys.path.append(os.path.join("/", "workspaces", "kg-box-emb", "code", "presentation"))
from boxplot2d import (
    plot_box_2d,
    plot_min_delta_boxes_2d_matplotlib,
    animate_boxes,
    animate_boxes_with_blitting,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(42)
torch.manual_seed(42)
# Enable detect anomaly mode
torch.autograd.set_detect_anomaly(True)
# %%

# %%
GNN_CHANNELS = [2 * 2]
LR = 0.05
LR_DECAY = 0.001
REGULARIZATION = 0
# BOX_REGULARIZATION = 0
# BOX_REGULARIZATION = 1e-7
# BOX_REGULARIZATION = 1e-5
BOX_REGULARIZATION = 0.001
# BOX_REGULARIZATION = 1000
EPOCHS = 501
NEG_WEIGHT = 0.5
NEG_RANDOM_WEIGHT = 0.1
LOSS_TYPE = "distance"
SCALE_LOSSES = False

# Outputs
PLOT_LAST_PRE_GNN = False
PLOT_LAST = True
ANIMATE = False


class TrainingLogger:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.training_logs = []

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        logging.info(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time() - self.epoch_start_time
        logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs["epoch_time"] = elapsed_time  # Add epoch time to logs
        self.training_logs.append(logs)  # Collect training logs


def box_loss(
    embeddings,
    gci0,
    loss_type="distance",
    box_transform="mindelta",
    inter="gumbel",
    inter_temp=0.1,
    vol="bessel",
    vol_temp=0.1,
    gamma=0.0,
    neg_data=None,
    neg=False,
    neg_random_weight=0.0,
    neg_classes_to_skip=0,
    **kwargs,
):
    match box_transform:
        case "mindelta":
            box = MinDeltaBoxTensor
        case "sigmoid":
            box = SigmoidBoxTensor
        case _:
            raise NotImplementedError()
    if loss_type == "inclusion":
        return box_loss_inclusion(
            embeddings,
            gci0,
            box=box,
            inter=inter,
            inter_temp=inter_temp,
            vol=vol,
            vol_temp=vol_temp,
            neg_data=neg_data,
            neg=neg,
            neg_random_weight=neg_random_weight,
            neg_classes_to_skip=neg_classes_to_skip,
        )
    if loss_type == "distance":
        return box_loss_distance(
            embeddings,
            gci0,
            box=box,
            gamma=gamma,
            neg_data=neg_data,
            neg=neg,
            neg_random_weight=neg_random_weight,
            neg_classes_to_skip=neg_classes_to_skip,
        )
    pass


def box_loss_inclusion(
    embeddings,
    gci0,
    box=MinDeltaBoxTensor,
    inter="gumbel",
    inter_temp=0.1,
    vol="bessel",
    vol_temp=0.1,
    neg_data=None,
    neg=False,
    neg_random_weight=0.0,
    neg_classes_to_skip=0,
    **kwargs,
):
    def neg_loss_func(A, B, volume, intersect, verbose=False):
        if verbose:
            print(
                f"\tLog arg: {(1 - (volume(intersect(A, B)) / torch.minimum(volume(A), volume(B)))).sort()}"
            )
            print(
                f"\tLog: {(1 - (volume(intersect(A, B)) / torch.minimum(volume(A), volume(B)))).log().sort()}"
            )
            print(
                f"\tClamped: {(1 - (volume(intersect(A, B)) / torch.minimum(volume(A), volume(B)))).clamp(min=1e-9, max=1).sort()}"
            )
            print(
                f"\tClamped log: {(1 - (volume(intersect(A, B)) / torch.minimum(volume(A), volume(B)))).clamp(min=1e-9, max=1).log().sort()}"
            )

            print("A:", volume(A).sort())
            print("B:", volume(B).sort())

        # Have division by zero here, need to fix this...
        return (
            (1 - (volume(intersect(A, B)) / torch.minimum(volume(A), volume(B))))
            .clamp(min=1e-9, max=1)
            .log()
            .sum()
        )

    # if neg or neg_data:
    #     raise NotImplementedError("Negative loss not yet implemented "
    #                               "for inclusion loss")
    match inter:
        case "gumbel":
            intersect = GumbelIntersection(intersection_temperature=inter_temp)
        case "hard":
            intersect = HardIntersection()
        case _:
            raise NotImplementedError()

    match vol:
        case "bessel":
            volume = BesselApproxVolume(
                intersection_temperature=inter_temp,
                volume_temperature=vol_temp,
                log_scale=False,
            )
        case "hard":
            volume = HardVolume(log_scale=False)
        case _:
            raise NotImplementedError()

    loss = 0
    neg_loss = 0
    for x_dict in embeddings:
        for k, emb in x_dict.items():

            if k == "genes":
                continue
            box_emb = box.from_vector(emb)

            subclasses = box_emb[gci0[k][:, 0], ...]
            supclasses = box_emb[gci0[k][:, 1], ...]

            loss -= (
                (volume(intersect(subclasses, supclasses)) / volume(subclasses))
                .clamp(min=1e-9, max=1)
                .log()
                .sum()
            )
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
                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]),),
                    device=gci0[k].device,
                )
                A = box_emb[rand_classes, ...]
                neg_loss -= neg_loss_func(A, supclasses, volume, intersect)

                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]),),
                    device=gci0[k].device,
                )
                A = box_emb[rand_classes, ...]
                neg_loss -= neg_loss_func(A, subclasses, volume, intersect)

                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]), 2),
                    device=gci0[k].device,
                )
                A = box_emb[rand_classes[:, 0], ...]
                B = box_emb[rand_classes[:, 1], ...]
                neg_loss -= neg_loss_func(A, B, volume, intersect)

            if neg_data:
                A = box_emb[neg_data[k][:, 0], ...]
                B = box_emb[neg_data[k][:, 1], ...]

                neg_loss -= neg_loss_func(A, B, volume, intersect, verbose=False)
                # print(f"Neg loss -= {neg_loss_func(A, B, volume, intersect)}")

    return loss, neg_loss


def box_loss_distance(
    embeddings,
    gci0,
    box=MinDeltaBoxTensor,
    gamma=0.0,
    neg_data=None,
    neg=False,
    neg_random_weight=0.0,
    neg_classes_to_skip=0,
):

    def dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False):
        n = -1 if neg else 1
        if neg:
            return (
                torch.relu(-torch.abs(sub_c - sup_c) + sub_o + sup_o + gamma)
                .norm(dim=-1)
                .sum()
            )
        else:
            return (
                torch.relu(torch.abs(sub_c - sup_c) + sub_o - sup_o - gamma)
                .norm(dim=-1)
                .sum()
            )

    loss = 0
    neg_loss = 0
    for x_dict in embeddings:
        for k, emb in x_dict.items():
            if k == "genes":
                continue
            box_emb = box.from_vector(emb)

            subclasses = box_emb[gci0[k][:, 0], ...]
            sub_c, sub_o = subclasses.centre, subclasses.centre - subclasses.z
            supclasses = box_emb[gci0[k][:, 1], ...]
            sup_c, sup_o = supclasses.centre, supclasses.centre - supclasses.z

            loss += dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False)

            if neg:
                max_i = len(emb)

                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]),),
                    device=gci0[k].device,
                )
                nsub = box_emb[rand_classes, ...]
                nsub_c, nsub_o = nsub.centre, nsub.centre - nsub.z
                neg_loss += neg_random_weight * dist_inclusion(
                    nsub_c, nsub_o, sup_c, sup_o, neg=True
                )

                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]),),
                    device=gci0[k].device,
                )
                nsup = box_emb[rand_classes, ...]
                nsup_c, nsup_o = nsup.centre, nsup.centre - nsup.z
                neg_loss += neg_random_weight * dist_inclusion(
                    sub_c, sub_o, nsup_c, nsup_o, neg=True
                )

                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]), 2),
                    device=gci0[k].device,
                )
                nsub = box_emb[rand_classes[:, 0], ...]
                nsub_c, nsub_o = nsub.centre, nsub.centre - nsub.z
                nsup = box_emb[rand_classes[:, 1], ...]
                nsup_c, nsup_o = nsup.centre, nsup.centre - nsup.z
                neg_loss += neg_random_weight * dist_inclusion(
                    nsub_c, nsub_o, nsup_c, nsup_o, neg=True
                )

            if neg_data:
                subclasses = box_emb[neg_data[k][:, 0], ...]
                sub_c = subclasses.centre
                sub_o = subclasses.centre - subclasses.z
                supclasses = box_emb[neg_data[k][:, 1], ...]
                sup_c = supclasses.centre
                sup_o = supclasses.centre - supclasses.z

                neg_loss += dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=True)

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
            loss += torch.relu(1 / box_sizes - 1).sum()
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
    neg_random_weight=NEG_RANDOM_WEIGHT,
    scale_losses=SCALE_LOSSES,
    save_weights=False,
    neg_classes_to_skip=0,
):
    print(
        f"""

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # %%
    model.requires_grad_(True)
    model.node_embeddings.requires_grad_(True)

    boxes = []
    weights = [] if save_weights else None
    last_epoch = 0
    try:
        for epoch in range(epochs):
            optimizer.zero_grad()

            x_dicts = model(graph, return_embs=True)
            # ^^^ List of dictionaries, one for each layer (inc. initial embeddings)
            # x_dicts = [model(graph, return_embs=False)]

            pos_loss, neg_loss = box_loss(
                x_dicts,
                gci["gci0"],
                loss_type=loss_type,
                inter="gumbel",
                vol="bessel",
                neg_data=gci["gci1_bot"],
                neg=True,
                neg_random_weight=neg_random_weight,
                neg_classes_to_skip=neg_classes_to_skip,
            )
            if box_regularization > 0.0:
                reg_loss = small_box_penalty(x_dicts)
            else:
                reg_loss = torch.tensor(0.0)
            pos_loss_scaled = pos_loss / len(gci["gci0"]["classes"])
            neg_loss_scaled = neg_loss / (
                3 * len(gci["gci0"]["classes"]) + len(gci["gci1_bot"]["classes"])
            )
            pos_ratio = torch.exp(-pos_loss_scaled)
            neg_ratio = 1 - torch.exp(-neg_loss_scaled)
            if scale_losses:
                loss = (
                    pos_loss_scaled
                    + neg_weight * neg_loss_scaled
                    + box_regularization * reg_loss
                )
            else:
                loss = pos_loss + neg_weight * neg_loss + box_regularization * reg_loss
            # loss = neg_loss
            # loss = 1 - pos_ratio + neg_weight * neg_ratio + box_regularization * reg_loss
            total_loss = loss.detach().item()

            if loss_type == "distance":
                print(
                    f"Epoch: {epoch}, total loss: {total_loss:.4g}, pos loss: {pos_loss:.6g}, neg loss: {neg_loss:.6g}, reg: {reg_loss:.3g}"
                )
            else:
                print(
                    f"Epoch: {epoch}, total loss: {total_loss:.4g}, pos ratio: {pos_ratio:.6g}, neg ratio: {neg_ratio:.6g}, reg: {reg_loss:.8g}"
                )

            # Backpropagate loss gradients
            loss.backward()
            optimizer.step()
            #
            if epoch % 1 == 0:

                if loss_type == "distance":
                    boxes.append(
                        (
                            get_boxes_from_model_and_graph(model, graph)
                            .data.detach()
                            .numpy(),
                            total_loss,
                            pos_loss.detach().item(),
                            neg_loss.detach().item(),
                            reg_loss.detach().item(),
                        )
                    )
                else:
                    boxes.append(
                        (
                            get_boxes_from_model_and_graph(model, graph)
                            .data.detach()
                            .numpy(),
                            total_loss,
                            pos_ratio.detach().item(),
                            neg_ratio.detach().item(),
                            reg_loss.detach().item(),
                        )
                    )
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


def plot_boxes_mpl(
    data,
    rev_class_dict,
    plot_boxes,
    base_fp,
    fig=None,
    ax=None,
    loss_type=LOSS_TYPE,
    plot_labels=True,
):
    w_list = [t[0, :] for t in plot_boxes.values()]
    d_list = [t[1, :] for t in plot_boxes.values()]

    g = rdflib.Graph()
    g.parse(os.path.join(base_fp, data["source_ontology"]))
    rev_superclass_dict = {}
    for key, sub in rev_class_dict.items():
        q = [
            t
            for t in g.triples((rdflib.URIRef(sub), RDF.type, None))
            if t[2] != rdflib.URIRef("http://www.w3.org/2002/07/owl#NamedIndividual")
        ]
        if len(q) == 0:
            rev_superclass_dict[key] = None
        else:
            rev_superclass_dict[key] = q[0][2]

    color_dict = dict(
        zip(
            sorted(list(set(rev_superclass_dict.values()))),
            [None, "green", "blue", "purple", "red"],
        )
    )
    colors = [color_dict.get(v) for v in rev_superclass_dict.values()]
    colors = ["black" if c == "red" else c for c in colors]
    labels = [
        rev_class_dict.get(k).split("/")[-1] if plot_labels else None
        for k in plot_boxes.keys()
    ]
    if w_list[0].shape == (3,):
        fig, ax = plot_min_delta_boxes_3d_matplotlib(
            w_list,
            d_list,
            colors,
            alphas=[
                1.0 if i < 4 else 0.0 if i < 6 else 0.3 for i in range(len(colors))
            ],
            draw_labels=True,
            labels=[l if i < 4 else None for i, l in enumerate(labels)],
            linewidths=[
                2.5 if i < 4 else 0.0 if i < 6 else 0.4 for i in range(len(colors))
            ],
            color_legend={"purple": "Women", "blue": "Men", "green": "Countries"},
            title=f"Box Embeddings - {'Overlap' if loss_type == 'inclusion' else 'Distance'}",
            fig=fig,
            ax=ax,
        )
    elif w_list[0].shape == (2,):
        fig, ax = plot_min_delta_boxes_2d_matplotlib(
            w_list,
            d_list,
            colors,
            alphas=[
                1.0 if i < 4 else 0.0 if i < 6 else 0.3 for i in range(len(colors))
            ],
            draw_labels=True,
            labels=[l if i < 4 else None for i, l in enumerate(labels)],
            linewidths=[
                2.5 if i < 4 else 0.0 if i < 6 else 0.4 for i in range(len(colors))
            ],
            color_legend={"purple": "Women", "blue": "Men", "green": "Countries"},
            title=f"Box Embeddings - {'Overlap' if loss_type == 'inclusion' else 'Distance'}",
            fig=fig,
            ax=ax,
        )
    else:
        raise NotImplementedError(
            "Plots for dimensions other than 2 or 3 not implemented."
        )

    return fig, ax


if __name__ == "__main__":

    lrs = [1e-2, 1e-1, 1e0]
    # lrs = [1e-1]
    lr_decays = [0, 0.001]
    box_regs = [0, 1e-4, 1]
    # box_regs = [0]
    # neg_weights = [1e-2, 1e-1, 1]
    neg_weights = [0.1, 1, 10]
    neg_rand_weights = [0.01, 1]
    scales = [True, False]
    # scales = [False]
    # gnns = [[2 * 2], [2 * 2, 2 * 2]]
    gnns = [[2 * 2]]

    # for i, (lr, dec, br, neg, neg_rand, scale, g) in tqdm(
    #     enumerate(
    #         product(
    #             lrs, lr_decays, box_regs, neg_weights, neg_rand_weights, scales, gnns
    #         )
    #     )
    # ):
    #     if i < 67:
    #         continue
    #     LR = lr
    #     LR_DECAY = dec
    #     BOX_REGULARIZATION = br
    #     NEG_WEIGHT = neg
    #     NEG_RANDOM_WEIGHT = neg_rand
    #     SCALE_LOSSES = scale
    #     GNN_CHANNELS = g

    print(
        f"""

GNN_CHANNELS: {GNN_CHANNELS}
LR: {LR}
LR_DECAY: {LR_DECAY}
EPOCHS: {EPOCHS}
LOSS_TYPE: {LOSS_TYPE}
REGULARIZATION: {REGULARIZATION}
BOX_REGULARIZATION: {BOX_REGULARIZATION}
NEG_WEIGHT: {NEG_WEIGHT}
SCALE_LOSSES: {SCALE_LOSSES}"""
    )

    # %%
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(BASE, "datasets/box_graph.pkl"), "rb") as fi:
        data = pickle.load(fi)
    graph = data["graph"].to(device)
    rev_class_dict = data["rev_class_dict"]
    rev_rel_dict = data["rev_rel_dict"]
    pprint(graph.edge_types)
    gci = data["gci"]
    gci = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in gci.items()}
    # graph['classes'].node_id = torch.arange(len(graph['classes'].x))
    # %%
    true_classes = set(gci["gci0"]["classes"][:, 1].detach().numpy())
    pprint({k: v for k, v in rev_class_dict.items() if k in true_classes})

    # Create an output directory if it doesn't exist
    # with current date and time in the directory name
    import datetime

    now = datetime.datetime.now()
    output_dir = os.path.join(
        BASE, "trained_models", f"{LOSS_TYPE}_loss_{now.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the hyperparameters and training information to a text file
    with open(os.path.join(output_dir, "training_info.txt"), "w") as fo:
        fo.write(f"Ontology source: {data['source_ontology']}\n")
        fo.write(f"Loss type: {LOSS_TYPE}\n")
        fo.write(f"Epochs: {EPOCHS}\n")
        fo.write(f"Learning rate: {LR}\n")
        fo.write(f"Learning rate decay: {LR_DECAY}\n")
        fo.write(f"Regularization: {REGULARIZATION}\n")
        fo.write(f"Box regularization: {BOX_REGULARIZATION}\n")
        fo.write(f"Negative weight: {NEG_WEIGHT}\n")
        fo.write(f"Negative (Random) weight: {NEG_RANDOM_WEIGHT}\n")
        fo.write(f"Losses scaled (pos. and neg.): {SCALE_LOSSES}\n")
        fo.write(f"Model channels: {GNN_CHANNELS}\n")
        fo.write(f"Training started at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # try:
    model, boxes, stop_epoch, weights = train_boxes_OntologyGNN(
        graph,
        gci,
        save_weights=True,
        # neg_classes_to_skip=len(true_classes) + 2,
        # lr=lr,
        # lr_decay=dec,
        # gnn_channels=g,
        # box_regularization=br,
        # neg_weight=neg,
        # neg_random_weight=neg_rand,
        # scale_losses=scale,
    )

    model.to("cpu")

    with open(os.path.join(output_dir, "training_info.txt"), "a") as fo:
        fo.write(f"Model: {model.__class__.__name__}\n")
        fo.write(
            f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n"
        )
        fo.write(f"Epochs completed: {stop_epoch + 1} of planned {EPOCHS} epochs")

    # Save the model to a file
    print(f"Saving model to {output_dir}")

    # Save the model and graph to a pickle file
    with open(os.path.join(output_dir, "box_model.pkl"), "wb") as fo:
        pickle.dump(model, fo)
    # %%

    # Get boxes and losses
    boxes_epochs = np.stack([b[0] for b in boxes])
    be_dict = {i: boxes_epochs[:, i, :, :] for i in range(boxes_epochs.shape[1])}
    # print([[t for t in b[1:]] for b in boxes])
    losses = np.array([b[1:] for b in boxes])

    plot_classes = set(
        c
        for c in gci["gci0"]["classes"][:, 1].detach().numpy()
        if (
            lambda k: any(
                [
                    k.startswith("http://purl.obolibrary.org/obo/APO"),
                    k.startswith("http://purl.obolibrary.org/obo/CHEBI"),
                    k == "http://hypo.project-genesis.io#organismState",
                ]
            )
        )(rev_class_dict[c])
    )
    sys.stdout = sys.__stdout__

    # Plot last embeddings
    if PLOT_LAST_PRE_GNN:
        first_boxes = get_initial_boxes_from_model(model, graph).data.detach().numpy()
        plot_boxes_pre = {i: first_boxes[i, :, :] for i in range(boxes_epochs.shape[1])}
        fig_pre, ax_pre = plot_boxes_mpl(data, rev_class_dict, plot_boxes_pre, BASE)
        fig_pre.savefig(os.path.join(output_dir, "final_boxes_pre_gnn.png"), dpi=300)
        fig_pre.savefig(os.path.join(output_dir, "final_boxes_pre_gnn.pdf"))
        # plt.close("all")

    if PLOT_LAST:
        plot_boxes = {k: v[-1] for k, v in be_dict.items()}
        fig, ax = plot_boxes_mpl(data, rev_class_dict, plot_boxes, BASE)
        fig.savefig(os.path.join(output_dir, "final_boxes.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, "final_boxes.pdf"))
        # plt.close("all")

    if ANIMATE:
        animate_boxes_with_blitting(
            be_dict,
            losses,
            save=True,
            fp=os.path.join(output_dir, "training.mp4"),
            box_filter=lambda k: k in true_classes,
            # box_filter=lambda k: k in plot_classes,
            # box_filter=lambda k: True,
            box_filter_type="bold",
            box_labels=rev_class_dict,
            box_label_filter=lambda k: k in true_classes,
            # box_label_filter=lambda k: k in plot_classes
        )
        # except Exception as e:
        #     print(traceback.format_exc(), file=sys.stderr)
        # finally:
        #     f.close()
        #     sys.stdout = orig_stdout

    # plt.close("all")
