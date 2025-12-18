# Import
import os
import sys
import pickle
import torch
import datetime
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy
from random import sample, choice
from torch_geometric import seed_everything
from sklearn.model_selection import train_test_split
from box_embeddings.parameterizations import MinDeltaBoxTensor
from train_boxes import train_boxes_OntologyGNN, box_loss

# Constants
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GNN_CHANNELS = [8, 8, 8, 8]
# GNN_CHANNELS = [2 * 2]
LR = 5e-1
LR_DECAY = 0.001
REGULARIZATION = 0
BOX_REGULARIZATION = 0.000
EPOCHS = 501
NEG_WEIGHT = 0.5
NEG_RANDOM_WEIGHT = 1
LOSS_TYPE = "inclusion"
SCALE_LOSSES = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(42)
torch.manual_seed(42)
# Enable detect anomaly mode
torch.autograd.set_detect_anomaly(True)


# Define functions
def load_graph():
    with open(os.path.join(BASE, "datasets/box_graph.pkl"), "rb") as fi:
        data = pickle.load(fi)
    graph = data["graph"].to(DEVICE)
    rev_class_dict = data["rev_class_dict"]
    rev_rel_dict = data["rev_rel_dict"]
    gci = data["gci"]
    gci = {k: {kk: vv.to(DEVICE) for kk, vv in v.items()} for k, v in gci.items()}

    return graph, gci, data, rev_class_dict, rev_rel_dict


def graph_train_test_split(G, ratio=0.7):
    G_train, G_test = deepcopy(G), deepcopy(G)
    for edge_type, edge_details in G.edge_items():
        train, test = train_test_split(
            edge_details["edge_index"].T, random_state=42, test_size=1 - ratio
        )
        G_train[edge_type]["edge_index"] = train.T
        G_test[edge_type]["edge_index"] = test.T
        assert (
            G_train[edge_type]["edge_index"].shape[0]
            == edge_details["edge_index"].shape[0]
        )
    return G_train, G_test


def train_and_save_model(
    G, gci, true_classes, output_dir
):  # -> tuple[Any, Any, Any, Any]:
    # Return model to device
    model, boxes, stop_epoch, weights = train_boxes_OntologyGNN(
        G,
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
        neg_classes_to_skip=len(true_classes) + 2,
        # gnn_channels=g,
        # box_regularization=br,
        # neg_weight=neg,
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

    return model, boxes, stop_epoch, weights


def box_embeddings_from_model(model, G, box=MinDeltaBoxTensor, final_only=True):
    if final_only:
        x_dicts = [model(G, return_embs=False)]
        return box.from_vector(x_dicts[0]["classes"]), x_dicts
    else:
        # x_dicts = model(G, return_embs=True)
        # ^^^ List of dictionaries, one for each layer (inc. initial embeddings)
        raise NotImplementedError


def dist_inclusion(sub_c, sub_o, sup_c, sup_o, gamma=0.0):
    return (
        torch.relu(torch.abs(sub_c - sup_c) + sub_o - sup_o - gamma).norm(dim=-1).sum()
    )


def embedding_distance(
    emb1,
    emb2,
    gamma=0.0,
):
    emb1_c, emb1_o = emb1.centre, emb1.Z - emb1.centre
    emb2_c, emb2_o = emb2.centre, emb2.Z - emb2.centre
    return 0.5 * (
        dist_inclusion(emb1_c, emb1_o, emb2_c, emb2_o, gamma)
        + dist_inclusion(emb2_c, emb2_o, emb1_c, emb1_o, gamma)
    )


def get_superclass(subclass, gci, person=False):
    if person:
        return 2
    else:
        cands = (
            gci["gci0"]["classes"][gci["gci0"]["classes"][:, 0] == subclass][:, 1]
            .detach()
            .tolist()
        )
        return next(filter(lambda c: c != 2, cands))


def get_random_subclass(superclass, gci):
    return choice(
        gci["gci0"]["classes"][gci["gci0"]["classes"][:, 1] == superclass][:, 0]
        .detach()
        .numpy()
    )


if __name__ == "__main__":
    # Load graph and find "true classes"
    G, gci, data, rev_class_dict, rev_rel_dict = load_graph()
    true_classes = set(gci["gci0"]["classes"][:, 1].detach().numpy())

    # Create an output directory if it doesn't exist
    # with current date and time in the directory name
    now = datetime.datetime.now()
    output_dir = os.path.join(
        BASE,
        "trained_models",
        "link_evaluation",
        f"{LOSS_TYPE}_loss_{now.strftime('%Y%m%d_%H%M%S')}",
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
        fo.write(f"Losses scaled (pos. and neg.): {SCALE_LOSSES}\n")
        fo.write(f"Model channels: {GNN_CHANNELS}\n")
        fo.write(f"Training started at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")

    #################################################################
    # Pseudocode
    # 1. Split graph into train, G:=(V,E) and test, G':=(V,E')
    # 2. Train embedding parameters using G
    # 3. Calculate embeddings B from G using trained parameters
    # 4. distances = {}
    # 5. For each edge, e, in E':
    # 6.   Create graph G*:=(V,E u {e})
    # 7.   calculate embeddings B* from G* using trained parameters
    # 8.   Set distance dist:=0
    # 9.   For v in V:
    # 10.    b:=B(v), b*=B*(v)
    # 11.    dist:= dist + 〈b,b*〉
    # 12.  distances[e]:= dist
    #################################################################

    # Pseudocode  (filled in)
    # 1. Split graph into train, G:=(V,E) and test, G':=(V,E')
    G_train, G_test = graph_train_test_split(G)

    # 2. Train embedding parameters using G
    model, boxes, stop_epoch, weights = train_and_save_model(
        G_train, gci, true_classes, output_dir
    )
    # 3. Calculate embeddings B from G using trained parameters
    B, x_dicts_orig = box_embeddings_from_model(model, G_train, box=MinDeltaBoxTensor)
    B_full, x_dicts_full = box_embeddings_from_model(model, G, box=MinDeltaBoxTensor)
    print(f"Total distance from G_train to G: {embedding_distance(B, B_full)}")
    # 4. distances = {}
    distances = {}
    random_distances = {}
    constrained_random_distances = {}
    i = 0
    # 5. For each edge, e, in E':
    for edge_type, edge_details in tqdm(G_test.edge_items()):
        if edge_type[1].startswith("rev_"):
            continue
        for source, target in tqdm(edge_details["edge_index"].T):
            # 6.   Create graph G*:=(V,E u {e})
            G_star = deepcopy(G_train)
            # TODO Also add rev_* edge
            G_star[edge_type]["edge_index"] = torch.cat(
                [G_star[edge_type]["edge_index"], torch.tensor([[source, target]]).T],
                dim=1,
            )
            # 6. Also random
            G_rand = deepcopy(G_train)
            rs, rt = sample(range(len(rev_class_dict)), k=2)
            G_rand[edge_type]["edge_index"] = torch.cat(
                [G_rand[edge_type]["edge_index"], torch.tensor([[rs, rt]]).T],
                dim=1,
            )
            # 6. Also constrained random
            G_crand = deepcopy(G_train)
            crs = get_random_subclass(get_superclass(source, gci), gci)
            crt = get_random_subclass(
                get_superclass(
                    target,
                    gci,
                    person=edge_type[1] in ["parentOf", "fatherOf", "motherOf"],
                ),
                gci,
            )
            G_crand[edge_type]["edge_index"] = torch.cat(
                [G_crand[edge_type]["edge_index"], torch.tensor([[crs, crt]]).T],
                dim=1,
            )
            # 7.   calculate embeddings B* from G* using trained parameters
            B_star, x_dicts_star = box_embeddings_from_model(
                model, G_star, box=MinDeltaBoxTensor
            )
            B_rand, x_dicts_rand = box_embeddings_from_model(
                model, G_rand, box=MinDeltaBoxTensor
            )
            B_crand, x_dicts_crand = box_embeddings_from_model(
                model, G_crand, box=MinDeltaBoxTensor
            )
            # 8.   Set distance dist:=0
            # 9.   For v in V:
            # 10.    b:=B(v), b*=B*(v)
            # 11.    dist:= dist + 〈b,b*〉
            # 12.  distances[e]:= dist
            distances[(edge_type, source.detach().item(), target.detach().item())] = (
                dict(
                    zip(
                        ["distance", "pos_loss", "neg_loss"],
                        (
                            embedding_distance(B, B_star).detach().item(),
                            *[
                                l.detach().item()
                                for l in box_loss(
                                    x_dicts_star,
                                    gci["gci0"],
                                    loss_type=LOSS_TYPE,
                                    neg=True,
                                    neg_data=gci["gci1_bot"],
                                    neg_random_weight=0,
                                )
                            ],
                        ),
                    )
                )
            )
            # print(
            #     f"From test: {(edge_type, source, target)}: {distances[(edge_type, source, target)]}"
            # )
            random_distances[(edge_type, rs, rt)] = dict(
                zip(
                    ["distance", "pos_loss", "neg_loss"],
                    (
                        embedding_distance(B, B_rand).detach().item(),
                        *[
                            l.detach().item()
                            for l in box_loss(
                                x_dicts_rand,
                                gci["gci0"],
                                loss_type=LOSS_TYPE,
                                neg=True,
                                neg_data=gci["gci1_bot"],
                                neg_random_weight=0,
                            )
                        ],
                    ),
                )
            )
            # print(
            #     f"Random:    {(edge_type, rs, rt)}: {random_distances[(edge_type, rs, rt)]}"
            # )
            constrained_random_distances[(edge_type, crs, crt)] = dict(
                zip(
                    ["distance", "pos_loss", "neg_loss"],
                    (
                        embedding_distance(B, B_crand).detach().item(),
                        *[
                            l.detach().item()
                            for l in box_loss(
                                x_dicts_crand,
                                gci["gci0"],
                                loss_type=LOSS_TYPE,
                                neg=True,
                                neg_data=gci["gci1_bot"],
                                neg_random_weight=0,
                            )
                        ],
                    ),
                )
            )
            # print(
            #     f"Random:    {(edge_type, rs, rt)}: {random_distances[(edge_type, rs, rt)]}"
            # )
            i += 1
            # print(
            #     f"Mean test distance ({i:>3}): {np.mean(list(distances.values())):.5f};  Mean random distance ({i:>3}): {np.mean(list(random_distances.values())):.5f}; Mean constr. random distance ({i:>3}): {np.mean(list(constrained_random_distances.values())):.5f}",
            #     end="\n",
            # )

        with open(os.path.join(output_dir, "link_eval_data.pkl"), "wb") as fo:
            pickle.dump(
                {
                    "distances": distances,
                    "random_distances": random_distances,
                    "constrained_random_distances": constrained_random_distances,
                },
                fo,
            )
