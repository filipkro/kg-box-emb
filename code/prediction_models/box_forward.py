# %%
from box_embeddings.parameterizations import MinDeltaBoxTensor
import torch
import pickle
import os
import sys
from matplotlib import pyplot as plt

sys.path.append(os.path.join("/", "workspaces", "kg-box-emb", "code", "presentation"))
from boxplot2d import plot_box_2d

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_graph_from_pickle():
    # %%
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(BASE, "datasets/box_graph.pkl"), "rb") as fi:
        data = pickle.load(fi)
    graph = data["graph"].to(device)
    gci = data["gci"]
    gci = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in gci.items()}

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "box_model.pkl"), "rb"
    ) as fi:
        model = pickle.load(fi)

    return model, graph


# %%


def get_boxes_from_model_and_graph(model, graph, return_embs=False):
    x_dict = model(graph, return_embs=return_embs)
    # To return all embeddings with deeper GNN: (will return list of dictionaries with embeddings for nodes, one dict per layer in GNN)
    # x_dicts = model(graph, return_embs=True)
    # %%
    return MinDeltaBoxTensor.from_vector(x_dict["classes"])


def get_initial_boxes_from_model(model, graph):
    x_dict = model(graph, return_embs=True)[0]
    # To return all embeddings with deeper GNN: (will return list of dictionaries with embeddings for nodes, one dict per layer in GNN)
    # x_dicts = model(graph, return_embs=True)
    # %%
    return MinDeltaBoxTensor.from_vector(x_dict["classes"])


if __name__ == "__main__":
    model, graph = load_model_and_graph_from_pickle()

    boxes = get_boxes_from_model_and_graph(model, graph)
    print(boxes.Z)
    # %%

    # Plotting the boxes
    fig, ax = plt.subplots()
    for bb in boxes:
        plot_box_2d(bb.z.detach().numpy(), bb.Z.detach().numpy(), fig=fig, ax=ax)

    fig.tight_layout()
    fig.show()
