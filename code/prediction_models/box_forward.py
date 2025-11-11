# %%
import os, pickle
import torch
from box_embeddings.parameterizations import MinDeltaBoxTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(BASE, 'datasets/box_graph.pkl'), 'rb') as fi:
    data = pickle.load(fi)
graph = data['graph'].to(device)
gci = data['gci']
gci = {k: {kk: vv.to(device) for kk, vv in v.items()} for k,v in gci.items()}

with open('box_model.pkl', 'rb') as fi:
    model = pickle.load(fi)
# %%

x_dict = model(graph.x_dict, graph.edge_index_dict)
# To return all embeddings with deeper GNN: (will return list of dictionaries with embeddings for nodes, one dict per layer in GNN) 
# x_dicts = model(graph.x_dict, graph.edge_index_dict, return_embs=True)
# %%
boxes = MinDeltaBoxTensor.from_vector(x_dict['classes'])
print(boxes.Z)
# %%
boxes = MinDeltaBoxTensor.from_vector(graph.x_dict['classes'])