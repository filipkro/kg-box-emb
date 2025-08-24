# %%
from utils.dataset_utils import get_normalized_el_dataset, get_bots
import os
import pickle

# import rdflib
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from argparse import ArgumentParser

# Argument for the dataset path
parser = ArgumentParser()
parser.add_argument(
    "--ontology_path", type=str, default="", help="Path to the ontology file"
)
args = parser.parse_args()
ontology_path = args.ontology_path
# %%
EMBED_DIMS = 2
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
full_fp = os.path.join(BASE, ontology_path)
# split_dir = os.path.join(BASE, 'datasets/split_datasets')
LOAD_NORMALIZED_DATA = False

# %%
gci, index = get_normalized_el_dataset(
    full_fp, merge_assertions=True, bypass_classes=True
)

# %%
i2c = {v: k for k, v in index["class_index"].items()}
c2i = index["class_index"]

gci["gci1_bot"] = get_bots(gci1_bot=gci["gci1_bot"], i2c=i2c, c2i=c2i, full_fp=full_fp)

# %%
# possibly pretrain embeddings for gci0 class here..

# %%
gci2 = gci["gci2"]
rev_rel_dict = {v: k for k, v in index["property_index"].items()}
rev_class_dict = {v: k for k, v in index["class_index"].items()}
# %%
rel_data = {}
for ri in gci2[:, 1].unique():
    ri = ri.item()
    rel_data[rev_rel_dict[ri]] = gci2[gci2[:, 1] == ri][:, [0, 2]]


# %%
try:
    top_classes = [
        "http://example.org/royals/Man",
        "http://example.org/royals/Woman",
        "http://example.org/royals/Person",
        "http://example.org/royals/Country",
    ]
    top_indices = [c2i[c] for c in top_classes]
    data = HeteroData()
    x = torch.randn(len(i2c), 2 * EMBED_DIMS)
    x[top_indices, EMBED_DIMS:] = x[top_indices, EMBED_DIMS:].abs()
    data["classes"].x = x
except KeyError:
    data = HeteroData()
    data["classes"].x = torch.randn(len(i2c), 2 * EMBED_DIMS)
data["classes"].node_id = torch.arange(len(i2c))
for r, v in rel_data.items():
    r = r.split("/")[-1].split("#")[-1]
    data["classes", r, "classes"].edge_index = torch.tensor(v, dtype=torch.int64).T
data = T.ToUndirected(merge=False)(data)
# %%

with open(os.path.join(BASE, "datasets/box_graph.pkl"), "wb") as fo:
    pickle.dump(
        {
            "source_ontology": ontology_path,
            "graph": data,
            "gci": {
                "gci0": {"classes": gci["gci0"]},
                "gci1_bot": {"classes": gci["gci1_bot"]},
            },
            "rev_class_dict": rev_class_dict,
            "rev_rel_dict": rev_rel_dict,
        },
        fo,
    )
# %%
