# %%
import pickle, os, time, sys
import torch as th

from sklearn.metrics import r2_score
from torch.cuda import is_available
from torch.nn.functional import mse_loss

from model import Regressor
from train_loop import cross_val

from parameters import (EPOCHS, LR, GNN_CHANNELS, NN_CHANNELS, REGULARIZATION,
                        TRAIN_EMBEDDING_EPOCH, TRAIN_GENES, BOX_WEIGHT,
                        DATASET, BOX_EMBEDDINGS, ONLY_GENE_BOXES, SPLIT,
                        SEMANTIC_WEIGHT, MIN_NBR_EDGES, NUM_BATCHES,
                        SEMANTIC_MEASURE, DROP_OUT, NEG_WEIGHT,
                        INTER_TYPE, VOL_TYPE, RANDOM_INIT_EMBS, EMBEDDING_DIMS, ONLY_BILINEAR, GENE_COMBINE, CUSTOM_OGGNN, TRANSFORMER)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embeddings')))
# %%
if is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)
assert device == 'cuda'
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(BASE, f'datasets/split_datasets/{DATASET}.pkl'),
          'rb') as fi:
    data = pickle.load(fi).contiguous()

# eb = data['mat_ent', 'encodedBy', 'genes']['edge_index']
# cb = data['reactions', 'catalyzedBy', 'mat_ent']['edge_index']
# cbg = []
# for r in cb.T:
#     if r[1] in eb[0,:]:
#         p = [r[0], eb[1,eb[0,:] == r[1]]]
#         cbg.append(p)
# cbgt = th.tensor(cbg).T
# data['reactions','catalyzedByGene', 'genes'].edge_index = cbgt
# data['genes','rev_catalyzedByGene', 'reactions'].edge_index \
#                     = cbgt.flip(dims=(0,))

data.to(device)

print(f'Dataset: {DATASET}')
print(f"gnn channels: {GNN_CHANNELS}")
print(f"nn channels: {NN_CHANNELS}")
print(f"splitting: {SPLIT}")
print(f"num batches: {NUM_BATCHES}")
# %%
print(f"epochs: {EPOCHS}")
print(f"lr: {LR}")

print(f'start training embeddings: {TRAIN_EMBEDDING_EPOCH}')
print(f'only gene boxes: {ONLY_GENE_BOXES}')
print(f'train gene embeddings: {TRAIN_GENES}')
print(f'use box embeddings: {BOX_EMBEDDINGS}')
print(f'box weight: {BOX_WEIGHT}')
print(f'regularization: {REGULARIZATION}')

print(f"semantic weight: {SEMANTIC_WEIGHT}")
print(f"min nbr edges: {MIN_NBR_EDGES}")

print(f"semantic measure: {SEMANTIC_MEASURE}")
print(f"dropout: {DROP_OUT}")
print(f"negative weight: {NEG_WEIGHT}")

print(f"interaction type: {INTER_TYPE}")
print(f"volume type: {VOL_TYPE}")

print(f"random init embs: {RANDOM_INIT_EMBS}")
print(f"random embedding dims: {EMBEDDING_DIMS}")

print(f"only bilinear: {ONLY_BILINEAR}")

print(f"gene combination method: {GENE_COMBINE}")

print(f"Custom OGGNN: {CUSTOM_OGGNN}")
print(f"Transformer base GNN: {TRANSFORMER}")

model_kwargs = {'gnn_channels': GNN_CHANNELS, 'nn_channels': NN_CHANNELS,
                'meta_data': data.metadata(), 'embeddings': data.x_dict}
if True:
    gci0 = {}
    for n in data.node_types:
        if n in ['genes', 'root']:
            continue
        with open(os.path.join(BASE, 'datasets/split_datasets/'
                                f'collected_{n}.pkl'), 'rb') as fi:
            gci0[n] = \
                  pickle.load(fi).training_datasets.gci0_dataset.data.to(device)
else:
    gci0 = None

for k in data.node_types:
    num_nodes = data[k].x.shape[0]
    self_loops = th.ones(2,num_nodes, dtype=th.int64) * th.arange(num_nodes, dtype=th.int64)
    #data[k, 'self', k].edge_index = self_loops
# %%
metrics, models, data_splits = cross_val(model_type=Regressor,
                                         model_kwargs=model_kwargs,
                                         data=data, epochs=EPOCHS,
                                         loss_function=mse_loss,
                                         metric=r2_score, device=device, lr=LR,
                                         gci0_data=gci0, folds=10, split=SPLIT, num_batches=NUM_BATCHES)


# %%
for m in models:
    m.to('cpu')
# file_name = time.strftime("%Y%m%d-%H%M%S") + '-DummyReg.pkl'
file_name = time.strftime("%Y%m%d-%H%M%S") + '-reg.pkl'
with open(os.path.join(BASE, 'large_files', file_name), 'wb') as fo:
    pickle.dump({'models': models, 'metrics': metrics}, fo)
    #pickle.dump({'models': models, 'metrics': metrics, 'data': data_splits}, fo)
