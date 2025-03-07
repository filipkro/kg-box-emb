# %%
import pickle, os, time
import torch as th

from sklearn.metrics import r2_score
from torch.cuda import is_available
from torch.nn.functional import mse_loss

from simple_gnn import Regressor, DummyModel
from train_loop import cross_val

from parameters import (EPOCHS, LR, GNN_CHANNELS, NN_CHANNELS, REGULARIZATION,
                        TRAIN_EMBEDDING_EPOCH, TRAIN_GENES, BOX_WEIGHT,
                        DATASET, BOX_EMBEDDINGS, ONLY_GENE_BOXES, SPLIT)
# %%
if is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE, f'datasets/split_datasets/{DATASET}.pkl'),
          'rb') as fi:
    data = pickle.load(fi).contiguous()

eb = data['mat_ent', 'encodedBy', 'genes']['edge_index']
cb = data['reactions', 'catalyzedBy', 'mat_ent']['edge_index']
cbg = []
for r in cb.T:
    if r[1] in eb[0,:]:
        p = [r[0], eb[1,eb[0,:] == r[1]]]
        cbg.append(p)
cbgt = th.tensor(cbg).T
data['reactions','catalyzedByGene', 'genes'].edge_index = cbgt
data['genes','rev_catalyzedByGene', 'reactions'].edge_index \
                    = cbgt.flip(dims=(0,))

data.to(device)

print(f'Dataset: {DATASET}')
print(f"gnn channels: {GNN_CHANNELS}")
print(f"nn channels: {NN_CHANNELS}")
print(f"splitting: {SPLIT}")
# %%
print(f"epochs: {EPOCHS}")
print(f"lr: {LR}")

print(f'start training embeddings: {TRAIN_EMBEDDING_EPOCH}')
print(f'only gene boxes: {ONLY_GENE_BOXES}')
print(f'train gene embeddings: {TRAIN_GENES}')
print(f'use box embeddings: {BOX_EMBEDDINGS}')
print(f'box weight: {BOX_WEIGHT}')
print(f'regularization: {REGULARIZATION}')

model_kwargs = {'gnn_channels': GNN_CHANNELS, 'nn_channels': NN_CHANNELS,
                'meta_data': data.metadata(), 'embeddings': data.x_dict}
if True:
    gci0 = {}
    for n in data.node_types:
        with open(os.path.join(BASE, 'datasets/split_datasets/'
                                f'collected_{n}.pkl'), 'rb') as fi:
            gci0[n] = \
                  pickle.load(fi).training_datasets.gci0_dataset.data.to(device)
else:
    gci0 = None
metrics, models, data_splits = cross_val(model_type=Regressor,
                                         model_kwargs=model_kwargs,
                                         data=data, epochs=EPOCHS,
                                         loss_function=mse_loss,
                                         metric=r2_score, device=device, lr=LR,
                                         gci0_data=gci0, folds=10, split=SPLIT)
#metrics, models = cross_val(model_type=DummyModel, model_kwargs=model_kwargs,
#                            data=data, epochs=EPOCHS, loss_function=mse_loss,
#                            metric=r2_score, device=device, lr=LR,
#                            gci0_data=gci0, folds=10)


# %%
for m in models:
    m.to('cpu')
# file_name = time.strftime("%Y%m%d-%H%M%S") + '-DummyReg.pkl'
file_name = time.strftime("%Y%m%d-%H%M%S") + '-reg.pkl'
with open(os.path.join(BASE, 'trained_gnns', file_name), 'wb') as fo:
    pickle.dump({'models': models, 'metrics': metrics}, fo)
    #pickle.dump({'models': models, 'metrics': metrics, 'data': data_splits}, fo)