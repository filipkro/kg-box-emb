# %%
import pickle, os, time
import torch as th
from sklearn.metrics import r2_score
from torch.cuda import is_available
from torch.nn.functional import mse_loss

from simple_gnn import Regressor
from train_loop import train_final_model, continue_final_training

from parameters import (EPOCHS, LR, GNN_CHANNELS, NN_CHANNELS, REGULARIZATION,
                        TRAIN_EMBEDDING_EPOCH, TRAIN_GENES, BOX_WEIGHT,
                        DATASET, BOX_EMBEDDINGS)

from argparse import ArgumentParser
# %%
parser = ArgumentParser()
parser.add_argument('--model_file', default='')
args = parser.parse_args()
# %%
if is_available():
    device = 'cuda'
else:
    device = 'cpu'
    # EPOCHS = 10
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

print(data['reactions','catalyzedByGene', 'genes'])
print(data['reactions','catalyzedByGene', 'genes'].edge_index.shape)

print(data['genes','rev_catalyzedByGene', 'reactions'])
print(data['genes','rev_catalyzedByGene', 'reactions'].edge_index.shape)
data.to(device)

print(f'Dataset: {DATASET}')
print(f"gnn channels: {GNN_CHANNELS}")
print(f"nn channels: {NN_CHANNELS}")
# %%
print(f"epochs: {EPOCHS}")
print(f"lr: {LR}")

print(f'start training embeddings: {TRAIN_EMBEDDING_EPOCH}')
print(f'train gene embeddings: {TRAIN_GENES}')
print(f'use box embeddings: {BOX_EMBEDDINGS}')
print(f'box weight: {BOX_WEIGHT}')
print(f'regularization: {REGULARIZATION}')

if not args.model_file:
    model_kwargs = {'gnn_channels': GNN_CHANNELS, 'nn_channels': NN_CHANNELS,
                    'meta_data': data.metadata(), 'embeddings': data.x_dict}
    metric, model = train_final_model(model_type=Regressor,
                                      model_kwargs=model_kwargs, data=data,
                                      epochs=EPOCHS, loss_function=mse_loss,
                                      metric=r2_score, device=device, lr=LR)
else:
    with open(os.path.join(BASE, args.model_file), 'rb') as fi:
        model = pickle.load(fi)['model']

    epochs = 400
    print(model)
    print(f"continue training for {epochs} epochs")
    print(f"epochs: {epochs}")
    metric, model = continue_final_training(model, data, epochs,
                                            loss_function=mse_loss,
                                            metric=r2_score, device=device,
                                            lr=LR)
# %%
model.to('cpu')
file_name = time.strftime("%Y%m%d-%H%M%S") + '-reg.pkl'
with open(os.path.join(BASE, 'trained_gnns', file_name), 'wb') as fo:
    pickle.dump({'model': model, 'metrics': metric}, fo)
# %%
