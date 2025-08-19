# %%

from train_loop import train_loop_final_sem #train_final_model, continue_final_training


from argparse import ArgumentParser

import pickle, os, time, sys
import torch as th

from sklearn.metrics import r2_score
from torch.cuda import is_available
from torch.nn.functional import mse_loss

from model import Regressor

from parameters import (EPOCHS, LR, GNN_CHANNELS, NN_CHANNELS, REGULARIZATION,
                        TRAIN_EMBEDDING_EPOCH, TRAIN_GENES, BOX_WEIGHT,
                        DATASET, BOX_EMBEDDINGS, ONLY_GENE_BOXES, SPLIT,
                        SEMANTIC_WEIGHT, MIN_NBR_EDGES, NUM_BATCHES,
                        SEMANTIC_MEASURE, DROP_OUT, NEG_WEIGHT,
                        INTER_TYPE, VOL_TYPE)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../embeddings')))

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
#                 = cbgt.flip(dims=(0,))

# print(data['reactions','catalyzedByGene', 'genes'])
# print(data['reactions','catalyzedByGene', 'genes'].edge_index.shape)

# print(data['genes','rev_catalyzedByGene', 'reactions'])
# print(data['genes','rev_catalyzedByGene', 'reactions'].edge_index.shape)
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

gci0 = {}
for n in data.node_types:
    if n in ['genes', 'root']:
        continue
    with open(os.path.join(BASE, 'datasets/split_datasets/'
                            f'collected_{n}.pkl'), 'rb') as fi:
        gci0[n] = pickle.load(fi).training_datasets.gci0_dataset.data.to(device)
            
# if not args.model_file:
model_kwargs = {'gnn_channels': GNN_CHANNELS, 'nn_channels': NN_CHANNELS,
            'meta_data': data.metadata(), 'embeddings': data.x_dict}
metric, model = train_loop_final_sem(model_type=Regressor,
                                    model_kwargs=model_kwargs, data=data,
                                    epochs=EPOCHS, loss_function=mse_loss,
                                    metric=r2_score, device=device, lr=LR)
# else:
#     with open(os.path.join(BASE, args.model_file), 'rb') as fi:
#         model = pickle.load(fi)['model']

#     epochs = 400
#     print(model)
#     print(f"continue training for {epochs} epochs")
#     print(f"epochs: {epochs}")
#     metric, model = continue_final_training(model, data, epochs,
#                                             loss_function=mse_loss,
#                                             metric=r2_score, device=device,
#                                             lr=LR)
# %%
model.to('cpu')
file_name = time.strftime("%Y%m%d-%H%M%S") + '-reg.pkl'
with open(os.path.join(BASE, 'large_files', file_name), 'wb') as fo:
    pickle.dump({'model': model, 'metrics': metric}, fo)
# %%
