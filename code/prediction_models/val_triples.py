# %%
import os, pickle
from torch_geometric.loader import LinkNeighborLoader
import torch as th
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# %%
def get_data_from_idx(data, idx, transform):
    new_data = data.clone()
    new_data['genes', 'interacts', 'genes'].edge_index = \
        new_data['genes', 'interacts', 'genes'].edge_index[:, idx]
    new_data['genes', 'interacts', 'genes'].edge_label = \
        new_data['genes', 'interacts', 'genes'].edge_label[idx]
    new_data, _, _ = transform(new_data)

    return new_data

def node_split(data, split_transform, v_idx, t_idx=None, device='cpu'):
    mask = (data['genes', 'interacts', 'genes'].edge_index.unsqueeze(2) ==
            th.tensor(v_idx, device=device)).any(dim=2)
    v_mask = mask.all(dim=0)
    t_mask = (~mask).all(dim=0)
    val_data = get_data_from_idx(data, v_mask, split_transform)
    train_data = get_data_from_idx(data, t_mask, split_transform)
    return train_data, val_data
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = 'cpu'
# %%
with open(os.path.join(BASE, 'datasets/split_datasets/'
                       'pyg_graph_c_DMA30_fitness.pkl'), 'rb') as fi:
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
data['genes', 'rev_catalyzedByGene', 'reactions'].edge_index \
                = cbgt.flip(dims=(0,))

print(data['reactions','catalyzedByGene', 'genes'])
print(data['reactions','catalyzedByGene', 'genes'].edge_index.shape)

print(data['genes','rev_catalyzedByGene', 'reactions'])
print(data['genes','rev_catalyzedByGene', 'reactions'].edge_index.shape)
data = data.contiguous()
# %%
with open(os.path.join(BASE, 'trained_gnns/20250221-165714-reg.pkl'),
          'rb') as fi:
    model = pickle.load(fi)['model']

# %%
with open(os.path.join(BASE, 'datasets/split_datasets/genes.pkl'), 'rb') as fi:
    gene_index = pickle.load(fi)['index']['class_index']
rev_gene = {v: k for k,v in gene_index.items()}

# %%
df = pd.read_csv(os.path.join(BASE, 'interaction_data/unique_triples.tsv'),
                 sep='\t')

df['g1_i'] = df.apply(lambda x: gene_index['http://sgd-kg.project-genesis.io#'
                                           + x['g1']], axis=1)
df['g2_i'] = df.apply(lambda x: gene_index['http://sgd-kg.project-genesis.io#'
                                           + x['g2']], axis=1)
df['g3_i'] = df.apply(lambda x: gene_index['http://sgd-kg.project-genesis.io#'
                                           + x['g3']], axis=1)
# %%
split_transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=("genes", "interacts", "genes")
    )
data_to_split = data['genes'].node_id
split_data = node_split

# %%
def predict_from_embedding(self, emb):
    if self.lin_layers:
        for l in self.lin_layers:
            emb = l(emb).relu()
    else:
        return emb.sum(dim=-1)

    return self.lin4(emb).squeeze()
# %%
model._neighbors_to_sample['neighbors'][('reactions', 'catalyzedByGene',
                                         'genes')] = [0, 0]
model._neighbors_to_sample['neighbors'][('genes', 'rev_catalyzedByGene',
                                         'reactions')] = [0, 0]
data_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=model._neighbors_to_sample['neighbors'],
        edge_label_index=(('genes', 'interacts', 'genes'),
                        data['genes', 'interacts',
                                'genes'].edge_index),
        edge_label=data['genes', 'interacts', 'genes'].edge_label,
        batch_size=2**25,
    )
sampled_data = next(iter(data_loader))
genes = data['genes', 'interacts', 'genes']['edge_index'].unique().numpy()
emb = model.gene_embedding(data)
triples = df[['g1_i', 'g2_i', 'g3_i']].to_numpy()
gene_emb = emb[triples[:,0]] * emb[triples[:,1]] * emb[triples[:,2]]

triple_fitness = predict_from_embedding(model, gene_emb).detach().numpy()
labels = df['Combined mutant fitness'].to_numpy()

print(r2_score(labels, triple_fitness))

# %%
heatmap, xedges, yedges = np.histogram2d(labels, triple_fitness, bins=(120,70),
                                         range=([0,1.2], [0.4,1.1]))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.figure()
cutoff = 20
heatmap[np.where(heatmap > cutoff)] = cutoff + \
    (heatmap[np.where(heatmap > cutoff)] -cutoff) / 5

cm = plt.cm.Blues
cm.set_under('w')
cm = cm.resampled(int(heatmap.max()))
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
im = ax.imshow(cm((heatmap.T + (heatmap.T < 1) * (-7)).astype('int')),
               extent=extent, origin='lower', cmap=cm)
plt.xlabel("Target fitness", size=13)
plt.ylabel("Predicted fitness", size=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.savefig(os.path.join(BASE, 'paper/figs/triple-parity.eps'),
            format='eps', bbox_inches='tight')
plt.show()
