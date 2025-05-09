# %%
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import os, pickle
from sklearn.model_selection import KFold
from train_loop import link_split, node_split, get_targets_preds
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import r2_score

# %%
class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'simple_gnn':
            module = 'model'
        return super().find_class(module, name)
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device = 'cpu'
# %%
with open(os.path.join(BASE, 'datasets/split_datasets/'
                       'pyg_graph_c_DMA30_fitness.pkl'), 'rb') as fi:
    data = pickle.load(fi).contiguous()
# %%
with open(os.path.join(BASE, 'large_files/20250202-102800-reg.pkl'),
          'rb') as fi:
    # l = pickle.load(fi)
    l = RenamingUnpickler(fi).load()
    models = l['models']
    results = l['metrics']

with open(os.path.join(BASE, 'large_files/datasplit.pkl'),
          'rb') as fi:
    d = RenamingUnpickler(fi).load()['data']

best = [r['best_metric'] for r in results]
print(f"{np.mean(best)} +- {np.std(best)}")

kf = KFold(n_splits=10, shuffle=True, random_state=42)
split_transform = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=("genes", "interacts", "genes")
    )
split = 'nodes'
if split == 'nodes':
    data_to_split = data['genes'].node_id
    split_data = node_split
elif split == 'links':
    data_to_split = data['genes', 'interacts', 'genes'].edge_index.T
    split_data = link_split
else:
    raise NotImplementedError(f"split for {split} is not implemented."
                                "Use nodes or links")
# %%
all_preds = []
all_labels = []
r2s = []
for i, ((_, val_data), model) in enumerate(zip(d, models)):
    print(f"Fold: {i}")
    val_data = val_data.contiguous()
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=model._neighbors_to_sample['neighbors'],
        edge_label_index=(('genes', 'interacts', 'genes'),
                          val_data['genes', 'interacts',
                                   'genes'].edge_label_index),
        edge_label=val_data['genes', 'interacts', 'genes'].edge_label,
        batch_size=2**22,
    )
    with th.no_grad():
        total_val_loss = val_examples = 0
        val_labels = []
        val_preds = []

        for sampled_data in val_loader:
            sampled_data.to(device)
            preds = model(sampled_data)

            val_examples += preds.numel()
            targets, preds = get_targets_preds(sampled_data, preds)
            val_labels = np.concatenate((val_labels, targets))
            val_preds = np.concatenate((val_preds, preds))

        pred_errors = val_preds - val_labels
        all_preds.extend(val_preds)
        all_labels.extend(val_labels)

        r2 = r2_score(val_labels, val_preds)
        print(r2)

        r2s.append(r2)
print(f"{np.mean(r2s)} +- {np.std(r2s)}")
# %%
heatmap, xedges, yedges = np.histogram2d(all_labels, all_preds,
                                         range=([0.05,1.25], [0.4,1.1]),
                                         bins=(120,70))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
print(r2_score(all_labels, all_preds))
plt.figure()
cutoff = 700
heatmap[np.where(heatmap > cutoff)] = cutoff + \
    (heatmap[np.where(heatmap > cutoff)] -cutoff) / 10
cm = plt.cm.Blues
cm.set_under('w')
cm = cm.resampled(int(heatmap.max()))
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axline((0, 0), slope=1, linestyle='--', color='red')
im = ax.imshow(cm((heatmap.T+ (heatmap.T < 1) * (-2)).astype('int')),
               extent=extent, origin='lower', cmap=cm)
plt.xlabel("Target fitness", size=13)
plt.ylabel("Predicted fitness", size=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
# plt.savefig(os.path.join(BASE, 'paper/figs/double-parity.eps'),
            # format='eps', bbox_inches='tight')

# %%
