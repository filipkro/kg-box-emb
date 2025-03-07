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
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = 'cpu'
# %%
with open(os.path.join(BASE, 'datasets/split_datasets/'
                       'pyg_graph_c_DMA30_fitness.pkl'), 'rb') as fi:
    data = pickle.load(fi).contiguous()
# %%
with open(os.path.join(BASE, 'trained_gnns/20250202-102800-reg.pkl'),
          'rb') as fi:
    l = pickle.load(fi)
    models = l['models']
    results = l['metrics']

with open(os.path.join(BASE, 'trained_gnns/20250222-163641-reg.pkl'),
          'rb') as fi:
    d = pickle.load(fi)['data']

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
        if False:
            abs_preds = np.abs(pred_errors)
            plt.hist(pred_errors, bins=20)
            plt.show()
            plt.figure()
            plt.hist(val_preds, bins=40, range=(0.0, 1.50), density=True)
            plt.hist(val_labels, bins=40, range=(0.0, 1.50), alpha=0.5,
                     density=True)
            plt.show()
            errors = abs_preds
            sorted_errors = np.sort(errors)
            # Set up the figure and gridspec
            fig = plt.figure(figsize=(8, 4))
            gs = fig.add_gridspec(1, 2, width_ratios=[6, 1], wspace=0.01)

            # Plot the sorted prediction errors
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(sorted_errors, label="Sorted Errors", color='blue')
            ax1.set_xticks([])
            ax1.set_ylabel("|Prediction Error|")
            # ax1.legend()
            plt.axhline(0.0, xmax=len(errors)-30000,  c='b', alpha=0.2,
                        linestyle=':')

            # Plot the histogram of errors
            ax2 = fig.add_subplot(gs[1], sharey=ax1)
            ax2.hist(errors, bins=80, orientation='horizontal', color='b',
                     alpha=0.2, rwidth=0.8)
            ax1.set_xlim((-1000, len(errors)))
            ax2.set_xlabel("")
            ax2.set_xticks([])
            ax2.set_ylabel("")  # Remove redundant y-label
            ax2.axis('off')
            ax2.grid(False)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)

            # Adjust axis visibility for the histogram plot
            ax2.tick_params(left=False)  # Disable y-ticks on the histogram

            plt.show()
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
im = ax.imshow(cm((heatmap.T+ (heatmap.T < 1) * (-2)).astype('int')),
               extent=extent, origin='lower', cmap=cm)
plt.xlabel("Target fitness", size=13)
plt.ylabel("Predicted fitness", size=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.savefig(os.path.join(BASE, 'paper/figs/double-parity.eps'),
            format='eps', bbox_inches='tight')
