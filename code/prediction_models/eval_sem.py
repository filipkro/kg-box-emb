# %%
import pickle, os, sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../embeddings')))
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE)


# %%
# with open(BASE + '/sem_weight_trained_models/20250813-172958-reg.pkl', 'rb') as fi:
# with open(BASE + '/sem_weight_trained_models/20250813-224854-reg.pkl', 'rb') as fi:
# with open(BASE + '/sem_weight_trained_models/20250814-214050-reg.pkl', 'rb') as fi:
with open(BASE + '/sem_weight_trained_models/20250819-142859-reg.pkl', 'rb') as fi:
    d = pickle.load(fi)
models = d['models']
metrics = d['metrics']
# %%
with open(BASE + '/sem_weight_trained_models/20250814-154851-reg.pkl', 'rb') as fi:
    d2 = pickle.load(fi)
models2 = d2['models']
metrics2 = d2['metrics']

# %%
with open(BASE + '/sem_weight_trained_models/20250817-020746-reg.pkl', 'rb') as fi:
    d3 = pickle.load(fi)
models3 = d3['models']
metrics3 = d3['metrics']
# %%
gci0 = {}
for n in metrics[0]['box_losses'][0][0][0]['pos'].keys():
    if n in ['genes', 'root']:
        continue
    with open(os.path.join(BASE, 'datasets/split_datasets/'
                            f'collected_{n}.pkl'), 'rb') as fi:
        gci0[n] = pickle.load(fi).training_datasets.gci0_dataset.data.to('cpu')

total_gci0 = sum([len(v) for v in gci0.values()])
# %%
moi = ['sem_losses']
sem_losses = [f['sem_losses'][:-21] for f in metrics]
neg_sem_losses = [f['neg_sem_losses'][:-21] for f in metrics]
val_metrics = [f['val_metrics'][:-21] for f in metrics]
# lens = [len(f) for f in sem_losses]
ml = max([len(f) for f in sem_losses])
# ml = max([len(q) for q in corr_lens])
padded_sem = np.array([f + (ml - len(f)) * [f[-1]] for f in sem_losses]) / (2*total_gci0)
padded_neg_sem = np.array([f + (ml - len(f)) * [f[-1]] for f in neg_sem_losses]) / (2*3*total_gci0)

# %%
x = np.arange(padded_sem.shape[1])
y = padded_sem.mean(axis=0)
std = padded_sem.std(axis=0)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, np.zeros_like(y), '--', alpha=0.2)
ax.fill_between(x, y - std, y + std, alpha=0.2)

x = np.arange(padded_neg_sem.shape[1])
y = padded_neg_sem.mean(axis=0)
std = padded_neg_sem.std(axis=0)
ax.plot(x, y)
ax.fill_between(x, y - std, y + std, alpha=0.2)
# ax.set_ylim(-0.1,0.5)
# %%
e1 = sem_losses[0]
# %%
lens = [len(f['train_losses']) for f in metrics]
# %%
plt.figure()
for f in metrics:
    plt.plot(f['val_losses'])
# %%
plt.figure()
for f in metrics:
    plt.plot(f['val_metrics'])
# plt.
# %%
corr_lens = [f['val_metrics'][:-21] for f in metrics]
# %%
ml = max([len(q) for q in corr_lens])
padded = [f + (ml - len(f)) * [f[-1]] for f in corr_lens]
# %%
plt.plot(np.array(padded).mean(axis=0))
# %%
x = np.arange(np.array(padded).shape[1])
y = np.array(padded).mean(axis=0)
std = np.array(padded).std(axis=0)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, np.zeros_like(y), '--', alpha=0.2)
ax.fill_between(x, y - std, y + std, alpha=0.2)
ax.set_ylim(-1,0.5)
# %%
# sem_losses = [f['sem_losses'][:-31] for f in metrics]
ml = max([len(f['sem_losses'][:-21]) for f in metrics])
fold_losses = []
domains = list(metrics[0]['box_losses'][0][0][0]['pos'].keys())
for f in metrics:
    bl = ({'pos': {k: [] for k in domains},
           'neg': {k: [] for k in domains}},
          {'pos': {k: [] for k in domains},
           'neg': {k: [] for k in domains}})
    for epoch in f['box_losses']:
        for dom in domains:
            for i in [0, 1]:
                for p in ['pos', 'neg']:
                    bl[i][p][dom].append(epoch[-1][i][p][dom])
    for dom in domains:
        for i in [0, 1]:
            for p in ['pos', 'neg']:
                losses = bl[i][p][dom][:-31]
                bl[i][p][dom] = np.array(losses + (ml - len(losses)) *
                                         [losses[-1]]) / len(gci0[dom])
    #         .append(epoch[-1][i]['pos'][dom])
    #         bl[i]['neg'][dom].append(epoch[-1][i]['neg'][dom])
    # np.array([f + (ml - len(f)) * [f[-1]] for f in sem_losses])
    fold_losses.append(bl)
dom_losses = {}
for dom in domains:
    dom_losses[dom] = {}
    for p in ['pos', 'neg']:
        dom_losses[dom][p] = [np.array([f[i][p][dom] for f in fold_losses]) for i in [0, 1]]
# %%
from matplotlib.colors import ListedColormap
cmap_dict = {
    (0, 'pos'): '#6baed6',  # light blue
    (0, 'neg'): '#08519c',  # dark blue
    (1, 'pos'): '#fd8d3c',  # light orange
    (1, 'neg'): '#a63603',  # dark orange
}

cmap_dict = {
    (0, 'pos'): '#74c476',  # light green
    (0, 'neg'): '#006d2c',  # dark green
    (1, 'pos'): '#9e9ac8',  # light purple
    (1, 'neg'): '#54278f',  # dark purple
}

# cmap_dict = {
#     (0, 'pos'): '#66c2a4',  # light teal
#     (0, 'neg'): '#006d65',  # dark teal
#     (1, 'pos'): '#fc9272',  # light red
#     (1, 'neg'): '#cb181d',  # dark red
# }
# cmap_dict = {
#     (0, 'pos'): '#56B4E9',  # sky blue
#     (0, 'neg'): '#0072B2',  # deep blue
#     (1, 'pos'): '#E69F00',  # light orange
#     (1, 'neg'): '#D55E00',  # dark orange
# }
# cmap_dict = {
#     (0, 'pos'): '#fdae6b',  # bright orange
#     (0, 'neg'): '#6baed6',  # bright blue
#     (1, 'pos'): '#e6550d',  # dark orange
#     (1, 'neg'): '#3182bd',  # dark blue
# }
pn_map = {'pos': 'Positive', 'neg': 'Negative'}
# If you want to create a ListedColormap for plotting
cmap = ListedColormap(list(cmap_dict.values()))

title_map = {'bio_proc': 'Biological process', 'mol_func': 'Molecular function', 'reguls': 'Regulation', 'cell_comp': 'Cellular component', 'reactions': 'Reaction or Pathway', 'quality': 'Quality', 'mat_ent': 'Material entity'}
x = np.arange(ml)
fontsize=20
for k, v in dom_losses.items():
    fig, ax = plt.subplots()
    
    # ax.set_title(title_map[k])

    for i in [1,0]:
        for p in ['pos', 'neg']:
            y = (v[p][i] / 3).mean(axis=0) if p == 'neg' else v[p][i].mean(axis=0)
            std = (v[p][i] / 3).std(axis=0) if p == 'neg' else v[p][i].std(axis=0)
            c = cmap_dict[(i, p)]
            ax.fill_between(x, y - std, y + std, color=c, alpha=0.2)
            ax.plot(x, y, color=c, label=f'Layer {i}, {p}')
    ax.set_xlabel('Epoch', size=fontsize)
    ax.set_ylabel('Distance', size=fontsize)
    ax.legend(fontsize=fontsize)
    # break
    fig.savefig(os.path.join(BASE, f'../nai-manuscript/paper/figs/sem_loss-{k}.pdf'),
                format='pdf', bbox_inches='tight')




# ax.plot(x, y)
# ax.plot(x, np.zeros_like(y), '--', alpha=0.2)


# x = np.arange(padded_neg_sem.shape[1])
# y = padded_neg_sem.mean(axis=0)
# std = padded_neg_sem.std(axis=0)
# ax.plot(x, y)
# ax.fill_between(x, y - std, y + std, alpha=0.2)
# %%
cmap_dict = {
    (0, 'pos'): '#74c476',  # light green
    (0, 'neg'): '#006d2c',  # dark green
    (1, 'pos'): '#9e9ac8',  # light purple
    (1, 'neg'): '#54278f',  # dark purple
}
# }
pn_map = {'pos': 'Positive', 'neg': 'Negative'}

title_map = {'bio_proc': 'Biological process', 'mol_func': 'Molecular function', 'reguls': 'Regulation', 'cell_comp': 'Cellular component', 'reactions': 'Reactions and Pathways', 'quality': 'Quality', 'mat_ent': 'Material entity'}
x = np.arange(ml)
fontsize=20

fig, axes = plt.subplots(2, 4, figsize=(16, 8))#, sharex=True, sharey=True)

# Flatten axes for easier indexing
axes = axes.flatten()

# --- Example plotting loop ---


for i, (k, v) in enumerate(dom_losses.items()):
    # fig, ax = plt.subplots()
    if i > 2:
        i += 1
    ax = axes[i]
    ax.set_title(title_map[k], size=fontsize)

    for i in [1,0]:
        for p in ['pos', 'neg']:
            y = (v[p][i] / 3).mean(axis=0) if p == 'neg' else v[p][i].mean(axis=0)
            std = (v[p][i] / 3).std(axis=0) if p == 'neg' else v[p][i].std(axis=0)
            c = cmap_dict[(i, p)]
            ax.fill_between(x, y - std, y + std, color=c, alpha=0.2)
            ax.plot(x, y, color=c, label=f'Layer {i}, {pn_map[p]}')

    ax.tick_params(axis='both', which='major', labelsize=18)

    # ax.xticks(fontsize=fontsize-4)
    # ax.yticks(fontsize=fontsize-4)
    # if i > -1:
    #     ax.set_xlabel('Epoch', size=fontsize)
    # ax.set_ylabel('Distance', size=fontsize)
    # ax.legend(fontsize=fontsize)

# Hide the 8th subplot and put legend there
axes[3].axis("off")
handles, labels = axes[0].get_legend_handles_labels()
axes[3].legend(handles, labels, loc="center", fontsize=fontsize)

axes[4].set_xlabel('Epoch', size=fontsize)
axes[5].set_xlabel('Epoch', size=fontsize)
axes[6].set_xlabel('Epoch', size=fontsize)
axes[7].set_xlabel('Epoch', size=fontsize)
axes[0].set_ylabel('Distance', size=fontsize)
axes[4].set_ylabel('Distance', size=fontsize)
plt.tight_layout()


plt.show()
fig.savefig(os.path.join(BASE, f'../nai-manuscript/paper/figs/sem_losses.pdf'),
                format='pdf', bbox_inches='tight')






# %%
with open(BASE + '/sem_weight_trained_models/20250814-162526-reg.pkl', 'rb') as fi:
    d = pickle.load(fi)
# models = d['models']
comp_metric = [f['best_metric'] for f in d['metrics']]

# %%
bmetric = [f['best_metric'] for f in metrics]
print(f"{np.mean(bmetric)} +- {np.std(bmetric)}")
# %%
from scipy import stats
t_stat, p_value = stats.ttest_rel(bmetric, comp_metric, alternative='greater')
print(t_stat)
print(p_value)
# %%
prev_res = [0.35877937,
 0.31752749,
 0.41195499,
 0.33200223,
 0.34888286,
 0.36384635,
 0.39444053,
 0.28950511,
 0.34489728,
0.44055303]
# %%
from scipy import stats
t_stat, p_value = stats.ttest_rel(bmetric, prev_res, alternative='greater')
print(t_stat)
print(p_value)
# %%
t_stat, p_value = stats.ttest_ind(bmetric, prev_res, alternative='greater')
print(t_stat)
print(p_value)
# %%
print('lol')
# %%
