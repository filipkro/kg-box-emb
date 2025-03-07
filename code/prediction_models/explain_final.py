# %%
import torch as th
import os, pickle
from torch_geometric.loader import LinkNeighborLoader
from model import Model
from torch_geometric.data import HeteroData
from torch_geometric.explain import CaptumExplainer, Explainer
import pandas as pd
import itertools
import warnings
import tqdm
warnings.filterwarnings("ignore")
# %%
def simplify_graph(graph, sampled_data):
    for k in graph.node_types:
        if len(graph[k]['x']) == 0:
            del graph[k]
        else:
            graph[k].node_id = sampled_data[k].node_id
    for k in graph.edge_types:
        if len(graph[k]['edge_mask']) == 0:
            del graph[k]
    return graph

def get_important_edges(graph, reduction='max'):
    df = pd.DataFrame()
    vals = []
    edges = []
    edge_index = graph.collect('edge_index')
    for k, v in graph.collect('edge_mask').items():
        edf = pd.DataFrame()
        edf['idx'] = edge_index[k][0,:].detach().numpy()
        edf['mask'] = v.detach().numpy()
        edf = edf.groupby(by='idx').sum()
        match reduction:
            case 'max':
                val = edf['mask'].max().item()
            case 'mean':
                val = edf['mask'].mean().item()
            case 'sum':
                val = edf['mask'].sum().item()
            case _:
                raise NotImplementedError()
        vals.append(val)
        edges.append(' '.join(k))
    df['edge'] = edges
    df['val'] = vals
    df.sort_values(by='val', inplace=True, ascending=False)
    print(df.head())
    return df

def get_top_indices_from_edge(graph, edge, K=1):
    df = pd.DataFrame()
    df['idx'] = graph.collect('edge_index')[edge][0,:].detach().numpy()
    df['mask'] = graph.collect('edge_mask')[edge].detach().numpy()
    df = df.groupby(by='idx').sum().reset_index()
    df.sort_values(by='mask', inplace=True, ascending=False)
    print(df.head())
    if K < 0:
        K = len(df)
    class_idx = graph[edge[0]]['node_id'][df['idx'][:K].values]
    return class_idx, th.tensor(df['mask'][:K].values)

def sort_rows(r1, r2):
    if r1['dom'] < r2['dom']:
        return r1, r2
    elif r1['dom'] > r2['dom']:
        return r2, r1
    else:
        if r1['rel'] < r2['rel']:
            return r1, r2
        elif r1['rel'] > r2['rel']:
            return r2, r1
        else:
            if r1['idx'] < r2['idx']:
                return r1, r2
            elif r1['idx'] > r2['idx']:
                return r2, r1
            
    return r1, r2

def get_pair_df(df1, df2, combination='mult'):
    prod = itertools.product(range(len(df1)), range(len(df2)))
    # print(prod)
    df_dict = {'idx': [], 'imp': [], 'rel': [], 'dom': []}
    for i1, i2 in prod:
        r1, r2 = sort_rows(df1.iloc[i1], df2.iloc[i2])
        df_dict['idx'].append(f"{r1['idx']}--{r2['idx']}")
        df_dict['rel'].append(f"{r1['rel']}--{r2['rel']}")
        df_dict['dom'].append(f"{r1['dom']}--{r2['dom']}")
        if combination == 'mult':
            df_dict['imp'].append(r1['imp'] * r2['imp'])
        elif combination == 'sum':
            df_dict['imp'].append(r1['imp'] + r2['imp'])
        else:
            raise NotImplementedError(f"Combination {combination} not "
                                      "implemented, use 'mult' or 'sum'.")
    
    return pd.DataFrame.from_dict(df_dict)

def get_class_names(row):
    dom1, dom2 = row['dom'].split('--')
    rel1, rel2 = row['rel'].split('--')
    idx1, idx2 = row['idx'].split('--')

    return (rev_dicts[dom1][int(idx1)], rel1, rev_dicts[dom2][int(idx2)], rel2)

def collaps_columns(row, combination='mult'):
    r1 = row[['dom_x', 'rel_x', 'idx_x', 'imp_x']]
    r1.index = ['dom', 'rel', 'idx', 'imp']
    r2 = row[['dom_y', 'rel_y', 'idx_y', 'imp_y']]
    r2.index = ['dom', 'rel', 'idx', 'imp']
    r1, r2 = sort_rows(r1, r2)
    if combination == 'mult':
        imp = r1['imp'] * r2['imp']
    elif combination == 'sum':
        imp = r1['imp'] + r2['imp']
    else:
        raise NotImplementedError(f"Combination {combination} not "
                                  "implemented, use 'mult' or 'sum'.")
    
    return f"{r1['idx']}--{r2['idx']}", f"{r1['rel']}--{r2['rel']}", \
                    f"{r1['dom']}--{r2['dom']}", imp
# %%
class RegressorFromModel(Model):
    def __init__(self, model, node_ids):
        super().__init__([1], [1], None, {k: v.weight for k,v in
                                          model.node_embeddings.items()},
                                          custom=False)
        self.lin_layers = model.lin_layers
        self.lin4 = model.lin4
        self.gnn = model.gnn
        self.node_embeddings = model.node_embeddings
        self.fp = model.fp
        self._neighbors_to_sample = model._neighbors_to_sample
        self.node_ids = node_ids

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        data = HeteroData()
        for k, v in x_dict.items():
            data[k].x = v
        for k, v in edge_index_dict.items():
            data[k].edge_index = v
        data['genes', 'interacts', 'genes'].edge_label_index = edge_label_index
        for k, v in self.node_ids.items():
            data[k].node_id = v
        z = self._forward(data)
        return self.lin4(z).squeeze()
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = 'cpu'
# %%
with open(os.path.join(BASE, 'trained_gnns/20250205-102437-reg.pkl'),
          'rb') as fi:
    m = pickle.load(fi)['model']
m.to(device)
# print(models[0])
# %%
rev_dicts = {}
with open(os.path.join(BASE, 'datasets/split_datasets/mat_ent.pkl'),
          'rb') as fi:
    mat_index = pickle.load(fi)['index']['class_index']
rev_mat = {v: k for k,v in mat_index.items()}
rev_dicts['mat_ent'] = rev_mat

with open(os.path.join(BASE, 'datasets/split_datasets/genes.pkl'), 'rb') as fi:
    gene_index = pickle.load(fi)['index']['class_index']
rev_gene = {v: k for k,v in gene_index.items()}
rev_dicts['genes'] = rev_gene

with open(os.path.join(BASE, 'datasets/split_datasets/quality.pkl'),
          'rb') as fi:
    quality_index = pickle.load(fi)['index']['class_index']
rev_quality = {v: k for k,v in quality_index.items()}
rev_dicts['quality'] = rev_quality

with open(os.path.join(BASE, 'datasets/split_datasets/bio_proc.pkl'),
          'rb') as fi:
    bio_proc_index = pickle.load(fi)['index']['class_index']
rev_bio_proc = {v: k for k,v in bio_proc_index.items()}
rev_dicts['bio_proc'] = rev_bio_proc

with open(os.path.join(BASE, 'datasets/split_datasets/cell_comp.pkl'),
          'rb') as fi:
    cell_comp_index = pickle.load(fi)['index']['class_index']
rev_cell_comp = {v: k for k,v in cell_comp_index.items()}
rev_dicts['cell_comp'] = rev_cell_comp

with open(os.path.join(BASE, 'datasets/split_datasets/mol_func.pkl'),
          'rb') as fi:
    mol_func_index = pickle.load(fi)['index']['class_index']
rev_mol_func = {v: k for k,v in mol_func_index.items()}
rev_dicts['mol_func'] = rev_mol_func

with open(os.path.join(BASE, 'datasets/split_datasets/reactions.pkl'),
          'rb') as fi:
    react_index = pickle.load(fi)['index']['class_index']
rev_react = {v: k for k,v in react_index.items()}
rev_dicts['reactions'] = rev_react

with open(os.path.join(BASE, 'datasets/split_datasets/reguls.pkl'), 'rb') as fi:
    reguls_index = pickle.load(fi)['index']['class_index']
rev_reguls = {v: k for k,v in reguls_index.items()}
rev_dicts['reguls'] = rev_reguls

with open(os.path.join(BASE, 'datasets/split_datasets/root.pkl'), 'rb') as fi:
    root_index = pickle.load(fi)['index']['class_index']
rev_root = {v: k for k,v in root_index.items()}
rev_dicts['root'] = rev_root
# %%
with open(os.path.join(BASE, 'datasets/split_datasets/'
                       'pyg_graph_c_DMA30_fitness.pkl'), 'rb') as fi:
    data = pickle.load(fi).contiguous()
data.to(device)
# %%
# ignore uninformative classes
i = mol_func_index['http://purl.obolibrary.org/obo/GO_0003674']
a = data[('genes', 'RO_0002327', 'mol_func')]['edge_index']
print(data[('genes', 'RO_0002327', 'mol_func')]['edge_index'].shape)
data[('genes', 'RO_0002327', 'mol_func')]['edge_index'] = a[:, a[1,:] != i]
print(data[('genes', 'RO_0002327', 'mol_func')]['edge_index'].shape)
print(data[('mol_func', 'rev_RO_0002327', 'genes')]['edge_index'].shape)
a = data[('mol_func', 'rev_RO_0002327', 'genes')]['edge_index']
data[('mol_func', 'rev_RO_0002327', 'genes')]['edge_index'] = a[:, a[0,:] != i]
print(data[('mol_func', 'rev_RO_0002327', 'genes')]['edge_index'].shape)

i = bio_proc_index['http://purl.obolibrary.org/obo/GO_0008150']
a = data[('genes', 'RO_0002331', 'bio_proc')]['edge_index']
print(data[('genes', 'RO_0002331', 'bio_proc')]['edge_index'].shape)
data[('genes', 'RO_0002331', 'bio_proc')]['edge_index'] = a[:, a[1,:] != i]
print(data[('genes', 'RO_0002331', 'bio_proc')]['edge_index'].shape)
a = data[('bio_proc', 'rev_RO_0002331', 'genes')]['edge_index']
print(data[('bio_proc', 'rev_RO_0002331', 'genes')]['edge_index'].shape)
data[('bio_proc', 'rev_RO_0002331', 'genes')]['edge_index'] = a[:, a[0,:] != i]
print(data[('bio_proc', 'rev_RO_0002331', 'genes')]['edge_index'].shape)

i = quality_index['http://purl.obolibrary.org/obo/APO_0000113']
a = data[('genes', 'RO_0002200', 'quality')]['edge_index']
print(data[('genes', 'RO_0002200', 'quality')]['edge_index'].shape)
data[('genes', 'RO_0002200', 'quality')]['edge_index'] = a[:, a[1,:] != i]
print(data[('genes', 'RO_0002200', 'quality')]['edge_index'].shape)
a = data[('quality', 'rev_RO_0002200', 'genes')]['edge_index']
print(data[('quality', 'rev_RO_0002200', 'genes')]['edge_index'].shape)
data[('quality', 'rev_RO_0002200', 'genes')]['edge_index'] = a[:, a[0,:] != i]
print(data[('quality', 'rev_RO_0002200', 'genes')]['edge_index'].shape)

i = quality_index['http://purl.obolibrary.org/obo/APO_0000112']
a = data[('genes', 'RO_0002200', 'quality')]['edge_index']
print(data[('genes', 'RO_0002200', 'quality')]['edge_index'].shape)
data[('genes', 'RO_0002200', 'quality')]['edge_index'] = a[:, a[1,:] != i]
print(data[('genes', 'RO_0002200', 'quality')]['edge_index'].shape)
a = data[('quality', 'rev_RO_0002200', 'genes')]['edge_index']
print(data[('quality', 'rev_RO_0002200', 'genes')]['edge_index'].shape)
data[('quality', 'rev_RO_0002200', 'genes')]['edge_index'] = a[:, a[0,:] != i]
print(data[('quality', 'rev_RO_0002200', 'genes')]['edge_index'].shape)
# if False:
for k in quality_index.keys():
    if 'APO_0000110' in k and 'CHEBI' not in k:
        i = quality_index[k]
        a = data[('genes', 'RO_0002200', 'quality')]['edge_index']
        data[('genes', 'RO_0002200', 'quality')]['edge_index'] = a[:, a[1,:]
                                                                        != i]
        a = data[('quality', 'rev_RO_0002200', 'genes')]['edge_index']
        data[('quality', 'rev_RO_0002200', 'genes')]['edge_index'] = a[:,a[0,:]
                                                                        != i]

print(data[('genes', 'RO_0002200', 'quality')]['edge_index'].shape)
print(data[('quality', 'rev_RO_0002200', 'genes')]['edge_index'].shape)

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

data = data.contiguous()

# %%
NBR_SAMPLES = 1000
CUTOFF = 1e-6
df_pairs = pd.DataFrame()

data_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=m._neighbors_to_sample['neighbors'],
        edge_label_index=(('genes', 'interacts', 'genes'),
                        data['genes', 'interacts',
                                'genes'].edge_index),
        edge_label=data['genes', 'interacts', 'genes'].edge_label,
        batch_size=2**20,
    )
print('running loop')

for sampled_val_data in tqdm.tqdm(data_loader):
    model = RegressorFromModel(m, sampled_val_data.node_id_dict)
    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('InputXGradient'),
        explanation_type='model',
        model_config=dict(
            mode='regression',
            task_level='edge',
            return_type='raw',
        ),
        edge_mask_type='object',
        threshold_config=dict(
            threshold_type='topk',
            value=10000,
        ),
    )
    preds = m(sampled_val_data)
    diffs = (preds - sampled_val_data['genes', 'interacts',
                                      'genes']['edge_label']).abs()
    mean_fitness = sampled_val_data['genes', 'interacts',
                                    'genes']['edge_label'].mean()
    interest_ratios = (sampled_val_data['genes', 'interacts',
                                        'genes']['edge_label'] -
                                        mean_fitness).abs() / diffs

    index = th.argsort(interest_ratios, descending=True)[:2*NBR_SAMPLES][::2]
    print()
    print('finding explanations')
    nbr_dup = 0
    df_fold = pd.DataFrame()
    for ii, i in enumerate(index):
        print(f'{ii} of {NBR_SAMPLES} done...', end='\r')
        explanation = explainer(
            sampled_val_data.x_dict,
            sampled_val_data.edge_index_dict,
            index=i,
            edge_label_index=sampled_val_data['genes', 'interacts',
                                              'genes']['edge_label_index']
        )
        genes = sampled_val_data['genes', 'interacts',
                                 'genes']['edge_label_index'][:,i]
        gs = explanation.get_explanation_subgraph()
        gs_s = simplify_graph(gs, sampled_val_data)
        g1 = genes[0]
        g2 = genes[1]
        
        df1_dict = {'idx': [], 'imp': [], 'rel': [], 'dom': []}
        df2_dict = {'idx': [], 'imp': [], 'rel': [], 'dom': []}
        for k in gs_s.edge_types:
            ei = gs_s.edge_index_dict[k]
            em = gs_s.edge_mask_dict[k]
            i1 = ei[1,:] == g1
            i2 = ei[1,:] == g2
            nid = gs_s[k[0]]['node_id']
            df1_dict['idx'].extend(nid[ei[0,:][i1].detach().numpy()])
            df1_dict['imp'].extend(em[i1].detach().numpy())
            df2_dict['idx'].extend(nid[ei[0,:][i2].detach().numpy()])
            df2_dict['imp'].extend(em[i2].detach().numpy())
            df1_dict['rel'].extend([k[1]] * i1.sum().item())
            df2_dict['rel'].extend([k[1]] * i2.sum().item())
            df1_dict['dom'].extend([k[0]] * i1.sum().item())
            df2_dict['dom'].extend([k[0]] * i2.sum().item())
            
        df1 = pd.DataFrame.from_dict(df1_dict)
        df2 = pd.DataFrame.from_dict(df2_dict)
        df1 = df1[df1['imp'] > CUTOFF]
        df2 = df2[df2['imp'] > CUTOFF]
        if len(df1) > 0 and len(df2) > 0:
            dfp = pd.DataFrame()
            dfp['idx'], dfp['rel'], dfp['dom'], dfp['imp'] \
                    = zip(*pd.merge(df1, df2,
                                    how='cross').apply(collaps_columns, axis=1))
        
            dfp['occ'] = [1] * len(dfp)
            df_fold = pd.concat([df_fold, dfp])
    df_pairs = pd.concat([df_pairs, df_fold])
    len_bef = len(df_pairs)
    df_pairs = df_pairs.groupby(['idx', 'rel', 'dom'],
                                as_index=False).agg('sum')
    nbr_dup += len_bef - len(df_pairs)
    print()
    print(f'Duplicate pairs in this sample: {nbr_dup}')
    print(df_pairs.shape)

df_pairs.sort_values(by='imp', ascending=False, inplace=True)
print(df_pairs.head(10))

df_pairs['class1'], df_pairs['rel1'], df_pairs['class2'], df_pairs['rel2'] \
                    = zip(*df_pairs.apply(get_class_names, axis=1))
print(df_pairs.head(10))
# %%

df_pairs.to_csv('prel_explanationsDMA30-InputXGradient-full_model-XX-10000.tsv',
                sep='\t')
# %%
