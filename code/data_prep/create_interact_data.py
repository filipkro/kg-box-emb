# %%
import pandas as pd
import pickle, os
import torch as th
# %matplotlib inline
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE)
# %%
df = pd.read_csv(os.path.join(BASE, 'data/interaction_data/SGA_NxN.txt'), delimiter='\t')
df.drop(['Query allele name', 'Array allele name'] , axis=1, inplace=True)

df = df.loc[df['P-value'] < 0.8]
df['abs-interact'] = df['Genetic interaction score (ε)'].abs()
df['Query Strain ID'] = df['Query Strain ID'].apply(lambda x: x.split('_')[0])
df['Array Strain ID'] = df['Array Strain ID'].apply(lambda x: x.split('_')[0])
print(df.head())
df_refined = df.loc[df['P-value'] < 0.05]
df_refined_DMA30 = df_refined.loc[df_refined['Arraytype/Temp'] == 'DMA30']
df_DMA30 = df.loc[df['Arraytype/Temp'] == 'DMA30']
# %%
dfE = pd.read_csv(os.path.join(BASE, 'data/interaction_data/SGA_ExE.txt'), delimiter='\t')
dfE.drop(['Query allele name', 'Array allele name'] , axis=1, inplace=True)
dfE = dfE.loc[dfE['P-value'] < 0.8]
dfE['abs-interact'] = dfE['Genetic interaction score (ε)'].abs()
dfE['Query Strain ID'] = dfE['Query Strain ID'].apply(lambda x: x.split('_')[0])
dfE['Array Strain ID'] = dfE['Array Strain ID'].apply(lambda x: x.split('_')[0])
# print(dfE.head())
dfE_refined = dfE.loc[dfE['P-value'] < 0.05]
print(dfE.shape)
print(dfE_refined.shape)
# %%
dfNE = pd.read_csv(os.path.join(BASE, 'data/interaction_data/SGA_ExN_NxE.txt'), delimiter='\t')
dfNE.drop(['Query allele name', 'Array allele name'] , axis=1, inplace=True)

dfNE = dfNE.loc[dfNE['P-value'] < 0.8]
dfNE['abs-interact'] = dfNE['Genetic interaction score (ε)'].abs()
dfNE['Query Strain ID'] = dfNE['Query Strain ID'].apply(lambda x: x.split('_')[0])
dfNE['Array Strain ID'] = dfNE['Array Strain ID'].apply(lambda x: x.split('_')[0])
dfNE_refined = dfNE.loc[dfNE['P-value'] < 0.05]
print(dfNE.shape)
print(dfNE_refined.shape)
dfNE_refined_DMA30 = dfNE_refined.loc[dfNE_refined['Arraytype/Temp'] == 'DMA30']
dfNE_DMA30 = dfNE.loc[dfNE['Arraytype/Temp'] == 'DMA30']
# %%

class Graph:
    def __init__(self):
        self.adjacency = {}
    
    def add_edge(self, n1, n2, i, f, p):
        pair = sorted([n1, n2])
        if pair[0] in self.adjacency:
            if pair[1] in self.adjacency[pair[0]]:
                # print(pair[0], pair[1])
                if p < self.adjacency[pair[0]][pair[1]][1]:
                    self.adjacency[pair[0]][pair[1]] = (i, f, p)
            else:
                self.adjacency[pair[0]][pair[1]] = (i, f, p)
        else:
            self.adjacency[pair[0]] = {pair[1]: (i, f, p)}

    def get_i(self, n1, n2):
        pair = sorted([n1, n2])
        if pair[0] in self.adjacency and pair[1] in self.adjacency[pair[0]]:
            return self.adjacency[pair[0]][pair[1]][0]
        else:
            return False
        
    def get_f(self, n1, n2):
        pair = sorted([n1, n2])
        if pair[0] in self.adjacency and pair[1] in self.adjacency[pair[0]]:
            return self.adjacency[pair[0]][pair[1]][1]
        else:
            return False

    def get_p(self, n1, n2):
        pair = sorted([n1, n2])
        if pair[0] in self.adjacency and pair[1] in self.adjacency[pair[0]]:
            return self.adjacency[pair[0]][pair[1]][2]
        else:
            return False
        
    def get_all_edges(self):
        edges = []
        for g in self.adjacency:
            for gg, v in self.adjacency[g].items():
                edges.append([g, gg, v[0], v[1]])
        return edges


# %%
print('populating graph')
g = Graph()
df.apply(lambda row: g.add_edge(row['Query Strain ID'], row['Array Strain ID'],
                                row['Genetic interaction score (ε)'],
                                row['Double mutant fitness'],
                                row['P-value']), axis=1)
# %%
dfE.apply(lambda row: g.add_edge(row['Query Strain ID'], row['Array Strain ID'],
                                row['Genetic interaction score (ε)'],
                                row['Double mutant fitness'],
                                row['P-value']), axis=1)
# %%
dfNE.apply(lambda row: g.add_edge(row['Query Strain ID'], row['Array Strain ID'],
                                row['Genetic interaction score (ε)'],
                                row['Double mutant fitness'],
                                row['P-value']), axis=1)

# %%
print('populating refined graph')
g_refined = Graph()
df_refined.apply(lambda row: g_refined.add_edge(row['Query Strain ID'], row['Array Strain ID'],
                                row['Genetic interaction score (ε)'],
                                row['Double mutant fitness'],
                                row['P-value']), axis=1)
# %%
dfE_refined.apply(lambda row: g_refined.add_edge(row['Query Strain ID'], row['Array Strain ID'],
                                row['Genetic interaction score (ε)'],
                                row['Double mutant fitness'],
                                row['P-value']), axis=1)
# %%
dfNE_refined.apply(lambda row: g_refined.add_edge(row['Query Strain ID'], row['Array Strain ID'],
                                row['Genetic interaction score (ε)'],
                                row['Double mutant fitness'],
                                row['P-value']), axis=1)

# %%
print('populating DMA30 graph')
g_DMA30 = Graph()
df_DMA30.apply(lambda row: g_DMA30.add_edge(row['Query Strain ID'], row['Array Strain ID'],
                                row['Genetic interaction score (ε)'],
                                row['Double mutant fitness'],
                                row['P-value']), axis=1)

# %%
dfNE_DMA30.apply(lambda row: g_DMA30.add_edge(row['Query Strain ID'], row['Array Strain ID'],
                                row['Genetic interaction score (ε)'],
                                row['Double mutant fitness'],
                                row['P-value']), axis=1)

# %%
print('populating refined DMA30 graph')
g_refined_DMA30 = Graph()
df_refined_DMA30.apply(lambda row: g_refined_DMA30.add_edge(row['Query Strain ID'], row['Array Strain ID'],
                                row['Genetic interaction score (ε)'],
                                row['Double mutant fitness'],
                                row['P-value']), axis=1)

# %%
dfNE_refined_DMA30.apply(lambda row: g_refined_DMA30.add_edge(row['Query Strain ID'], row['Array Strain ID'],
                                row['Genetic interaction score (ε)'],
                                row['Double mutant fitness'],
                                row['P-value']), axis=1)
# %%
print('retrieving edges from graph')
edges = g.get_all_edges()
print(len(edges))
# %%
print('retrieving edges from refined graph')
refined_edges = g_refined.get_all_edges()
print(len(refined_edges))
# %%
print('retrieving edges from DMA30 graph')
edges_DMA30 = g_DMA30.get_all_edges()
print(len(edges_DMA30))
# %%
print('retrieving edges from refined DMA30 graph')
refined_edges_DMA30 = g_refined_DMA30.get_all_edges()
print(len(refined_edges_DMA30))
# %%
print('loading index')
with open(os.path.join(BASE, 'datasets/split_datasets/genes.pkl'), 'rb') as fi:
    index = pickle.load(fi)['index']['class_index']
# %%
kg = 'http://sgd-kg.project-genesis.io#'
edges_idx = []
missing = []
for trip in edges:
    try:
        edges_idx.append([index[kg + trip[0]], index[kg + trip[1]], trip[2], trip[3]])
    except KeyError as e:
        missing.append(str(e)[1:-1])


print(len(edges))
print(len(edges_idx))

edges_tensor = th.tensor(edges_idx)

# %%
edges_idx_refined = []
missing_refined = []
for trip in refined_edges:
    try:
        edges_idx_refined.append([index[kg + trip[0]], index[kg + trip[1]], trip[2], trip[3]])
    except KeyError as e:
        missing_refined.append(str(e)[1:-1])


print(len(refined_edges))
print(len(edges_idx_refined))

edges_tensor_refined = th.tensor(edges_idx_refined)

# %%
edges_idx_DMA30 = []
missing_DMA30 = []
for trip in edges_DMA30:
    try:
        edges_idx_DMA30.append([index[kg + trip[0]], index[kg + trip[1]], trip[2], trip[3]])
    except KeyError as e:
        missing_DMA30.append(str(e)[1:-1])


print(len(edges_DMA30))
print(len(edges_idx_DMA30))

edges_tensor_DMA30 = th.tensor(edges_idx_DMA30)

# %%
edges_idx_refined_DMA30 = []
missing_refined_DMA30 = []
for trip in refined_edges_DMA30:
    try:
        edges_idx_refined_DMA30.append([index[kg + trip[0]], index[kg + trip[1]], trip[2], trip[3]])
    except KeyError as e:
        missing_refined_DMA30.append(str(e)[1:-1])


print(len(refined_edges_DMA30))
print(len(edges_idx_refined_DMA30))

edges_tensor_refined_DMA30 = th.tensor(edges_idx_refined_DMA30)
# %%
with open(os.path.join(BASE, 'datasets/split_datasets/interactions.pkl'), 'wb') as fo:
    pickle.dump(edges_tensor, fo)

with open(os.path.join(BASE, 'datasets/split_datasets/refined_interactions.pkl'), 'wb') as fo:
    pickle.dump(edges_tensor_refined, fo)

with open(os.path.join(BASE, 'datasets/split_datasets/interactions_DMA30.pkl'), 'wb') as fo:
    pickle.dump(edges_tensor_DMA30, fo)

with open(os.path.join(BASE, 'datasets/split_datasets/refined_interactions_DMA30.pkl'), 'wb') as fo:
    pickle.dump(edges_tensor_refined_DMA30, fo)
# %%
print('done')
# %%
