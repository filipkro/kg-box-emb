# %%
import pandas as pd
import os
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# %%
def extract_genes(row):
    ga, gb = row['Query strain ID'].split('_')[0].split('+')
    gc = row['Array strain ID'].split('_')[0]

    g1, g2, g3 = sorted([ga, gb, gc])
    return g1, g2, g3
# %%
df = pd.read_csv(os.path.join(BASE, 'data/interaction_data/aao1729_data_s1.tsv'),
                 sep='\t')
# %%
df.drop(columns=['Query allele name', 'Combined mutant type',
                 'Raw genetic interaction score (epsilon)',
                 'Adjusted genetic interaction score (epsilon or tau)',
                 'Array allele name', 'Query single/double mutant fitness',
                 'Array single mutant fitness'], inplace=True)
# %%
df['g1'], df['g2'], df['g3'] = zip(*df.apply(extract_genes, axis=1))
# %%
df.sort_values(by='P-value', ascending=False, inplace=True)
df = df[df.duplicated(['g1', 'g2', 'g3'], keep='last')]
# %%
df.to_csv(os.path.join(BASE, 'interaction_data/unique_triples.tsv'), sep='\t')
# %%
