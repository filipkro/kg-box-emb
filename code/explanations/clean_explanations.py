# %%
import pandas as pd
# %%
exp_name = 'prel_explanationsDMA30-InputXGradient-full_model-feb02-10000.tsv'
df = pd.read_csv(exp_name, sep='\t')
print(df.shape)
print(df['occ'].max())
# %%
df = df[df['class1'] != 'http://purl.obolibrary.org/obo/APO_0000113']
df = df[df['class2'] != 'http://purl.obolibrary.org/obo/APO_0000113']
df = df[df['class1'] != 'http://purl.obolibrary.org/obo/APO_0000112']
df = df[df['class2'] != 'http://purl.obolibrary.org/obo/APO_0000112']
df = df[~df['class1'].str.contains('APO_0000110')]
df = df[~df['class2'].str.contains('APO_0000110')]
print(df.shape)
# %%
print(df['occ'].max())
df.drop(['Unnamed: 0', 'idx', 'rel', 'dom'], axis=1, inplace=True)
# %%
df['individual_imp'] = df['imp'] / df['occ']
df_ind = df.sort_values(by='individual_imp', ascending=False, inplace=False)
# %%
exp_name.split('.tsv')[0] + '-cleaned.tsv'
df.to_csv(exp_name.split('.tsv')[0] + '-cleaned.tsv', sep='\t')
df.iloc[:10000].to_csv(exp_name.split('.tsv')[0] + '-cleaned-top10000.tsv',
                       sep='\t')
df_ind.iloc[:10000].to_csv(exp_name.split('.tsv')[0] +
                           '-cleaned-top10000ind.tsv', sep='\t')
# %%
df_ind[df_ind['occ'] > 5].iloc[:10000].to_csv(exp_name.split('.tsv')[0] +
                                              '-cleaned-top10000ind_min5.tsv',
                                              sep='\t')
# %%
