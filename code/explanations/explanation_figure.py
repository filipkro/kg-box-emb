# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
# %%
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# %%
df = pd.read_csv(os.path.join(BASE, 'src/prel_explanationsDMA30-InputXGradient-full_model-feb02-10000-cleaned-top10000.tsv'), sep = '\t', index_col = 0)
# %%
filtered_df = df[
    df['class1'].str.contains("APO_0000096", na=False) | # General nutrient utilization
    df['class2'].str.contains("APO_0000096", na=False) | # General nutrient utilization
    df['class1'].str.contains("APO_0000097", na=False) | # Auxotrophy
    df['class2'].str.contains("APO_0000097", na=False) | # Auxotrophy
    df['class1'].str.contains("APO_0000099", na=False) | # Utilization of Nitrogen Source
    df['class2'].str.contains("APO_0000099", na=False) | # Utilization of Nitrogen Source
    df['class1'].str.contains("APO_0000100", na=False) | # Nutrient uptake
    df['class2'].str.contains("APO_0000100", na=False) | # Nutrient uptake
    df['class1'].str.contains("APO_0000125", na=False) | # Utilization of phosphorous Source
    df['class2'].str.contains("APO_0000125", na=False) | # Utilization of phosphorous Source
    df['class1'].str.contains("APO_0000219", na=False) | # utilization of sulfur source
    df['class2'].str.contains("APO_0000219", na=False) | # utilization of sulfur source
    df['rel1'].str.contains("Nutrient", na=False) # General nutrient utilization edge from SGD?
]
# %% Decide on the experimental measurable
'''
In the first case, the interaction of interest would be the combination between a regulator and nutrient uptake. 
The experimental outcome would then be an extracellular measurement over time, and the measurable would be the rate at which it is consumed. 
E.g. Genes that are {qualifier} regulated by X have {qualifier} utilization of chemical Y. 
Could be interesting to do overexpression and test. If we do a knockout, it is just straight up part of the dataset the phenotypes are derived from. 
So maybe not interesting in the interest of time.
'''

experiment_df = filtered_df[filtered_df['class1'].str.contains(
                    "http://sgd-kg.project-genesis.io#Y", na=False) |
                            filtered_df['class2'].str.contains(
                    "http://sgd-kg.project-genesis.io#Y", na=False)]

'''
In the second case, the interaction of interest would be the combination between a measurable phenotype and nutrient uptake. 
A bit broad, but involves chemical perturbants, or things related to, for example, fitness. E.g. Decreased uptake of chemical Y is connected to decreased chronological lifespan
Potentially tested by providing the nutrient in different concentrations and measuring the response.
'''

experiment_df = filtered_df[(filtered_df['class1'].str.contains("CHEBI", na=False) |
                            filtered_df['class2'].str.contains("CHEBI", na=False)) & 
                            (~filtered_df['rel1'].str.contains("reg", na=False))]

'''
In the third case, the interaction of interest would be the combination between sensitivity to a specific chemical perturbant and nutrient uptake. Just an additional filter on the previous category.
Potentially tested by providing the nutrient in different concentrations (for auxotrophic strains?) and perturbing it via some chemical, then measuring the response.
'''

experiment_df = filtered_df[(filtered_df['class1'].str.contains("CHEBI", na=False) &
                            filtered_df['class2'].str.contains("CHEBI", na=False)) & 
                            (~filtered_df['rel1'].str.contains("reg", na=False))]

# %% Let's plot the top X in a barplot
experiment_df['identifier'] = experiment_df['rel1'] + experiment_df['class1'] \
    + experiment_df['rel2'] + experiment_df['class2']

plt.figure(figsize=(4,5))
palette = ['#bec3cf' if i != 2 else '#f03a26' for i in range(10)]
sns.barplot(y='identifier', x='imp', data=experiment_df.iloc[:10,:],
            palette=palette)

locs, labels = plt.yticks()
plt.yticks(locs, labels=[f'f{f}' for f in locs], fontsize=11)
plt.xticks([0.001, 0.002, 0.003],fontsize=11)
plt.xlabel('')
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'paper/figs/feature_importance.eps'),
            format='eps', bbox_inches='tight')
plt.show()

# %%
lt = experiment_df.iloc[:10,:]
lt['class1'] = lt['class1'].apply(lambda x: x.split('/')[-1])
lt['class2'] = lt['class2'].apply(lambda x: x.split('/')[-1].split('#')[-1])
lt['rel1'] = lt['rel1'].apply(lambda x: x.split('rev_')[-1])
lt['rel2'] = lt['rel2'].apply(lambda x: x.split('rev_')[-1])
lt.drop(['occ', 'individual_imp', 'identifier'], axis=1, inplace=True)
print(lt.to_latex())
# %%
