# %%
import os, pickle
from utils import dataset_utils
from datasets import CollectedDatasets
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# %%
graph_dir = os.path.join(BASE, 'graphs/split_graphs')
dataset_dir = os.path.join(BASE, 'datasets/split_datasets')

top_parent_dict = {'genes': 'http://biocyc.project-genesis.io#All-Genes',
                   'mat_ent': 'http://purl.obolibrary.org/obo/BFO_0000040',
                   'cell_comp': 'http://purl.obolibrary.org/obo/GO_0005575',
                   'reactions': 'http://biocyc.project-genesis.io#Generalized-Reactions',
                   'reguls': 'http://purl.obolibrary.org/obo/INO_0000002',
                   'bio_proc': 'http://purl.obolibrary.org/obo/GO_0008150',
                   'mol_func': 'http://purl.obolibrary.org/obo/GO_0003674',
                   'quality': 'http://purl.obolibrary.org/obo/BFO_0000019',
                   'root': 'http://purl.obolibrary.org/obo/NCBITaxon_1'}
for p in os.listdir(graph_dir):
    dataset_name = p.split('.')[0]
    print(f'Generating {dataset_name}')
    full_data, full_index = dataset_utils.get_normalized_dataset(os.path.join(graph_dir, p))
    dataset_dicts = dataset_utils.ParentDicts(full_index, full_data)
    dataset_dicts.find_all_class_parents(top_parent=top_parent_dict[dataset_name])
    print('saving data')
    with open(os.path.join(dataset_dir, f'{dataset_name}.pkl'), 'wb') as fo:
        pickle.dump({'train': full_data, 'index': full_index,
                 'class_parents': dataset_dicts.class_parents}, fo)
    dataset = CollectedDatasets(train_dict=full_data, index_dict=full_index,
                                train_class_parents=dataset_dicts.class_parents)
    with open(os.path.join(dataset_dir,
                           f'collected_{dataset_name}.pkl'), 'wb') as fo:
        pickle.dump(dataset, fo)
