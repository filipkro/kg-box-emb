# %%
import pickle, os
import torch as th
import rdflib
from rdflib.plugins.stores import sparqlstore

# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE)

graph_dir = os.path.join(BASE, 'graphs/split_graphs')
dataset_dir = os.path.join(BASE, 'datasets/split_datasets')

sp_store = sparqlstore.SPARQLStore('http://localhost:3030/kg')
kg = rdflib.Graph(store=sp_store)
for p in os.listdir(graph_dir):
    dataset_name = p.split('.')[0]

    with open(os.path.join(dataset_dir, f'collected_{dataset_name}.pkl'),
              'rb') as fi:
        dataset = pickle.load(fi)

    print(len(dataset.training_datasets.gci1_bot_dataset.data))

    new_bots = set()
    i2c = {v:k for k,v in dataset.class_to_id.items()}
    c2i = dataset.class_to_id
    print(dataset_name)
    if len(dataset.training_datasets.gci1_bot_dataset.data > 0):
        for i, tensor_pair in enumerate(dataset.training_datasets.gci1_bot_dataset.data[:,:2]):
            print(f"{i} of {len(dataset.training_datasets.gci1_bot_dataset.data)} completed...", end="\r")
            pair = tuple(sorted(tensor_pair.tolist()))
            if not pair in new_bots:
                a = i2c[pair[0]]
                b = i2c[pair[1]]
                q = f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT DISTINCT ?a ?b
                WHERE {{
                    ?a rdfs:subClassOf{{,5}} <{a}> .
                    ?b rdfs:subClassOf{{,5}} <{b}> .
                    }} LIMIT 50000"""
                
                res = kg.query(q)
                for r in res:
                    bot_pair = tuple(sorted([c2i[str(r[0])], c2i[str(r[1])]]))
                    new_bots.add(bot_pair)

        bot_class = dataset.training_datasets.gci1_bot_dataset.data[0,2].item()
        tensor_bots = th.tensor(list(new_bots), dtype=th.int32)
        new_dataset = th.empty((2*len(tensor_bots), 3), dtype=th.int32)
        new_dataset[:len(tensor_bots), :2] = tensor_bots
        new_dataset[len(tensor_bots):, 0] = tensor_bots[:,1]
        new_dataset[len(tensor_bots):, 1] = tensor_bots[:,0]
        new_dataset[:,2] = th.ones(new_dataset[:,2].shape, dtype=th.int32) * bot_class

        dataset.training_datasets.gci1_bot_dataset._data = new_dataset

# %%
    with open(os.path.join(dataset_dir, f'collected_{dataset_name}_w_bot.pkl'),
              'wb') as fo:
        pickle.dump(dataset, fo)
