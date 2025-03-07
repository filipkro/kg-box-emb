import mowl
mowl.init_jvm("12g")
from mowl.datasets.el import ELDataset
from mowl.datasets.base import PathDataset
import torch as th
from sklearn.model_selection import train_test_split as sk_train_test_split

import pickle, copy

import rdflib
from rdflib.plugins.stores import sparqlstore
from rdflib.namespace import OWL, RDF, RDFS

def get_normalized_dataset(kg_fp, role_fp=None):
    if role_fp == None:
        role_fp = kg_fp

    data = PathDataset(kg_fp)
    el_dataset = ELDataset(data.ontology)
    el_dataset.load()
    gcis = el_dataset.get_gci_datasets()
    gci_dict = {k: gcis[k].data for k in gcis}

    role_properties = role_properties_in_graph(role_fp)

    sym_object_property_index_dict = {}
    object_property_index_dict = copy.deepcopy(el_dataset.object_property_index_dict)
    id_to_name = {v: k for k,v in object_property_index_dict.items()}
    

    gci2 = gci_dict['gci2']
    gci2_sym = []
    if role_properties[0]:
        sym = []
        for r in get_symmetric_roles(role_fp):
            sym_object_property_index_dict[r] = len(sym_object_property_index_dict)
            if r in object_property_index_dict:
                sym.append(object_property_index_dict[r])

        sym = th.tensor(sym)

        object_property_index_dict = copy.deepcopy(el_dataset.object_property_index_dict)
        # split gci2
        mask = th.isin(gci_dict['gci2'][:,1], sym)
        gci2 = gci_dict['gci2'][~mask]
        gci2_sym = gci_dict['gci2'][mask]

        # rename sym roles in gci2_sym
        map_to_sym_ids = {object_property_index_dict[k]: v for k,v in sym_object_property_index_dict.items() if k in object_property_index_dict}
        gci2_sym[:, 1].apply_(lambda x: map_to_sym_ids[x])

        # remove sym roles from dict, rehash, keep track of map
        map_to_new_ids = {i: i for i in range(len(object_property_index_dict))}
        for k in map_to_sym_ids:
            object_property_index_dict.pop(id_to_name[k])
            renamed_index = len(map_to_new_ids) - 1
            map_to_new_ids[renamed_index] = k
            try:
                object_property_index_dict[id_to_name[renamed_index]] = k
            except KeyError as e:
                print(k)
                print(renamed_index)
                print(id_to_name[renamed_index])
                print(object_property_index_dict[id_to_name[renamed_index]])
                raise KeyError(e)
            id_to_name.pop(k)
            map_to_new_ids.pop(k)

        # rename roles in gci2
        gci2[:, 1].apply_(lambda x: map_to_new_ids[x])

    # add subproperties to one of three gci0_role datasets (r subpropof s).
    # first, r and s are symmetric, second only r is symmetric, third neither
    # r nor s is symmetric
    # after thinking more about it, this probably doesn't hold, can also be the fourth case

    gci0_role = []
    gci0_sym_role = []
    gci0_nonsym_sym_role = []
    gci0_sym_nonsym_role = []

    if role_properties[1]:
        for r in get_subproperties(role_fp):
            if r[0] in sym_object_property_index_dict and \
                    r[1] in sym_object_property_index_dict:
                gci0_sym_role.append([sym_object_property_index_dict[r[0]],
                                    sym_object_property_index_dict[r[1]]])
            elif r[0] in sym_object_property_index_dict:
                if r[1] not in object_property_index_dict:
                    object_property_index_dict[r[1]] = len(object_property_index_dict)
                gci0_sym_nonsym_role.append([sym_object_property_index_dict[r[0]],
                                            object_property_index_dict[r[1]]])
            elif r[1] in sym_object_property_index_dict:
                if r[0] not in object_property_index_dict:
                    object_property_index_dict[r[0]] = len(object_property_index_dict)
                gci0_nonsym_sym_role.append([object_property_index_dict[r[0]],
                                            sym_object_property_index_dict[r[1]]])
            else:
                if r[0] not in object_property_index_dict:
                    object_property_index_dict[r[0]] = len(object_property_index_dict)
                if r[1] not in object_property_index_dict:
                    object_property_index_dict[r[1]] = len(object_property_index_dict)
                gci0_role.append([object_property_index_dict[r[0]],
                                object_property_index_dict[r[1]]])

    # disjoint
    # here only three datasets needed; both sym, one sym (flip so always the same),
    # and no sym
    # gci1_bot
    gci1_bot_sym_role = []
    gci1_bot_sym_nonsym_role = []
    gci1_bot_role = []

    if role_properties[2]:
        for r in get_disjoint_roles(role_fp):
            if r[0] in sym_object_property_index_dict and \
                    r[1] in sym_object_property_index_dict:
                gci1_bot_sym_role.append([sym_object_property_index_dict[r[0]],
                                        sym_object_property_index_dict[r[1]]])
            elif r[0] in sym_object_property_index_dict:
                if r[1] not in object_property_index_dict:
                    object_property_index_dict[r[1]] = len(object_property_index_dict)
                gci1_bot_sym_nonsym_role.append([sym_object_property_index_dict[r[0]],
                                                object_property_index_dict[r[1]]])
            elif r[1] in sym_object_property_index_dict:
                if r[0] not in object_property_index_dict:
                    object_property_index_dict[r[0]] = len(object_property_index_dict)
                gci1_bot_sym_nonsym_role.append([sym_object_property_index_dict[r[1]],
                                                object_property_index_dict[r[0]]])
            else:
                if r[0] not in object_property_index_dict:
                    object_property_index_dict[r[0]] = len(object_property_index_dict)
                if r[1] not in object_property_index_dict:
                    object_property_index_dict[r[1]] = len(object_property_index_dict)
                gci1_bot_role.append([object_property_index_dict[r[0]],
                                    object_property_index_dict[r[1]]])


    for role in get_all_roles(role_fp):
        if role not in object_property_index_dict:
            object_property_index_dict[role] = len(object_property_index_dict)
    # save as tensors to the dict, then this part should be done

    # merge sym gcis into gcis, save as pkl
    # one big index dict with the three different dicts and save as pkl

    gci_dict['gci2'] = gci2 if isinstance(gci2, th.Tensor) else th.tensor(gci2)
    gci_dict['gci2_sym'] = gci2_sym if isinstance(gci2_sym, th.Tensor) \
        else th.tensor(gci2_sym)

    gci_dict['gci0_roles'] = gci0_role if isinstance(gci0_role, th.Tensor) \
        else th.tensor(gci0_role)
    gci_dict['gci0_sym_roles'] = gci0_sym_role if isinstance(gci0_sym_role,
                                                             th.Tensor) \
        else th.tensor(gci0_sym_role)
    gci_dict['gci0_nonsym_sym_roles'] = gci0_nonsym_sym_role if \
        isinstance(gci0_nonsym_sym_role, th.Tensor) \
            else th.tensor(gci0_nonsym_sym_role)
    gci_dict['gci0_sym_nonsym_roles'] = gci0_sym_nonsym_role if \
        isinstance(gci0_sym_nonsym_role, th.Tensor) \
            else th.tensor(gci0_sym_nonsym_role)

    gci_dict['gci1_bot_roles'] = gci1_bot_role if isinstance(gci1_bot_role,
                                                             th.Tensor) \
        else th.tensor(gci1_bot_role)
    gci_dict['gci1_bot_sym_roles'] = gci1_bot_sym_role if \
        isinstance(gci1_bot_sym_role, th.Tensor) \
            else th.tensor(gci1_bot_sym_role)
    gci_dict['gci1_bot_sym_nonsym_roles'] = gci1_bot_sym_nonsym_role if \
        isinstance(gci1_bot_sym_nonsym_role, th.Tensor) \
            else th.tensor(gci1_bot_sym_nonsym_role)

    index_dict = {'class_index': el_dataset.class_index_dict,
                'property_index': object_property_index_dict,
                'sym_property_index': sym_object_property_index_dict}
    
    return gci_dict, index_dict
    
def save_normalized_datasets(kg_fp, gci_fp, index_fp, role_fp=None):
    gci_dict, index_dict = get_normalized_dataset(kg_fp, role_fp)
    
    with open(gci_fp, 'wb') as fo:
        pickle.dump(gci_dict, fo)
    with open(index_fp, 'wb') as fo:
        pickle.dump(index_dict, fo)

def remove_dataset(dataset, gci):
    dataset[gci] = th.tensor([])
    return dataset

def get_symmetric_roles(fp):
    g = rdflib.Graph()
    g.parse(fp)

    return [str(s) for s in g.subjects(RDF.type, OWL.SymmetricProperty)]

def get_subproperties(fp):
    g = rdflib.Graph()
    g.parse(fp)

    return [(str(s), str(o)) for s, o in g.subject_objects(RDFS.subPropertyOf)]

def get_disjoint_roles(fp):
    g = rdflib.Graph()
    g.parse(fp)

    return [(str(s), str(o)) for s, o in g.subject_objects(OWL.propertyDisjointWith)]

def get_all_roles(fp):
    g = rdflib.Graph()
    g.parse(fp)

    return [str(s) for s in g.subjects(RDF.type, OWL.ObjectProperty)]

def role_properties_in_graph(fp):
    g = rdflib.Graph()
    g.parse(fp)

    return [(None, RDF.type, OWL.SymmetricProperty) in g,
            (None, RDFS.subPropertyOf, None) in g,
            (None, OWL.propertyDisjointWith, None) in g]

def concat_datasets(data1, data2):
    assert data1.keys() == data2.keys()
    return_data = {}
    for k in data1:
        return_data[k] = th.cat((data1[k], data2[k]), 0)
    return return_data
    


def reindex_dataset_gci2(data, from_index, to_index,
                    consider_nfs=['gci2', 'gci2_sym']):
    if consider_nfs != ['gci2', 'gci2_sym']:
        raise NotImplementedError("Evaluation only implemented for gci2")
    if consider_nfs:
        for k in data:
            if k not in consider_nfs:
                data[k] = th.tensor([])
    class_map = {}
    for x in from_index['class_index']:
        class_map[from_index['class_index'][x]] = to_index['class_index'][x]
    role_map = {}
    for x in from_index['property_index']:
        role_map[from_index['property_index'][x]] = to_index['property_index'][x]
    sym_role_map = {}
    for x in from_index['sym_property_index']:
        sym_role_map[from_index['sym_property_index'][x]] = to_index['sym_property_index'][x]

    for k in data:
        if len(data[k]) == 0:
            continue
        # currently only for gci2 and gci2_sym
        r_map = role_map if k == 'gci2' else sym_role_map
        gci2 = data[k]
        gci2[:, 0].apply_(lambda x: class_map[x])
        gci2[:, 2].apply_(lambda x: class_map[x])
        gci2[:, 1].apply_(lambda x: r_map[x])
        data[k] = gci2

    return data

def reindex_dataset(data, from_index, to_index):

    ignore_gcis = ['gci1', 'gci3', 'gci3_bot', 'gci0_bot']
    class_map = {}
    for x in from_index['class_index']:
        class_map[from_index['class_index'][x]] = to_index['class_index'][x]
    role_map = {}
    for x in from_index['property_index']:
        role_map[from_index['property_index'][x]] = to_index['property_index'][x]
    sym_role_map = {}
    for x in from_index['sym_property_index']:
        sym_role_map[from_index['sym_property_index'][x]] = to_index['sym_property_index'][x]

    for k in data:
        if len(data[k]) == 0 or k in ignore_gcis:
            continue
        # currently only for gci2 and gci2_sym
        
        if 'roles' in k:
            s0 = ['gci0_sym_roles', 'gci1_bot_sym_roles',
                  'gci0_sym_nonsym_roles']
            ns1 = ['gci0_roles', 'gci1_bot_roles', 'gci0_sym_nonsym_roles']
            m0 = sym_role_map if k in s0 else role_map
            m1 = role_map if k in ns1 else sym_role_map
            gci = data[k]
            gci[:, 0].apply_(lambda x: m0[x])
            gci[:, 1].apply_(lambda x: m1[x])
            data[k] = gci
        elif 'gci2' in k:
            r_map = sym_role_map if 'sym' in k else role_map
            gci2 = data[k]
            gci2[:, 0].apply_(lambda x: class_map[x])
            gci2[:, 2].apply_(lambda x: class_map[x])
            gci2[:, 1].apply_(lambda x: r_map[x])
            data[k] = gci2
        elif k == 'gci0' or k == 'gci1_bot':
            gci = data[k]
            gci[:, 0].apply_(lambda x: class_map[x])
            gci[:, 1].apply_(lambda x: class_map[x])
        else:
            raise NotImplementedError(k)

    return data


def train_test_split(data, train_split=0.7, consider_nfs=['gci2', 'gci2_sym'],
                     seed=0):
    if consider_nfs != ['gci2', 'gci2_sym']:
        raise NotImplementedError("Evaluation only implemented for gci2")
    train_data = {}
    test_data = {}
    for k in data:
        if k not in consider_nfs or len(data[k]) == 0:
            train_data[k] = th.tensor([])
            test_data[k] = th.tensor([])
            continue

        train, test = sk_train_test_split(data[k], train_size=train_split,
                                          random_state=seed)
        train_data[k] = train
        test_data[k] = test

    return train_data, test_data


def get_subClassOf(parent, kg_endpoint='http://localhost:3030/kg'):
    sp_store = sparqlstore.SPARQLStore(kg_endpoint)
    kg = rdflib.Graph(store=sp_store)
    q = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?a
        WHERE {{
            ?a rdfs:subClassOf* <{parent}> .
            }}"""
    
    res = kg.query(q)

    return list(res)

class ParentDicts:
    def __init__(self, index, data, kg_endpoint='http://localhost:3030/kg'):
        sp_store = sparqlstore.SPARQLStore(kg_endpoint)
        self.kg = rdflib.Graph(store=sp_store)
        self.index = index
        self.rev_index = {ds: {v: k for k, v in index[ds].items()} for ds in index}
        self.data = data
        self._class_parents = {}

        self._parents_found = False

    def set_class_parents(self, parents):
        self._class_parents = parents
        self._parents_found = True

    @property
    def class_parents(self):
        if not self._parents_found:
            self.find_all_class_parents()
        return self._class_parents

    def get_parent_query(self, top_parents):
        if isinstance(top_parents, str):
            return f"\n?a rdfs:subClassOf* <{top_parents}> ."
        elif isinstance(top_parents, list) and len(top_parents) > 0:
            if len(top_parents) == 1:
                return self.get_parent_query(top_parents[0])
            qs = [f'{{?a rdfs:subClassOf* <{a}> }}' for a in top_parents]
            return '\n' + ' UNION '.join(qs) + ' .'
        elif top_parents == None:
            return ""
        else:
            raise ValueError("top_parent is expected to be a string, list of strings, or None")

    def find_all_class_parents(self, top_parent=None):
        all_classes = self.data['gci0'].flatten()
        print(f"Len gci0: {len(all_classes)}")
        par_q = self.get_parent_query(top_parent)

        for i, d in enumerate(all_classes):
            if i % 100000 == 0:
                print(len(self._class_parents))
            if d.item() not in self._class_parents:
                q = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?a
    WHERE {{
        <{self.rev_index['class_index'][d.item()]}> rdfs:subClassOf* ?a .{par_q}
        }}"""
                self._class_parents[d.item()] = {self.index['class_index'][str(e[0])]
                                for e in self.kg.query(q)
                                if type(e[0]) == rdflib.term.URIRef}
                
        self._parents_found = True

class ExampleDicts(ParentDicts):
    def __init__(self, r_fp, index, data,
                 kg_endpoint='http://localhost:3030/kg'):
        super.__init__(index, data, kg_endpoint=kg_endpoint)
        self.roles = rdflib.Graph()
        self.roles.parse(r_fp)
        self.role_parents = {'gci2': {}, 'gci2_sym': {}}

        self._roles_found = False
        self._positive_found = False
        self._positive_found_rev = False
        self._negative_found = False

    def set_role_parents(self, parents):
        self.role_parents = parents
        self._roles_found = True

    def set_positive_examples(self, examples):
        self.positive_dict = examples
        self._positive_found = True

    def set_positive_rev_examples(self, examples):
        self.positive_dict_rev = examples
        self._positive_found_rev = True
                
    def find_all_role_parents(self):
        
        print(f"len of nonsym gci2: {len(self.data['gci2'])}")
        if len(self.data['gci2']) > 0:
            for d in self.data['gci2'][:,1]:
                if d.item() not in self.role_parents['gci2']:
                    q = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?a
                WHERE {{
                    <{self.rev_index['property_index'][d.item()]}> rdfs:subPropertyOf* ?a .
                    }}"""
                    self.role_parents['gci2'][d.item()] = set()
                    for e in self.roles.query(q):
                        if type(e[0]) == rdflib.term.URIRef:
                            if (e[0], RDF.type, OWL.SymmetricProperty) in self.roles:
                                self.role_parents['gci2'][d.item()].add((self.index['sym_property_index'][str(e[0])], True))
                            else:
                                self.role_parents['gci2'][d.item()].add((self.index['property_index'][str(e[0])], False))

        print(f"len of sym gci2: {len(self.data['gci2_sym'])}")
        if len(self.data['gci2_sym']) > 0:
            for d in self.data['gci2_sym'][:,1]:
                if d.item() not in self.role_parents['gci2_sym']:
                    q = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?a
                WHERE {{
                    <{self.rev_index['sym_property_index'][d.item()]}> rdfs:subPropertyOf* ?a .
                    }}"""
                    self.role_parents['gci2_sym'][d.item()] = set()
                    for e in self.roles.query(q):
                        if type(e[0]) == rdflib.term.URIRef:
                            if (e[0], RDF.type, OWL.SymmetricProperty) in self.roles:
                                self.role_parents['gci2_sym'][d.item()].add((self.index['sym_property_index'][str(e[0])], True))
                            else:
                                self.role_parents['gci2_sym'][d.item()].add((self.index['property_index'][str(e[0])], False))

        self._roles_found = True


    def find_positive_examples_rev(self):
        if not self._parents_found:
            self.find_all_class_parents()
        if not self._roles_found:
            self.find_all_role_parents()
        return_dict = {'gci2': {}, 'gci2_sym': {}, 'gci2_focus': {},
                       'gci2_sym_focus': {}}
        
        for k in return_dict:
            print(f'\n{k}')
            if k not in self.data:
                continue
            for i, row in enumerate(self.data[k]):
                if i % 10000:
                    print(f"{i / len(self.data[k])} done", end='\r')
                class_parents = copy.deepcopy(self._class_parents[row[0].item()])
                role_type = 'gci2_sym' if 'sym' in k else 'gci2'
                for p, sym in self.role_parents[role_type][row[1].item()]:
                    c = 'gci2_sym' if sym else 'gci2'
                    if 'focus' in k:
                        c = c + '_focus'
                    if (row[2].item(), p) in return_dict[c]:
                        return_dict[c][(row[2].item(), p)].update(class_parents)
                    else:
                        return_dict[c][(row[2].item(), p)] = class_parents

        self.positive_dict_rev = return_dict
        self._positive_found_rev = True


    def find_positive_examples(self):
        if not self._parents_found:
            self.find_all_class_parents()
        if not self._roles_found:
            self.find_all_role_parents()
        return_dict = {'gci2': {}, 'gci2_sym': {}, 'gci2_focus': {},
                       'gci2_sym_focus': {}}
        for k in return_dict:
            print(f'\n{k}')
            if k not in self.data:
                continue
            for i, row in enumerate(self.data[k]):
                if i % 10000:
                    print(f"{i / len(self.data[k])} done", end='\r')
                class_parents = copy.deepcopy(self._class_parents[row[2].item()])
                role_type = 'gci2_sym' if 'sym' in k else 'gci2'
                for p, sym in self.role_parents[role_type][row[1].item()]:
                    c = 'gci2_sym' if sym else 'gci2'
                    if 'focus' in k:
                        c = c + '_focus'
                    if (row[0].item(), p) in return_dict[c]:
                        return_dict[c][(row[0].item(), p)].update(class_parents)
                    else:
                        return_dict[c][(row[0].item(), p)] = class_parents

        self.positive_dict = return_dict
        self._positive_found = True

    def get_positive_examples(self):
        if not self._positive_found:
            self.find_positive_examples()

        return self.positive_dict
    
    def get_positive_rev_examples(self):
        if not self._positive_found_rev:
            self.find_positive_examples_rev()

        return self.positive_dict

    def find_negative_examples(self):
        if not self._positive_found:
            self.find_positive_examples()
        neg_examples = {'gci2': {}, 'gci2_sym': {}}
        all_classes = self.index['class_index'].values()
        for gci in neg_examples:
            print('\n' + gci)
            full_len = len(self.positive_dict[gci])
            for i, (k, pos) in enumerate(self.positive_dict[gci].items()):
                print(f"{i / full_len} done", end='\r')
                neg_examples[gci][k] = [a for a in all_classes if not a in pos]

        self.negative_dict = neg_examples
        self._negative_found = True

    def get_negative_examples(self):
        if not self._negative_found:
            self.find_negative_examples()

        return self.negative_dict

    def find_all_parents(self):
        self.find_all_class_parents()
        self.find_all_role_parents()