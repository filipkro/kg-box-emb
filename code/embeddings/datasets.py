import torch as th
from torch.utils.data import Dataset
import pickle, warnings
import random

class CollectedDatasets:
    def __init__(self, train_fp=None, train_dict=None, index_fp=None,
                 index_dict=None, val_fp=None, val_dict=None, test_ds=None,
                 device='cpu', train_pos_ex=None, train_pos_ex_rev=None,
                 train_class_parents=None, dtype=th.int32):
        if train_fp == None and train_dict == None:
            raise ValueError('Either a filepath to a pickle file (train_fp) '
                             'or a dataset dictionary (train_dict) for the '
                             'training dataset needs to be provided.')
        if train_fp != None and train_dict != None:
            warnings.warn('Both filepath and dictionary was provided for '
                          'training data. Dictionary will be used.')
        if index_fp == None and index_dict == None:
            raise ValueError('Either a filepath to a pickle file (index_fp) '
                             'or a dataset dictionary (index_dict) for the '
                             'index dictionaries needs to be provided.')
        if index_fp != None and index_dict != None:
            warnings.warn('Both filepath and dictionary was provided for '
                          'index dictionaries. Dictionary will be used.')
        if train_dict:
            training_datasets = train_dict
        else:
            with open(train_fp, 'rb') as fi:
                training_datasets = pickle.load(fi)
        if index_dict:
            index_dicts = index_dict
        else:
            with open(index_fp, 'rb') as fi:
                index_dicts = pickle.load(fi)

        self.device = device
        self.dtype = dtype
        self.class_to_id = index_dicts['class_index']
        self.object_property_to_id = index_dicts['property_index']
        self.sym_object_property_to_id = index_dicts['sym_property_index']
        self.training_datasets = ELNormalizedDataset(training_datasets,
                                                     self.class_to_id,
                                                     self.object_property_to_id,
                                                     self.sym_object_property_to_id,
                                                     device=self.device,
                                                     pos_examples=train_pos_ex,
                                                     pos_rev_examples=train_pos_ex_rev,
                                                     class_parents=train_class_parents,
                                                     dtype=dtype)
        # if train_pos_ex:
        #     self.training_datasets.set_pos_examples(train_pos_ex)
        if val_fp != None and val_dict != None:
            warnings.warn('Both filepath and dictionary was provided for '
                          'validation data. Dictionary will be used.')
        if val_dict:
            self.validation_datasets = ELNormalizedDataset(val_dict,
                                                           self.class_to_id,
                                                           self.object_property_to_id,
                                                           self.sym_object_property_to_id,
                                                           device=self.device,
                                                           dtype=dtype)
        elif val_fp:
            with open(val_fp, 'rb') as fi:
                validation_datasets = pickle.load(fi)
                self.validation_datasets = ELNormalizedDataset(validation_datasets,
                                                               self.class_to_id,
                                                               self.object_property_to_id,
                                                               self.sym_object_property_to_id,
                                                               device=self.device,
                                                               dtype=dtype)
        else:
            self.validation_datasets = None
        if test_ds:
            raise NotImplementedError('Test datasets not implemented')
            with open(test_ds, 'rb') as fi:
                testing_datasets = pickle.load(fi)
            self.testing_datasets = ELNormalizedDataset(testing_datasets,
                                                     self.class_to_id,
                                                     self.object_property_to_id,
                                                     self.sym_object_property_to_id,
                                                     device=self.device)
        else:
            self.testing_datasets = None
        # self._evaluation_classes = set(("http://www.semanticweb.org/filipkro/ontologies/2023/10/untitled-ontology-91#A", "http://www.semanticweb.org/filipkro/ontologies/2023/10/untitled-ontology-91#C"))
        self._evaluation_classes = set(['http://purl.obolibrary.org/obo/GO_0005575',
                                        'http://purl.obolibrary.org/obo/GO_0008150',
                                        'http://purl.obolibrary.org/obo/GO_0009987',
                                        'http://purl.obolibrary.org/obo/GO_0042770',
                                        'http://purl.obolibrary.org/obo/GO_0051716',
                                        'http://purl.obolibrary.org/obo/GO_0061709',
                                        'http://purl.obolibrary.org/obo/GO_0071072',
                                        'http://purl.obolibrary.org/obo/GO_0071595'])
        

    def set_device(self, device):
        self.device = device

        self.training_datasets.set_device(self.device)
        if self.validation_datasets:
            self.validation_datasets.set_device(self.device)
        if self.testing_datasets:
            self.testing_datasets.set_device(self.device)


        
    def get_negative_examples(self, dataset, nbr_of_ex, rev=False):
        # assert dataset in ['gci2', 'gci2_sym']
        assert "gci2" in dataset or dataset == 'gci0'

        if dataset == 'gci0':
            try:
                parents = self.training_datasets.class_parents
                par_exists = True
            except AttributeError:
                par_exists = False

            gci_data = self.training_datasets.gci0_dataset
            neg_data = gci_data[random.choices(range(len(gci_data)), k=nbr_of_ex)]

            if par_exists:
                for i, row in enumerate(neg_data):
                    neg = row[1].item()
                    v = row[0].item()
                    while neg in parents[v]:
                        neg = random.choices(range(len(self.class_to_id)), k=1)[0]
                    neg_data[i, 1] = neg
            else:
                neg_data[:, 1] = th.tensor(random.choices(range(len(gci_data)), k=nbr_of_ex))
        elif 'gci2' in dataset:
            try:
                if rev:
                    pos_ex = self.training_datasets.positive_rev_examples
                    pos_exists = True
                else:
                    pos_ex = self.training_datasets.positive_examples
                    pos_exists = True
            except AttributeError:
                pos_exists = False

            if pos_exists:
                # sample #nbr_of_ex from pos example keys and one for each from
                # class indices, resample if ex in positive ex
                
                # HERE SOMETHING NEEDS TO BE DONE FOR FOCUS DATASET??

                pos = pos_ex[dataset]
                neg_keys = random.choices(list(pos), k=nbr_of_ex)
                neg_vals = random.choices(range(len(self.class_to_id)), k=nbr_of_ex)
                neg_data = th.zeros(nbr_of_ex, 3)
                for i, (k, v) in enumerate(zip(neg_keys, neg_vals)):
                    while v in pos[k]:
                        v = random.choices(range(len(self.class_to_id)), k=1)[0]
                    if rev:
                        neg_data[i, 0] = v
                        neg_data[i, 1:] = th.tensor([k[1], k[0]])
                    else:
                        neg_data[i, :2] = th.tensor(k)
                        neg_data[i, 2] = v
            else:
                # just sample #nbr_of_ex from current dataset and #nbr_of_ex from
                # class indices 

                # HERE SOMETHING NEEDS TO BE DONE FOR FOCUS DATASET probably
                gci_data = self.training_datasets.gci2_sym_dataset \
                    if 'sym' in dataset else self.training_datasets.gci2_dataset
                        
                neg_data = gci_data[random.choices(range(len(gci_data)), k=nbr_of_ex)]
                neg_vals = th.tensor(random.choices(range(len(self.class_to_id)), k=nbr_of_ex))
                if rev:
                    neg_data[:,0] = neg_vals
                else:
                    neg_data[:,2] = neg_vals
        else:
            raise NotImplementedError("Negative data only implemented for gci0 "
                                      "and gci2")

        return neg_data.to(self.dtype).to(self.device)
        
    def set_validation_dataset(self, val_dict=None, val_fp=None):
        if val_dict == None and val_fp == None:
            raise ValueError('Either a filepath to a pickle file (val_fp) '
                             'or a dataset dictionary (val_dict) for the '
                             'validation dataset needs to be provided.')
        if val_dict != None and val_fp != None:
            warnings.warn('Both filepath and dictionary was provided for '
                          'validation data. Dictionary will be used.')
        
        if val_dict:
            val = val_dict
        elif val_fp:
            with open(val_fp, 'rb') as fi:
                val = pickle.load(fi)
        self.validation_datasets = ELNormalizedDataset(val, self.class_to_id,
                                                       self.object_property_to_id,
                                                       self.sym_object_property_to_id,
                                                       device=self.device)
                
    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation
        """

        if self._evaluation_classes is None:
            raise NotImplementedError

        return self._evaluation_classes

    def get_evaluation_property(self):
        return "http://www.semanticweb.org/filipkro/ontologies/2023/10/untitled-ontology-91#r"



class GCIDataset(Dataset):
    def __init__(self, data, class_index_dict, object_property_index_dict=None,
                 sym_object_property_index_dict=None, device="cpu",
                 dtype=th.int32):
        super().__init__()
        self.class_index_dict = class_index_dict
        self.object_property_index_dict = object_property_index_dict
        self.sym_object_property_index_dict = sym_object_property_index_dict
        self.device = device
        self.dtype = dtype
        self._data = data
        self.push_to_device()

    @property
    def data(self):
        return self._data

    def set_device(self, device):
        self.device = device
        self.push_to_device()

    def push_to_device(self):

        self._data = self._data.to(self.dtype).to(self.device)
    

    def get_data(self):
        raise NotImplementedError()

    def extend_from_indices(self, other):
        if isinstance(other, list):
            tensor = th.tensor(other, device=self.device)
        elif th.is_tensor(other):
            tensor = other
        else:
            raise TypeError("Extending element must be either a list or a Pytorch tensor.")

        assert self.data.shape[1:] == tensor.shape[1:], "Tensors must have the same shape except \
        in the first dimension."

        tensor_elems = th.unique(tensor, return_counts=False, sorted=True)
        in_indices = sum(tensor_elems == i for i in list(self.class_index_dict.values())).bool()

        if not all(in_indices):
            raise ValueError("Extending element contains not recognized index.")

        new_tensor = th.cat([self._data, tensor.to(self.device)], dim=0)
        self._data = new_tensor

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class ELNormalizedDataset():
    """This class provides data-related methods to work with :math:`\mathcal{EL}` description \
    logic language. In general, it receives an ontology, normalizes it into 4 or 7 \
    :math:`\mathcal{EL}` normal forms and returns a :class:`torch.utils.data.Dataset` per normal \
    form. In the process, the classes and object properties names are mapped to an integer values \
    to create the datasets and the corresponding dictionaries can be input or created from scratch.

    :param gcis: Input ontology normalized into :math:`\mathcal{EL}` normal forms as a dictionary with GCIDatasets
    :type gcis: :dict
    :param extended: If true, the normalization process will return 7 normal forms. If false, \
    only 4 normal forms. See :doc:`/embedding_el/index` for more information. Defaults to \
    ``True``.
    :type extended: bool, optional
    :param class_index_dict: Dictionary containing information `class name --> index`. If not \
    provided, a dictionary will be created from the ontology classes. Defaults to ``None``.
    :type class_index_dict: dict
    :param object_property_index_dict: Dictionary containing information `object property \
    name --> index`. If not provided, a dictionary will be created from the ontology object \
    properties. Defaults to ``None``.
    :type object_property_index_dict: dict
    """

    def __init__(
        self,
        gcis,
        class_index_dict,
        object_property_index_dict,
        sym_object_property_index_dict,
        extended=True,
        device="cpu",
        pos_examples=None,
        pos_rev_examples=None,
        class_parents=None,
        dtype=th.int32
    ):
        # instead of ontology should propably be dict of the different GCI datasets, index dicts are required
        # all these should come from a loaded ELDataset (ie, ELDataset where load() has been called)

        if not isinstance(class_index_dict, dict):
            raise TypeError("Parameter class_index_dict must be of type dict")

        obj = object_property_index_dict
        if not isinstance(obj, dict):
            raise TypeError("Parameter object_property_index_dict must be of type dict")

        if not isinstance(extended, bool):
            raise TypeError("Optional parameter extended must be of type bool")

        if not isinstance(device, str):
            raise TypeError("Optional parameter device must be of type str")

        # self._ontology = ontology
        self._loaded = True
        self._extended = extended
        self._class_index_dict = class_index_dict
        self._object_property_index_dict = object_property_index_dict
        self.device = device
        self._extended = True

        self._gci0_dataset = GCI0Dataset(gcis["gci0"], class_index_dict, object_property_index_dict, device=self.device, dtype=dtype)
        self._gci1_dataset = GCI1Dataset(gcis["gci1"], class_index_dict, object_property_index_dict, device=self.device, dtype=dtype)



        if 'gci2_sym' in gcis:
            self._gci2_sym_dataset = GCI2Dataset(gcis['gci2_sym'],
                                                 class_index_dict,
                                                 object_property_index_dict,
                                                 sym_object_property_index_dict,
                                                 device=self.device, dtype=dtype)
        else:
            self._gci2_sym_dataset = GCI2Dataset(th.tensor([]), class_index_dict,
                                                 object_property_index_dict,
                                                 sym_object_property_index_dict,
                                                 device=self.device, dtype=dtype)
        if 'gci2_sym_focus' in gcis:
            self._gci2_sym_focus_dataset = GCI2Dataset(gcis['gci2_sym_focus'],
                                                       class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        else:
            self._gci2_sym_focus_dataset = GCI2Dataset(th.tensor([]), class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        if 'gci2_focus' in gcis:
            self._gci2_focus_dataset = GCI2Dataset(gcis["gci2_focus"],
                                                   class_index_dict, object_property_index_dict,
                                                   device=self.device, dtype=dtype)
        else:
            self._gci2_focus_dataset = GCI2Dataset(th.tensor([]),
                                                   class_index_dict,
                                                   object_property_index_dict,
                                                   device=self.device, dtype=dtype)
        if 'gci0_roles' in gcis:
            self._gci0_role_dataset = GCI0RoleDataset(gcis['gci0_roles'], class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        else:
            self._gci0_role_dataset = GCI0RoleDataset(th.tensor([]), class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        if 'gci0_sym_roles' in gcis:
            self._gci0_sym_role_dataset = GCI0RoleDataset(gcis['gci0_sym_roles'], class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        else:
            self._gci0_sym_role_dataset = GCI0RoleDataset(th.tensor([]), class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        if 'gci0_nonsym_sym_roles' in gcis:
            self._gci0_nonsym_sym_role_dataset = GCI0RoleDataset(gcis['gci0_nonsym_sym_roles'], class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        else:
            self._gci0_nonsym_sym_role_dataset = GCI0RoleDataset(th.tensor([]), class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        if 'gci0_sym_nonsym_roles' in gcis:
            self._gci0_sym_nonsym_role_dataset = GCI0RoleDataset(gcis['gci0_sym_nonsym_roles'], class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        else:
            self._gci0_sym_nonsym_role_dataset = GCI0RoleDataset(th.tensor([]), class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        if 'gci1_bot_roles' in gcis:
            self._gci1_bot_role_dataset = GCI1RoleDataset(gcis['gci1_bot_roles'], class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        else:
            self._gci1_bot_role_dataset = GCI1RoleDataset(th.tensor([]), class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        if 'gci1_bot_sym_roles' in gcis:
            self._gci1_bot_sym_role_dataset = GCI1RoleDataset(gcis['gci1_bot_sym_roles'], class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        else:
            self._gci1_bot_sym_role_dataset = GCI1RoleDataset(th.tensor([]), class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        if 'gci1_bot_sym_nonsym_roles' in gcis:
            self._gci1_bot_sym_nonsym_role_dataset = GCI1RoleDataset(gcis['gci1_bot_sym_nonsym_roles'], class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)
        else:
            self._gci1_bot_sym_nonsym_role_dataset = GCI1RoleDataset(th.tensor([]), class_index_dict, object_property_index_dict, sym_object_property_index_dict, device=self.device, dtype=dtype)


        self._gci2_dataset = GCI2Dataset(gcis["gci2"], class_index_dict, object_property_index_dict, device=self.device, dtype=dtype)
        self._gci3_dataset = GCI3Dataset(gcis["gci3"], class_index_dict, object_property_index_dict, device=self.device, dtype=dtype)
        if 'gci0_bot' in gcis:
            self._gci0_bot_dataset = GCI0Dataset(gcis["gci0_bot"],
                                                 class_index_dict,
                                                 object_property_index_dict,
                                                 device=self.device,
                                                 dtype=dtype)
        else:
            self._gci0_bot_dataset = GCI0Dataset(th.tensor([]),
                                                 class_index_dict,
                                                 object_property_index_dict,
                                                 device=self.device,
                                                 dtype=dtype)
        if 'gci1_bot' in  gcis:
            self._gci1_bot_dataset = GCI1Dataset(gcis["gci1_bot"],
                                                 class_index_dict,
                                                 object_property_index_dict,
                                                 device=self.device,
                                                 dtype=dtype)
        else:
            self._gci1_bot_dataset = GCI1Dataset(th.tensor([]),
                                                 class_index_dict,
                                                 object_property_index_dict,
                                                 device=self.device,
                                                 dtype=dtype)
        if 'gci3_bot' in gcis:
            self._gci3_bot_dataset = GCI3Dataset(gcis["gci3_bot"],
                                                 class_index_dict,
                                                 object_property_index_dict,
                                                 device=self.device,
                                                 dtype=dtype)
        else:
            self._gci3_bot_dataset = GCI3Dataset(th.tensor([]),
                                                 class_index_dict,
                                                 object_property_index_dict,
                                                 device=self.device,
                                                 dtype=dtype)
            
        self._positive_examples = pos_examples
        self._positive_rev_examples = pos_rev_examples
        self._class_parents = class_parents

    def set_positive_examples(self, pos_ex):
        self._positive_examples = pos_ex
    
    def set_positive_rev_examples(self, pos_ex):
        self._positive_rev_examples = pos_ex

    def set_class_parents(self, class_parents):
        self._class_parents = class_parents

    def set_device(self, device):
        self.device = device

        for dataset in self.get_gci_datasets().values():
            dataset.set_device(self.device)

    def get_gci_datasets(self):
        """Returns a dictionary containing the name of the normal forms as keys and the \
        corresponding datasets as values. This method will return 7 datasets if the class \
        parameter `extended` is True, otherwise it will return only 4 datasets.

        :rtype: dict
        """
        datasets = {
            "gci0": self.gci0_dataset,
            "gci1": self.gci1_dataset,
            "gci2": self.gci2_dataset,
            "gci3": self.gci3_dataset,
            "gci0_bot": self.gci0_bot_dataset,
            "gci1_bot": self.gci1_bot_dataset,
            "gci3_bot": self.gci3_bot_dataset,
            "gci0_roles": self.gci0_role_dataset,
            "gci0_sym_roles": self.gci0_sym_role_dataset,
            "gci0_sym_nonsym_roles": self.gci0_sym_nonsym_role_dataset,
            "gci0_nonsym_sym_roles": self.gci0_nonsym_sym_role_dataset,
            "gci1_bot_roles": self.gci1_bot_role_dataset,
            "gci1_bot_sym_roles": self.gci1_bot_sym_role_dataset,
            "gci1_bot_sym_nonsym_roles": self.gci1_bot_sym_nonsym_role_dataset,
            "gci2_sym": self.gci2_sym_dataset,
            "gci2_focus": self.gci2_focus_dataset,
            "gci2_sym_focus": self.gci2_sym_focus_dataset
        }

        if self._extended:
            datasets["gci0_bot"] = self.gci0_bot_dataset
            datasets["gci1_bot"] = self.gci1_bot_dataset
            datasets["gci3_bot"] = self.gci3_bot_dataset

        return datasets

 
    @property
    def class_index_dict(self):
        """Returns indexed dictionary with class names present in the dataset.

        :rtype: dict
        """
        # self.load()
        return self._class_index_dict

    @property
    def object_property_index_dict(self):
        """Returns indexed dictionary with object property names present in the dataset.

        :rtype: dict
        """

        # self.load()
        return self._object_property_index_dict

    @property
    def gci0_dataset(self):
        # self.load()
        return self._gci0_dataset

    @property
    def gci1_dataset(self):
        # self.load()
        return self._gci1_dataset

    @property
    def gci2_dataset(self):
        # self.load()
        return self._gci2_dataset

    @property
    def gci3_dataset(self):
        # self.load()
        return self._gci3_dataset

    @property
    def gci0_bot_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci0_bot_dataset
    
    @property
    def gci0_role_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci0_role_dataset
    
    @property
    def gci0_sym_role_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci0_sym_role_dataset
    
    @property
    def gci0_nonsym_sym_role_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci0_nonsym_sym_role_dataset
    
    @property
    def gci0_sym_nonsym_role_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci0_sym_nonsym_role_dataset
    

    @property
    def gci1_bot_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci1_bot_dataset
    
    @property
    def gci1_bot_role_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci1_bot_role_dataset
    
    @property
    def gci1_bot_sym_role_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci1_bot_sym_role_dataset
    
    @property
    def gci1_bot_sym_nonsym_role_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci1_bot_sym_nonsym_role_dataset
    
    @property
    def gci2_sym_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci2_sym_dataset
    
    @property
    def gci2_focus_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        return self._gci2_focus_dataset
        # self.load()
        try:
            return self._gci2_focus_dataset
        except AttributeError:
            return self._gci2_dataset
    
    @property
    def gci2_sym_focus_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")
        return self._gci2_sym_focus_dataset
        # self.load()
        try:
            return self._gci2_sym_focus_dataset
        except AttributeError:
            return self._gci2_sym_dataset

    @property
    def gci3_bot_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        # self.load()
        return self._gci3_bot_dataset
    
    @property
    def positive_examples(self):
        if not self._positive_examples:
            raise AttributeError("No positive examples was provided")

        # self.load()
        return self._positive_examples
    
    @property
    def positive_rev_examples(self):
        if not self._positive_rev_examples:
            raise AttributeError("No positive rev examples was provided")

        # self.load()
        return self._positive_rev_examples
    
    @property
    def class_parents(self):
        if not self._class_parents:
            raise AttributeError("No class parents was provided")

        # self.load()
        return self._class_parents


class GCI0Dataset(GCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def push_to_device(self, data):
    #     pretensor = []
    #     for gci in data:
    #         subclass = self.class_index_dict[gci.subclass]
    #         superclass = self.class_index_dict[gci.superclass]
    #         pretensor.append([subclass, superclass])
    #     tensor = th.tensor(pretensor).to(self.device)
    #     return tensor

    def get_data_(self):
        for gci in self.data:
            subclass = self.class_index_dict[gci.subclass]
            superclass = self.class_index_dict[gci.superclass]
            yield subclass, superclass


class GCI0RoleDataset(GCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GCI1Dataset(GCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def push_to_device(self, data):
    #     pretensor = []
    #     for gci in data:
    #         left_subclass = self.class_index_dict[gci.left_subclass]
    #         right_subclass = self.class_index_dict[gci.right_subclass]
    #         superclass = self.class_index_dict[gci.superclass]
    #         pretensor.append([left_subclass, right_subclass, superclass])

    #     tensor = th.tensor(pretensor).to(self.device)
    #     return tensor

    def get_data_(self):
        for gci in self.data:
            left_subclass = self.class_index_dict[gci.left_subclass]
            right_subclass = self.class_index_dict[gci.right_subclass]
            superclass = self.class_index_dict[gci.superclass]
            yield left_subclass, right_subclass, superclass

class GCI1RoleDataset(GCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GCI2Dataset(GCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def push_to_device(self, data):
    #     pretensor = []
    #     for gci in data:
    #         subclass = self.class_index_dict[gci.subclass]
    #         object_property = self.object_property_index_dict[gci.object_property]
    #         filler = self.class_index_dict[gci.filler]
    #         pretensor.append([subclass, object_property, filler])
    #     tensor = th.tensor(pretensor).to(self.device)
    #     return tensor

    def get_data_(self):
        for gci in self.data:
            subclass = self.class_index_dict[gci.subclass]
            object_property = self.object_property_index_dict[gci.object_property]
            filler = self.class_index_dict[gci.filler]
            yield subclass, object_property, filler


class GCI3Dataset(GCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def push_to_device(self, data):
    #     pretensor = []
    #     for gci in data:
    #         object_property = self.object_property_index_dict[gci.object_property]
    #         filler = self.class_index_dict[gci.filler]
    #         superclass = self.class_index_dict[gci.superclass]
    #         pretensor.append([object_property, filler, superclass])
    #     tensor = th.tensor(pretensor).to(self.device)
    #     return tensor

    def get_data_(self):
        for gci in self.data:
            object_property = self.object_property_index_dict[gci.object_property]
            filler = self.class_index_dict[gci.filler]
            superclass = self.class_index_dict[gci.superclass]
            yield object_property, filler, superclass
