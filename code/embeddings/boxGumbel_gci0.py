import torch as th
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
from .boxGumbelModule_gci0 import BoxGumbelModule
import pickle, os
import tempfile
import time
from box_embeddings.modules.regularization import L2SideBoxRegularizer

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "embeddings.boxGumbelModule_gci0" and name == "BoxRelELModule":
            from embeddings.boxGumbelModule_gci0 import BoxGumbelModule
            return BoxGumbelModule
        return super().find_class(module, name)

def boxRel_to_cpu(fp):
    model = BoxGumbelModel(None, from_file=fp)
    # model.module.set_device('cpu')
    model.module.to('cpu')
    with open(fp + '/model.pkl', 'wb') as fo:
        pickle.dump(model.module, fo)

class BoxGumbelModel:

    def __init__(self, dataset, embed_dims=50, learning_rate=0.001, epochs=1000,
                 batch_size=4096 * 8, model_filepath=None, intersect_temp=0.5,
                 vol_temp=0.5, device='cpu', neg_ratio=1.0, copy_dest='',
                 from_file=None, bot_factor=1.0, reg_factor=1e-5,
                 dataset_label='full', best_loss=float('inf'), load_epoch=-1,
                 params_from_file=0, init_for_train=True) -> None:
  
        self._extended = True
        self.embed_dims = embed_dims
        self.device = device
        self.early_stop_delay = 15
        self.copy_dest = copy_dest
        self.batch_size = batch_size
        self.intersect_temp = intersect_temp
        self.vol_temp = vol_temp
        self.bot_factor = bot_factor
        self.dataset_label = dataset_label

        self.save_all_epochs = False
        self.load_epoch = load_epoch
        self.init_for_train = init_for_train

        self.box_regularizer = L2SideBoxRegularizer(weight=reg_factor,
                                                    log_scale=True)
        self.reg_factor = reg_factor

        self.dataset = dataset
        if dataset:
            self._training_datasets = self.dataset.training_datasets.get_gci_datasets()
            if self.dataset.validation_datasets:
                self._validation_datasets = self.dataset.validation_datasets.get_gci_datasets()
            else:
                self._validation_datasets = None
            if self.dataset.testing_datasets:
                self._testing_datasets = self.dataset.testing_datasets.get_gci_datasets()
            else:
                self._testing_datasets = None

            self._datasets_loaded = True


        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self._model_filepath = model_filepath
        self._testing_set = None
        self.neg_ratio = neg_ratio
        self._evaluator_initiated = False
        self.best_loss = best_loss
        if from_file:
            if self.load_epoch > -1:
                self._model_filepath = os.path.join(from_file,
                                                    f'model_{self.load_epoch}.pkl')
            else:
                self._model_filepath = os.path.join(from_file, 'model.pkl')
            
            if params_from_file:
                print(params_from_file)
                print(bool(params_from_file))
                self.read_params_from_file(from_file)
            else:
                file = os.path.join(from_file, 'hyper_params.txt')
                with open(file, 'r') as fi:
                    for l in fi.readlines():
                        if 'dataset_label' in l:
                            self.dataset_label = l.split('\t')[-1]
            self.load_best_model()
            if init_for_train:
                time_str = time.strftime("%Y%m%d-%H%M%S")
                model_dir = f"{from_file}--{time_str}"
                if self.load_epoch > -1:
                    self._model_filepath = os.path.join(model_dir,
                                                        f'model_{self.load_epoch}.pkl')
                else:
                    self._model_filepath = os.path.join(model_dir, 'model.pkl')
                self.best_loss_path = os.path.join(model_dir, 'best_loss.txt')
                os.makedirs(model_dir)
                with open(os.path.join(model_dir, 'hyper_params.txt'), 'w') as fo:
                    fo.write(self.params_str())
        else:
            self.init_module()
            if self._model_filepath is not None:
                self.init_save_dir()

    def read_params_from_file(self, directory):
        file = os.path.join(directory, 'hyper_params.txt')
        if os.path.exists(self.best_loss_path):
            with open(os.path.join(directory, 'best_loss.txt'), 'r') as fi:
                l = fi.readlines()
                if len(l) > 0:
                    self.best_loss = float(l[0])
                else:
                    self.best_loss = 0
        with open(file, 'r') as fi:
            for l in fi.readlines():
                if 'embed_dims' in l:
                    self.embed_dims = int(l.split('\t')[-1])
                elif 'learning_rate' in l:
                    self.learning_rate = float(l.split('\t')[-1])
                elif 'epochs' in l:
                    self.epochs = int(l.split('\t')[-1])
                elif 'batch_size' in l:
                    self.batch_size = int(l.split('\t')[-1])
                elif 'neg_ratio' in l:
                    self.neg_ratio = float(l.split('\t')[-1])
                elif 'intersect_temp' in l:
                    self.intersect_temp = float(l.split('\t')[-1])
                elif 'vol_temp' in l:
                    self.vol_temp = float(l.split('\t')[-1])
                elif 'reg_factor' in l:
                    self.reg_factor = float(l.split('\t')[-1])
                    self.box_regularizer = L2SideBoxRegularizer(
                                    weight=self.reg_factor, log_scale=True)
                elif 'bot_factor' in l:
                    self.bot_factor = float(l.split('\t')[-1])
                elif 'dataset_label' in l:
                    self.dataset_label = l.split('\t')[-1]

                    
    def init_save_dir(self):
        time_str = time.strftime("%Y%m%d-%H%M%S") + f'-{self.dataset_label}'
        model_dir = os.path.join(self.model_filepath, time_str)
        self.best_loss_path = os.path.join(model_dir, 'best_loss.txt')
        os.makedirs(model_dir)
        with open(os.path.join(model_dir, 'hyper_params.txt'), 'w') as fo:
            fo.write(self.params_str())
        
        self._model_filepath = os.path.join(model_dir, 'model.pkl')

    def params_str(self):
        string = f"model_type:\t\tGumbel_gci0\n" + \
                 f"dataset_label:\t\t{self.dataset_label}\n" + \
                 f"embed_dims:\t\t{self.embed_dims}\n" + \
                 f"learning_rate:\t{self.learning_rate}\n" + \
                 f"epochs:\t\t\t{self.epochs}\n" + \
                 f"batch_size:\t\t{self.batch_size}\n" + \
                 f"neg_ratio:\t\t{self.neg_ratio}\n" + \
                 f"intersect_temp:\t\t{self.intersect_temp}\n" + \
                 f"vol_temp:\t\t{self.vol_temp}\n" + \
                 f"reg_factor:\t\t{self.reg_factor}\n" + \
                 f"bot_factor:\t\t{self.bot_factor}\n" 
        return string
    

    def init_module(self):
        self.module = BoxGumbelModule(nbr_classes=len(self.class_index_dict),
                                     embed_dims=self.embed_dims,
                                     vol_temp=self.vol_temp,
                                     intersect_temp=self.intersect_temp,
                                     device=self.device).to(self.device)

    def get_embeddings(self, trained=True):
        raise NotImplementedError()
        self.init_module()

        if trained:
            print('Load the best model', self.model_filepath)
            self.load_best_model()
                
        ent_embeds = {k: (v, o, b) for k, v, o, b in
                      zip(self.class_index_dict.keys(),
                          self.module.class_center.weight.cpu().detach().numpy(),
                          self.module.class_offset.weight.cpu().detach().numpy(),
                          self.module.bump.weight.cpu().detach().numpy())}

        return ent_embeds#, rel_embeds, sym_rel_embeds

    def train(self):
        bce = nn.BCELoss(reduction='sum')
        bce_val = nn.BCELoss(reduction='mean')
        mse_val = nn.MSELoss(reduction='mean')

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        scheduler = th.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.9)
        best_loss = self.best_loss

        full_gci0 = self.training_datasets['gci0'].data
        eval_labels = th.cat((th.ones(len(full_gci0), requires_grad=False),
                              th.zeros(2*len(full_gci0), requires_grad=False)),
                             dim=0).to(self.device)
        
        box_eval_labels = th.zeros(len(eval_labels), requires_grad=False).to(self.device)

        train_dataloader = self.training_dataloaders
        
        batch_ones = th.ones(self.batch_size,
                             requires_grad=False).to(self.device)
        neg_zeros = th.zeros(int(round(self.batch_size * self.neg_ratio)),
                             requires_grad=False).to(self.device)

        epochs_since_improved = 0
        for epoch in trange(self.epochs):
            #print('\n')
            self.module.train()

            train_loss = 0
            
            for gci_name, gci_dataloader in train_dataloader.items():
                if gci_name not in ['gci0', 'gci1_bot']:
                    continue

                for data in gci_dataloader:
                    loss = 0
                    optimizer.zero_grad()
                    if gci_name == 'gci1_bot':
                        dst = self.module(data[:,:2], 'gci0')
                        l = neg_zeros if len(dst) == len(neg_zeros) else \
                                th.zeros(dst.shape, requires_grad=False,
                                         device=self.device)
                        loss += self.bot_factor * bce(dst, l)

                    elif gci_name == 'gci0':
                        dst = self.module(data, gci_name)
                        l = batch_ones if len(dst) == len(batch_ones) else \
                                th.ones(dst.shape, requires_grad=False,
                                        device=self.device)
                        loss += bce(dst, l)

                        neg_data = self.dataset.get_negative_examples(gci_name,
                                int(round(self.neg_ratio * self.batch_size)))
                        dst = self.module(neg_data, gci_name)
                        l = neg_zeros if len(dst) == len(neg_zeros) else \
                                th.zeros(dst.shape, requires_grad=False,
                                         device=self.device)
                        
                        loss += bce(dst, l)
                        
                    regul = th.exp(self.box_regularizer(self.module.class_boxes.all_boxes))
                    loss += regul             
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
            # continue
            if epoch % 10 == 0 and epoch > 0:
                # scheduler.step()
                pass
            # print(f'Epoch {epoch}, loss: {train_loss}')
            # instead of val loss, run forward pass on training set with temp params s.t. no smoothing (also no regularization if this is used), ie get loss just corresponding to box placement - use this for possible early stopping (should negative examples be included - probably yes) - should be possible to run entire training set, since no gradients saved

            # maybe use distance loss to measure this? also good to check so that loss is also decreasing during training??

            with th.no_grad():
                neg_eval = self.dataset.get_negative_examples('gci0', 2*len(full_gci0))
                eval_data = th.cat((full_gci0, neg_eval), dim=0)
                dst = self.module(eval_data, 'gci0', val=True)
                bce_valid_loss = bce_val(dst, eval_labels)
                box_dst = self.module.box_dist_gci0_loss(full_gci0)
                box_dist_neg = self.module.box_dist_gci0_loss(neg_eval, neg=True)
                box_eval = th.cat((box_dst, box_dist_neg), dim=0)
                box_loss = mse_val(box_eval, box_eval_labels)
                valid_loss = (bce_valid_loss + box_loss) / 2


            print(f'Epoch {epoch}: Train loss: {train_loss} '
                f'Box loss: {box_loss} BCE valid loss: {bce_valid_loss} Valid loss: {valid_loss}')
            if best_loss > valid_loss or True:
                epochs_since_improved = 0
                best_loss = valid_loss
                if self.save_all_epochs:
                    self.save_model(epoch=epoch)
                else:
                    self.save_model()
                
                with open(self.best_loss_path, 'w') as fo:
                    fo.write(str(best_loss.item()))
            else:
                epochs_since_improved += 1
                if epochs_since_improved > self.early_stop_delay:
                    print(f'Model has not improved in {self.early_stop_delay} '
                        'epochs, stop training...')
                    break

            if epoch > 0 and epoch % 100 == 0 and self.copy_dest:
                print('copying')
                model_dir = os.path.dirname(self.model_filepath)

                boxRel_to_cpu(model_dir)
                os.system(f"cp -r {model_dir} {self.copy_dest}")

                
        self.best_loss = best_loss
        return 1

    def save_model(self, epoch=-1):
        print("Saving model..")
        save_path = f"{self.model_filepath.split('.')[0]}_{epoch}.pkl" \
            if epoch > -1 else self.model_filepath
        with open(save_path, 'wb') as fo:
            pickle.dump(self.module, fo)
    
    def load_best_model(self):
        # self.init_module()
        # self.module.load_state_dict(th.load(self.model_filepath,
        #                                 map_location=th.device(self.device)))
        with open(self.model_filepath, 'rb') as fi:
            self.module = RenameUnpickler(fi).load()
        # with open(self.model_filepath, 'rb') as fi:
        #     self.module = pickle.load(fi)
        
        # self.module.set_device(self.device)
        self.module.to(self.device)
        # self.module.eval()

    @property
    def training_set(self):
        self.load_eval_data()
        return self._training_set


    @property
    def testing_set(self):
        self.load_eval_data()
        return self._testing_set

    @property
    def head_entities(self):
        self.load_eval_data()
        return self._head_entities

    @property
    def tail_entities(self):
        self.load_eval_data()
        return self._tail_entities

    def load_pkl_dataset(self, train_path, val_path=None,
                         test_path=None):
        if self._datasets_loaded:
            return
        
        with open(train_path, 'rb') as fi:
            training_el_dataset = pickle.load(fi)
        self._training_datasets = training_el_dataset.get_gci_datasets()

        self._validation_datasets = None
        if val_path:
            with open(val_path, 'rb') as fi:
                validation_el_dataset = pickle.load(fi)
            self._validation_datasets = validation_el_dataset.get_gci_datasets()

        self._testing_datasets = None
        if test_path:
            with open(test_path, 'rb') as fi:
                testing_el_dataset = pickle.load(fi)
            self._testing_datasets = testing_el_dataset.get_gci_datasets()

        self._datasets_loaded = True

    def _load_dataloaders(self):

        self._training_dataloaders = {
            k: DataLoader(v, batch_size=self.batch_size, pin_memory=False, shuffle=True) for k, v in
            self._training_datasets.items() if len(v) > 0}

        if self._validation_datasets:
            self._validation_dataloaders = {
                k: DataLoader(v, batch_size=self.batch_size, pin_memory=False, shuffle=True) for k, v in
                self._validation_datasets.items() if len(v) > 0}

        if self._testing_datasets:
            self._testing_dataloaders = {
                k: DataLoader(v, batch_size=self.batch_size, pin_memory=False, shuffle=True) for k, v in
                self._testing_datasets.items() if len(v) > 0}


    @property
    def training_datasets(self):
        """Returns the training datasets for each GCI type. Each dataset is an instance \
of :class:`mowl.datasets.el.ELDataset`

        :rtype: dict
        """
        # self._load_datasets()
        return self.dataset.training_datasets.get_gci_datasets()

    @property
    def validation_datasets(self):
        """Returns the validation datasets for each GCI type. Each dataset is an instance \
of :class:`mowl.datasets.el.ELDataset`

        :rtype: dict
        """
        if self.dataset.validation_datasets is None:
            raise AttributeError("Validation dataset is None.")

        # self._load_datasets()
        return self.dataset.validation_datasets.get_gci_datasets()

    @property
    def testing_datasets(self):
        """Returns the testing datasets for each GCI type. Each dataset is an instance \
of :class:`mowl.datasets.el.ELDataset`

        :rtype: dict
        """
        if self.dataset.testing_datasets is None:
            raise AttributeError("Testing dataset is None.")

        # self._load_datasets()
        return self.dataset.testing_datasets

    @property
    def training_dataloaders(self):
        """Returns the training dataloaders for each GCI type. Each dataloader is an instance \
of :class:`torch.utils.data.DataLoader`

        :rtype: dict
        """
        self._load_dataloaders()
        return self._training_dataloaders

    @property
    def validation_dataloaders(self):
        """Returns the validation dataloaders for each GCI type. Each dataloader is an instance \
of :class:`torch.utils.data.DataLoader`

        :rtype: dict
        """
        if self.dataset.validation is None:
            raise AttributeError("Validation dataloader is None.")

        self._load_dataloaders()
        return self._validation_dataloaders

    @property
    def testing_dataloaders(self):
        """Returns the testing dataloaders for each GCI type. Each dataloader is an instance \
of :class:`torch.utils.data.DataLoader`

        :rtype: dict
        """
        if self.dataset.testing is None:
            raise AttributeError("Testing dataloader is None.")

        self._load_dataloaders()
        return self._testing_dataloaders
    
    @property
    def model_filepath(self):
        """Path for saving the model.

        :rtype: str
        """
        if self._model_filepath is None:
            model_filepath = tempfile.NamedTemporaryFile()
            self._model_filepath = model_filepath.name
        return self._model_filepath

    @property
    def class_index_dict(self):
        """Dictionary with class names as keys and class indexes as values.

        :rtype: dict
        """
        # class_to_id = {v: k for k, v in enumerate(self.dataset.classes.as_str)}
        return self.dataset.class_to_id

    @property
    def individual_index_dict(self):
        """Dictionary with individual names as keys and indexes as values.

        :rtype: dict
        """
        individual_to_id = {v: k for k, v in enumerate(self.dataset.individuals.as_str)}
        return individual_to_id
                            
    @property
    def object_property_index_dict(self):
        """Dictionary with object property names as keys and object property indexes as values.

        :rtype: dict
        """
        # object_property_to_id = {v: k for k, v in enumerate(self.dataset.object_properties.as_str)}
        return self.dataset.object_property_to_id
    
    @property
    def sym_object_property_index_dict(self):
        """Dictionary with object property names as keys and object property indexes as values.

        :rtype: dict
        """
        # object_property_to_id = {v: k for k, v in enumerate(self.dataset.object_properties.as_str)}
        return self.dataset.sym_object_property_to_id
