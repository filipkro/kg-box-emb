from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
import torch as th
from parameters import LINKS, BOX_EMBEDDINGS, ONLY_GENE_BOXES
from box_embeddings.modules.intersection import GumbelIntersection
from box_embeddings.parameterizations import MinDeltaBoxTensor

class GNNBase(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_dict, edge_index_dict, return_embs=False):
        embs = []
        for conv in self.layers:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x for key, x in x_dict.items()}
            if return_embs:
                embs.append(x_dict)
        if return_embs:
            return embs
        return x_dict

class HeteroGNNCustom(GNNBase):
    def __init__(self, channels, edge_types, embeddings):
        super().__init__()
        self.layers = th.nn.ModuleList()
        prev_c = 0
        ed = {k: 0 for k in embeddings.keys()}
        for e, v in edge_types.items():
            ed[e[0]] += v
            ed[e[2]] += v
        self.es = {}
        print(ed)
        for k, v in ed.items():
            if v / 500000 > 1:
                self.es[k] = 2
            elif v / 100000 > 1:
                self.es[k] = 1
            elif v / 10000 > 1:
                self.es[k] = 0.5
            else:
                self.es[k] = 0.25
        es = self.es
        for i, c in enumerate(channels):
            layer_sizes = {k: max(1, c // 2) if v / 1000 < 1 else c
                           for k, v in edge_types.items()}
            conv = HeteroConv({
                    e: SAGEConv((int(i==0) * embeddings[e[0]].shape[1] +
                                    int(i>0)*max((1,int(prev_c * es[e[0]]))),
                                 int(i==0)*embeddings[e[2]].shape[1] +
                                    int(i>0)*max((1,int(prev_c * es[e[2]])))),
                                max((1,int(c * es[e[2]]))), normalize=True,
                                root_weight=True, project=True, aggr='max')
                               for e, _ in layer_sizes.items()} , aggr='mean')
            prev_c = c
            self.layers.append(conv)

class HeteroGNN(GNNBase):
    def __init__(self, channels, edge_types, embeddings):
        super().__init__()
        self.layers = th.nn.ModuleList()
        prev_c = 0
        self.es = {k:1 for k in embeddings.keys()}
        for i, c in enumerate(channels):
            conv = HeteroConv({
                e: SAGEConv((int(i==0) * embeddings[e[0]].shape[1] +
                             int(i>0)*prev_c,int(i==0)*embeddings[e[2]].shape[1]
                             + int(i>0) * prev_c), c, normalize=True,
                             root_weight=True, project=True, aggr='max')
                               for e in edge_types}, aggr='mean')
            prev_c = c
            self.layers.append(conv)


class Model(th.nn.Module):
    def __init__(self, gnn_channels: list, nn_channels: list, meta_data,
                 embeddings, inter_temp=0.1,
                 edge_types=[('genes', 'interacts', 'genes')], save_path=None,
                 custom=True):
        super().__init__()

        if custom:
            # varying sizes of embeddings for different target domains
            self.gnn = HeteroGNNCustom(gnn_channels, edge_types, embeddings)
        else:
            self.gnn = HeteroGNN(gnn_channels, edge_types, embeddings)

        if ONLY_GENE_BOXES:
            self.node_embeddings = th.nn.ModuleDict(
                [[k, th.nn.Embedding(num_embeddings=v.shape[0],
                                     embedding_dim=v.shape[1])]
                    for k,v in embeddings.items()])
            self.node_embeddings['genes'] = th.nn.Embedding.from_pretrained(
                embeddings['genes'].clone(), freeze=True)
        elif BOX_EMBEDDINGS:
            self.node_embeddings = th.nn.ModuleDict(
                [[k, th.nn.Embedding.from_pretrained(v.clone(), freeze=True)]
                 for k,v in embeddings.items()])
        else:
            self.node_embeddings = th.nn.ModuleDict([[k, th.nn.Embedding(num_embeddings=v.shape[0], embedding_dim=v.shape[1])] for k,v in embeddings.items()])
        prev_width = max((1, int(gnn_channels[-1] * self.gnn.es['genes'])))
        layers = []
        if len(nn_channels) > 0:
            for c in nn_channels:
                layers.append(th.nn.Linear(prev_width, c, bias=True))
                prev_width = c
            self.lin_layers = th.nn.ModuleList(layers)
        else:
            self.lin_layers = None
        
        self.fp = save_path
        self._neighbors_to_sample = None
        self.intersect = GumbelIntersection(intersection_temperature=inter_temp)

    @property
    def neighbors_to_sample(self):
        if self._neighbors_to_sample == None:
            raise AttributeError("neighbors_to_sample is not set")
        else:
            return self._neighbors_to_sample
        
    def set_neighbors_to_sample(self, neighbors, val_neighbors=None):
        if val_neighbors == None:
            val_neighbors = neighbors
        self._neighbors_to_sample = {'neighbors': neighbors,
                                     'val_neighbors': val_neighbors}

    def forward(self, data: HeteroData):
        raise NotImplementedError()
        
    def _forward(self, data: HeteroData, return_embs=False) -> th.Tensor:
        links_to_pred = data[LINKS].edge_label_index
        x_dict = {k: self.node_embeddings[k](data[k].node_id)
                  for k in self.node_embeddings}
        x_dict = self.gnn(x_dict, data.edge_index_dict,
                          return_embs=return_embs)
        embs = x_dict[-1] if return_embs else x_dict

        z = embs[LINKS[0]][links_to_pred[0]] * embs[LINKS[2]][links_to_pred[1]]
        if self.lin_layers:
            for i, l in enumerate(self.lin_layers):
                z = l(z)
                if i > 0:
                    z = z.relu()
        else:
            z = z.sum(dim=-1)
        
        if return_embs:
            return z, x_dict
        else:
            return z
    
    def gene_embedding(self, data: HeteroData) -> th.Tensor:
        x_dict = {k: self.node_embeddings[k](data[k].node_id)
                  for k in self.node_embeddings}
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        return x_dict['genes']
    
class DummyModel(Model):
    def __init__(self, gnn_channels, nn_channels, meta_data, embeddings, edge_types=[('genes', 'interacts', 'genes')], save_path=None):
        super().__init__(gnn_channels, nn_channels, meta_data, embeddings, edge_types, save_path)
        self.transition_layer = th.nn.Linear(self.node_embeddings['genes'].embedding_dim, self.lin_layers[0].in_features, bias=True)
        self.lin4 = th.nn.Linear(self.lin_layers[-1].out_features, 1)

    def forward(self, data: HeteroData):
        links_to_pred = data[LINKS].edge_label_index
        x_dict = {k: self.node_embeddings[k](data[k].node_id)
                  for k in self.node_embeddings}
        # z = x_dict[LINKS[0]][links_to_pred[0]] * x_dict[LINKS[2]][links_to_pred[1]]
        gene_boxes = (MinDeltaBoxTensor.from_vector(x_dict[LINKS[0]][links_to_pred[0]]),
                      MinDeltaBoxTensor.from_vector(x_dict[LINKS[2]][links_to_pred[1]]))
        z = self.intersect(gene_boxes[0], gene_boxes[1])
        z = self.transition_layer(z).relu()
        if self.lin_layers:
            for l in self.lin_layers:
                z = l(z).relu()
        else:
            z = z.sum(dim=-1)

        return self.lin4(z).squeeze()
        
    
class Regressor(Model):
    def __init__(self, gnn_channels: list, nn_channels: list, meta_data, embeddings, edge_types, save_path=None, custom=True):
        super().__init__(gnn_channels, nn_channels, meta_data, embeddings, edge_types, save_path, custom)
        if len(nn_channels) > 0:
            self.lin4 = th.nn.Linear(nn_channels[-1], 1)
        else:
            self.lin4 = th.nn.Linear(1, 1)

    def forward(self, data: HeteroData, return_embs=False):
        if return_embs:
            z, x_dicts = self._forward(data, return_embs=return_embs)
            return self.lin4(z).squeeze(), x_dicts
        else:
            z = self._forward(data)
            return self.lin4(z).squeeze()
    
    def predict_from_embedding(self, emb):
        if self.lin_layers:
            for l in self.lin_layers:
                z = l(emb).relu()
        else:
            z = emb.sum(dim=-1)

        return self.lin4(z).squeeze()

class Classifier(Model):
    def __init__(self, gnn_channels: list, nn_channels: list, meta_data, embeddings, edge_types, nbr_classes=2, save_path=None):
        super().__init__(gnn_channels, nn_channels, meta_data, embeddings, edge_types, save_path)
        self.activation = th.nn.Sigmoid()

    def forward(self, data: HeteroData) -> th.Tensor:

        z = self._forward(data).sum(dim=-1)
        return self.activation(z)
