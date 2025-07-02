from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
import torch as th
from parameters import LINKS, BOX_EMBEDDINGS, ONLY_GENE_BOXES
from box_embeddings.modules.intersection import GumbelIntersection
from box_embeddings.parameterizations import MinDeltaBoxTensor
from torch_geometric.nn.aggr import MultiAggregation, SoftmaxAggregation, PowerMeanAggregation, MLPAggregation, AttentionalAggregation

class GNNBase(th.nn.Module):
    def __init__(self, channels, edge_types, embeddings, edge_index_max=None):
        super().__init__()
        if edge_index_max:
            print('using attentional aggregation')
        self.layers = th.nn.ModuleList()
        # self.hetero_aggrs = th.nn.ModuleDict({k: th.nn.ModuleList()
        #                                       for k in embeddings})
        self.hetero_aggrs = th.nn.ModuleList()
        self.init_edge_dicts(embeddings, edge_types)
        es = self.es
        # sma = SoftmaxAggregation(learn=True)
        # pma = PowerMeanAggregation(learn=True)
        # multi_agg = MultiAggregation(aggrs=[sma, 'max'], mode='mean')
        prev_c = 0
        for i, c in enumerate(channels):
            # layer_sizes = self.get_layer_sizes(c, edge_types)
            # layer_sizes = {k: max(1, c // 2) if v / 1000 < 1 else c
            #                for k, v in edge_types.items()}
            # conv = HeteroConv({
            #         e: SAGEConv((int(i==0) * embeddings[e[0]].shape[1] +
            #                         int(i>0)*max((1,int(prev_c * es[e[0]]))),
            #                      int(i==0)*embeddings[e[2]].shape[1] +
            #                         int(i>0)*max((1,int(prev_c * es[e[2]])))),
            #                     max((1,int(c * es[e[2]]))), normalize=False, bias=True,
            #                     root_weight=True, project=True, aggr='max')
            #                    for e, _ in layer_sizes.items()} , aggr='mean')

            conv_dict = {}
            aggr_dict = th.nn.ModuleDict()
            for e in edge_types:
                # source_channels
                source_channels = int(i==0) * embeddings[e[0]].shape[1] + \
                    int(i>0)*max((1,int(prev_c * es[e[0]])))
                target_channels = int(i==0)*embeddings[e[2]].shape[1] + \
                    int(i>0)*max((1,int(prev_c * es[e[2]])))
                out_channels = max((1,int(c * es[e[2]])))
                if True:
                    if e[2] not in aggr_dict:
                        hidden_dim = int(out_channels // 2)
                        aggr_dict[e[2]] = AttentionalAggregation(
                            gate_nn=th.nn.Sequential(th.nn.LayerNorm(out_channels),
                                                     th.nn.Linear(out_channels, hidden_dim),
                                                     th.nn.ReLU(),
                                                     th.nn.Linear(hidden_dim, 1)))
                    hidden_dim = int(source_channels // 2)
                    # aggr = MLPAggregation(in_channels=in_channel,
                    #                       out_channels=in_channel,
                    #                       max_num_elements=edge_index_max[e],
                    #                       num_layers=1)
                    #aggr = AttentionalAggregation(gate_nn=th.nn.Sequential(
                    #    th.nn.LayerNorm(source_channels),
                    #    th.nn.Linear(source_channels, hidden_dim),
                    #    th.nn.ReLU(),
                    #    th.nn.Linear(hidden_dim, 1)))#th.nn.Sequential(
                    aggr = AttentionalAggregation(gate_nn=th.nn.Sequential(
                        th.nn.LayerNorm(source_channels),
                        th.nn.Linear(source_channels, 1)))
                    #        th.nn.Linear(in_channel, hidden_dim),
                    #        th.nn.ReLU(),
                    #        th.nn.Linear(hidden_dim, 1)
                    #    ))
                else:
                    aggr = 'max'
                root_weight = bool(i) or e[0] != 'genes'
                conv_dict[e] = SAGEConv((source_channels, target_channels),
                                out_channels, normalize=False, bias=True,
                                root_weight=root_weight, project=False, aggr=aggr)
            conv = HeteroConv(conv_dict, aggr=None)
            # for k, v in aggr_dict.items():
            self.hetero_aggrs.append(aggr_dict)
            # print(aggr_dict)
            # print(conv.convs)
            # {
            #         e: SAGEConv((int(i==0) * embeddings[e[0]].shape[1] +
            #                         int(i>0)*max((1,int(prev_c * es[e[0]]))),
            #                      int(i==0)*embeddings[e[2]].shape[1] +
            #                         int(i>0)*max((1,int(prev_c * es[e[2]])))),
            #                     max((1,int(c * es[e[2]]))), normalize=False, bias=True,
            #                     root_weight=True, project=True, aggr='max')
            #                    for e, _ in layer_sizes.items()}
            prev_c = c
            self.layers.append(conv)

    def init_edge_dicts(self, embeddings, edge_types):
        raise NotImplementedError()
    
    # def get_layer_sizes(self,):
    #     raise NotImplementedError()

    def forward(self, x_dict, edge_index_dict, return_embs=False):
        embs = []
        for conv, aggr in zip(self.layers, self.hetero_aggrs):
            x_dict = conv(x_dict, edge_index_dict)
            for k in x_dict:
                x = x_dict[k]
                N, T, F = x.size()
                x_flat = x.view(-1, F)
                index = th.arange(N, device=x.device).repeat_interleave(T)
                x_dict[k] = aggr[k](x_flat, index=index, dim_size=N)
            # x_dict = {key: aggr[key](x, index=th.arange(x.shape[1]).repeat_interleave(x.shape[0]), dim_size=x.shape[1]) for key, x in x_dict.items()}
            if return_embs:
                embs.append(x_dict)
        if return_embs:
            return embs
        return x_dict

class HeteroGNNCustom(GNNBase):
    def __init__(self, channels, edge_types, embeddings, edge_index_max=None):
        super().__init__(channels, edge_types, embeddings, edge_index_max)
        # self.layers = th.nn.ModuleList()
        # prev_c = 0
        # ed = {k: 0 for k in embeddings.keys()}
        # for e, v in edge_types.items():
        #     ed[e[0]] += v
        #     ed[e[2]] += v
        # self.es = {}
        # print(ed)
        # for k, v in ed.items():
        #     if v / 500000 > 1:
        #         self.es[k] = 2
        #     elif v / 100000 > 1:
        #         self.es[k] = 1
        #     elif v / 10000 > 1:
        #         self.es[k] = 0.5
        #     else:
        #         self.es[k] = 0.25

    def init_edge_dicts(self, embeddings, edge_types):
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


class HeteroGNN(GNNBase):
    def __init__(self, channels, edge_types, embeddings, edge_index_max=None):
        super().__init__(channels, edge_types, embeddings, edge_index_max)
        # self.layers = th.nn.ModuleList()
        # prev_c = 0
        # self.es = {k:1 for k in embeddings.keys()}
        # sma = SoftmaxAggregation(learn=True)

        # multi_agg = MultiAggregation(aggrs=[sma, 'max'], mode='mean')
        # pma = PowerMeanAggregation(learn=True)

        # for i, c in enumerate(channels):
        #     conv = HeteroConv({
        #         e: SAGEConv((int(i==0) * embeddings[e[0]].shape[1] +
        #                      int(i>0)*prev_c,int(i==0)*embeddings[e[2]].shape[1]
        #                      + int(i>0) * prev_c), c, normalize=False, bias=True,
        #                      root_weight=True, project=True, aggr='max')
        #                        for e in edge_types}, aggr='mean')
        #     prev_c = c
        #     self.layers.append(conv)

    def init_edge_dicts(self, embeddings, edge_types):
        self.es = {k:1 for k in embeddings.keys()}


class Model(th.nn.Module):
    def __init__(self, gnn_channels: list, nn_channels: list, meta_data,
                 embeddings, edge_types=[('genes', 'interacts', 'genes')], save_path=None,
                 custom=True, inter_temp=0.1, edge_index_max=None):
        super().__init__()

        if custom:
            # varying sizes of embeddings for different target domains
            self.gnn = HeteroGNNCustom(gnn_channels, edge_types, embeddings, edge_index_max)
        else:
            self.gnn = HeteroGNN(gnn_channels, edge_types, embeddings, edge_index_max)

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

        gene_boxes = (MinDeltaBoxTensor.from_vector(embs[LINKS[0]][links_to_pred[0]]),
                      MinDeltaBoxTensor.from_vector(embs[LINKS[2]][links_to_pred[1]]))
        intersects = self.intersect(gene_boxes[0], gene_boxes[1])
        z = th.cat([intersects.z, intersects.Z], dim=-1)
        #z = th.cat([(embs[LINKS[0]][links_to_pred[0]] + embs[LINKS[0]][links_to_pred[0]]) / 2, embs[LINKS[0]][links_to_pred[0]] * embs[LINKS[2]][links_to_pred[1]], z], dim=-1)

        # links_to_pred = data[LINKS].edge_label_index
        # if return_embs:
        #     embs, x_dicts = self._forward(data, return_embs=return_embs)
        # else:
        #     embs = self._forward(data)
        # for ii in range(2):
        #     z = th.concat((embs[LINKS[0]][links_to_pred[ii % 2]], embs[LINKS[2]][links_to_pred[(ii + 1) % 2]]), dim=-1)
        #     if self.lin_layers:
        #         for i, l in enumerate(self.lin_layers):
        #             z = l(z)
        #             if i > 0:
        #                 z = z.relu()
        #     else:
        #         z = z.sum(dim=-1)

        #     if ii > 0:
        #         inter = th.stack((inter, z), dim=-1)
        #     else:
        #         inter = z

        #     # inter = th.concat((inter, z), dim=-1)
        # z = self.lin4(z)
        # z = z.mean(dim=-1).squeeze()
        # z = embs[LINKS[0]][links_to_pred[0]] * embs[LINKS[2]][links_to_pred[1]]
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
    
        
    
class Regressor(Model):
    def __init__(self, gnn_channels: list, nn_channels: list, meta_data, embeddings, edge_types, save_path=None, custom=True, inter_temp=0.1, edge_index_max=None):
        super().__init__(gnn_channels, nn_channels, meta_data, embeddings, edge_types, save_path, custom, inter_temp, edge_index_max)
        if len(nn_channels) > 0:
            self.lin4 = th.nn.Linear(nn_channels[-1], 1)
        else:
            self.lin4 = th.nn.Linear(1, 1)

    def forward(self, data: HeteroData, return_embs=False):
        # links_to_pred = data[LINKS].edge_label_index
        # if return_embs:
        #     embs, x_dicts = self._forward(data, return_embs=return_embs)
        # else:
        #     embs = self._forward(data)
        # for ii in range(2):
        #     z = th.concat((embs[LINKS[0]][links_to_pred[ii % 2]], embs[LINKS[2]][links_to_pred[(ii + 1) % 2]]), dim=-1)
        #     if self.lin_layers:
        #         for i, l in enumerate(self.lin_layers):
        #             z = l(z)
        #             if i > 0:
        #                 z = z.relu()
        #     else:
        #         z = z.sum(dim=-1)

        #     if ii > 0:
        #         inter = th.stack((inter, z), dim=-1)
        #     else:
        #         inter = z

        #     # inter = th.concat((inter, z), dim=-1)
        # z = self.lin4(z)
        # z = z.mean(dim=-1).squeeze()
        if return_embs:
            z, x_dicts = self._forward(data, return_embs=return_embs)
            return self.lin4(z).squeeze(), x_dicts
            # return z, x_dicts
        else:
            z = self._forward(data)
            return self.lin4(z).squeeze()
            # return z
    
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
