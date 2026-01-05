from torch_geometric.nn import SAGEConv, HeteroConv, GATv2Conv, TransformerConv
from sage_conv_mod import SAGEConvMod
from torch_geometric.data import HeteroData
import torch as th
import torch.nn as nn
from parameters import LINKS, BOX_EMBEDDINGS, ONLY_GENE_BOXES, DROP_OUT, RANDOM_INIT_EMBS, EMBEDDING_DIMS, ONLY_BILINEAR, GENE_COMBINE
from box_embeddings.modules.intersection import GumbelIntersection
from box_embeddings.parameterizations import MinDeltaBoxTensor
from torch_geometric.nn.aggr import (
    MultiAggregation,
    SoftmaxAggregation,
    PowerMeanAggregation,
    MLPAggregation,
    AttentionalAggregation,
)
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import EdgeType, NodeType

# from parameters import HEADS


def group(xs: List[th.Tensor], aggr: Optional[str]) -> Optional[th.Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return th.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return th.cat(xs, dim=-1)
    else:
        out = th.stack(xs, dim=0)
        out = getattr(th, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class HeteroConvHeads(HeteroConv):
    def __init__(
        self,
        convs: Dict[EdgeType, MessagePassing],
        aggr: Optional[str] = "sum",
        heads: Optional[int] = 4,
    ):
        super().__init__(convs=convs, aggr=aggr)
        self.heads = heads

    def forward(
        self,
        *args_dict,
        head_aggr=None,
        **kwargs_dict,
    ) -> Dict[NodeType, th.Tensor]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.
            *args_dict (optional): Additional forward arguments of individual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For example, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """
        out_dict: Dict[str, List[th.Tensor]] = {}

        for edge_type, conv in self.convs.items():
            # print(edge_type)
            src, rel, dst = edge_type

            has_edge_level_arg = False

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    has_edge_level_arg = True
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (
                            value_dict.get(src, None),
                            value_dict.get(dst, None),
                        )
                    )

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                if not arg.endswith("_dict"):
                    raise ValueError(
                        f"Keyword arguments in '{self.__class__.__name__}' "
                        f"need to end with '_dict' (got '{arg}')"
                    )

                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    has_edge_level_arg = True
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (
                        value_dict.get(src, None),
                        value_dict.get(dst, None),
                    )

            if not has_edge_level_arg:
                continue

            out = conv(*args, **kwargs)

            # here aggregate between heads
            if head_aggr and self.heads > 1:
                # dict with aggregators for each edge type
                # out has form [N, H*C], H-heads, C-out_dim
                # -> reshape to [N, C] using attentional aggregator
                aggr = head_aggr[str(edge_type)]
                aggr_size = aggr.gate_nn[1].in_features
                N, HC = out.size()
                C = HC / self.heads
                assert C == aggr_size, f"Sizes doesn't match: C={C}, aggr={aggr_size}"
                out = out.view(N, self.heads, aggr_size)
                out = aggr(out).squeeze(dim=1)

            if dst not in out_dict:
                out_dict[dst] = [out]
            else:
                out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict

class GNNBase(th.nn.Module):
    def __init__(self, embeddings, edge_types, aggr="attn"):
        super().__init__()
        self.heads = 1
        self.layers = nn.ModuleList()
        self.init_edge_dicts(embeddings, edge_types)
        # es = self.es
        assert aggr in ["attn", "mean", "max"], aggr
        self.aggr = aggr
        if self.aggr == "attn":
            self.hetero_aggrs = nn.ModuleList()

    def init_edge_dicts(self, embeddings, edge_types):
        raise NotImplementedError()

    def forward_head_attention(self, x_dict, edge_index_dict, return_embs=False):
        embs = []
        for conv, aggr, head_aggr in zip(
            self.layers, self.hetero_aggrs, self.head_aggrs
        ):
            x_dict = conv(x_dict, edge_index_dict, head_aggr=head_aggr)
            for k in x_dict:
                x = x_dict[k]
                # print(x.size())
                N, T, F = x.size()
                x_flat = x.view(-1, F)
                index = th.arange(N, device=x.device).repeat_interleave(T)
                x_dict[k] = aggr[k](x_flat, index=index, dim_size=N)
            if return_embs:
                embs.append(x_dict)
        if return_embs:
            return embs
        return x_dict

    def forward_attention(self, x_dict, edge_index_dict, return_embs=False):
        embs = []
        for conv, aggr in zip(self.layers, self.hetero_aggrs):
            x_dict = conv(x_dict, edge_index_dict)
            for k in x_dict:
                x = x_dict[k]
                N, T, F = x.size()
                x_flat = x.view(-1, F)
                index = th.arange(N, device=x.device).repeat_interleave(T)
                x_dict[k] = aggr[k](x_flat, index=index, dim_size=N)
            if return_embs:
                embs.append(x_dict)
        if return_embs:
            return embs
        return x_dict

    def forward(self, x_dict, edge_index_dict, return_embs=False):
        if self.heads > 1 and self.aggr == "attn":
            return self.forward_head_attention(
                x_dict, edge_index_dict, return_embs=return_embs
            )
        if self.aggr == "attn":
            return self.forward_attention(
                x_dict, edge_index_dict, return_embs=return_embs
            )
        else:
            embs = [x_dict]
            for conv in self.layers:
                x_dict = conv(x_dict, edge_index_dict)
                if return_embs:
                    embs.append(x_dict)
            if return_embs:
                return embs
            return x_dict


class GNNBaseTransfromer(GNNBase):
    def __init__(
        self, channels, edge_types, embeddings, aggr="attn", heads=4, skip_last=True
    ):
        super().__init__(embeddings=embeddings, edge_types=edge_types, aggr=aggr)
        self.heads = heads
        print(f"number of heads: {heads}")
        es = self.es
        prev_c = 0
        self.head_aggrs = th.nn.ModuleList()
        for i, c in enumerate(channels):
            conv_dict = {}
            aggr_dict = th.nn.ModuleDict()
            head_aggr_dict = th.nn.ModuleDict()
            for e in edge_types:
                if skip_last and (i == len(channels) - 1 and e[2] != "genes"):
                    continue
                source_channels = int(i == 0) * embeddings[e[0]].shape[1] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[0]])))
                target_channels = int(i == 0) * embeddings[e[2]].shape[1] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[2]])))
                out_channels = max((1, int(c * es[e[2]])))

                if self.aggr == "attn" and e[2] not in aggr_dict:
                    # aggregate different edge types to each node type
                    # hidden_dim = int(out_channels // 2)
                    aggr_dict[e[2]] = AttentionalAggregation(
                        gate_nn=nn.Sequential(
                            nn.LayerNorm(out_channels),
                            nn.Linear(out_channels, 1, bias=True),
                        )
                    )
                if self.heads > 1 and self.aggr == "attn":
                    # aggregate between heads for each edge type
                    head_aggr_dict[str(e)] = AttentionalAggregation(
                        gate_nn=nn.Sequential(
                            nn.LayerNorm(out_channels),
                            nn.Linear(out_channels, 1, bias=True),
                        )
                    )
                root_weight = bool(i) or e[2] != "genes"
                concat_heads = self.heads > 1 and self.aggr == "attn"
                conv_dict[e] = TransformerConv(
                    (source_channels, target_channels),
                    out_channels,
                    heads=self.heads,
                    concat=concat_heads,
                    bias=True,
                    root_weight=root_weight,
                    beta=True,
                )
            aggr = None if self.aggr == "attn" else self.aggr
            # if aggregate heads should use custom heteroconv
            if self.heads > 1 and self.aggr == "attn":
                conv = HeteroConvHeads(conv_dict, aggr=aggr, heads=self.heads)
            else:
                conv = HeteroConv(conv_dict, aggr=aggr)

            if self.aggr == "attn":
                self.hetero_aggrs.append(aggr_dict)
            if self.heads > 1 and self.aggr == "attn":
                self.head_aggrs.append(head_aggr_dict)

            prev_c = c
            self.layers.append(conv)

    def init_edge_dicts(self, embeddings, edge_types):
        raise NotImplementedError()


class HeteroGNNTransformerCustom(GNNBaseTransfromer):
    def __init__(
        self, channels, edge_types, embeddings, aggr="attn", heads=4, skip_last=True
    ):
        print("HeteroGNNTransformerCustom")
        super().__init__(
            channels,
            edge_types,
            embeddings,
            aggr=aggr,
            heads=heads,
            skip_last=skip_last,
        )

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
            else:
                self.es[k] = 0.5


class HeteroGNNTransformer(GNNBaseTransfromer):
    def __init__(
        self, channels, edge_types, embeddings, aggr="attn", heads=4, skip_last=False
    ):
        print("HeteroGNNTransformer")
        super().__init__(
            channels,
            edge_types,
            embeddings,
            aggr=aggr,
            heads=heads,
            skip_last=skip_last,
        )

    def init_edge_dicts(self, embeddings, edge_types):
        self.es = {k: 1 for k in embeddings.keys()}


class GNNBaseGAT(GNNBase):
    def __init__(self, channels, edge_types, embeddings, aggr="attn"):
        super().__init__(embeddings=embeddings, edge_types=edge_types, aggr=aggr)
        es = self.es
        prev_c = 0
        for i, c in enumerate(channels):
            conv_dict = {}
            aggr_dict = th.nn.ModuleDict()
            for e in edge_types:
                source_channels = int(i == 0) * embeddings[e[0]].shape[1] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[0]])))
                target_channels = int(i == 0) * embeddings[e[2]].shape[1] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[2]])))
                out_channels = max((1, int(c * es[e[2]])))

                if self.aggr == "attn" and e[2] not in aggr_dict:
                    aggr_dict[e[2]] = AttentionalAggregation(
                        gate_nn=th.nn.Sequential(
                            th.nn.LayerNorm(out_channels),
                            th.nn.Linear(out_channels, 1, bias=True),
                        )
                    )

                conv_dict[e] = GATv2Conv(
                    (source_channels, target_channels),
                    out_channels,
                    add_self_loops=False,
                    heads=2,
                    concat=False,
                )
            aggr = None if self.aggr == "attn" else self.aggr
            conv = HeteroConv(conv_dict, aggr=aggr)
            print("mod sageconv")

            if self.aggr == "attn":
                self.hetero_aggrs.append(aggr_dict)

            prev_c = c
            self.layers.append(conv)

    def init_edge_dicts(self, embeddings, edge_types):
        raise NotImplementedError()


class HeteroGNNGAT(GNNBaseGAT):
    def __init__(self, channels, edge_types, embeddings, aggr="attn"):
        print("HeteroGNNGAT")
        super().__init__(channels, edge_types, embeddings, aggr=aggr)

    def init_edge_dicts(self, embeddings, edge_types):
        self.es = {k: 1 for k in embeddings.keys()}


class HeteroGNNGATCustom(GNNBaseGAT):
    def __init__(self, channels, edge_types, embeddings, aggr="attn"):
        print("HeteroGNNGATCustom")
        super().__init__(channels, edge_types, embeddings, aggr=aggr)

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
            else:
                self.es[k] = 0.5


class GNNBaseSAGE(GNNBase):
    def __init__(self, channels, edge_types, embeddings, aggr="attn", skip_last=True):
        super().__init__(embeddings=embeddings, edge_types=edge_types, aggr=aggr)
        es = self.es
        prev_c = 0
        for i, c in enumerate(channels):
            conv_dict = {}
            aggr_dict = th.nn.ModuleDict()
            for e in edge_types:
                if skip_last and (i == len(channels) - 1 and e[2] != "genes"):
                    continue
                source_channels = int(i == 0) * embeddings[e[0]].shape[1] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[0]])))
                target_channels = int(i == 0) * embeddings[e[2]].shape[1] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[2]])))
                out_channels = max((1, int(c * es[e[2]])))

                if self.aggr == "attn":
                    if e[2] not in aggr_dict:
                        aggr_dict[e[2]] = AttentionalAggregation(
                            gate_nn=nn.Sequential(
                                nn.LayerNorm(out_channels),
                                nn.Linear(out_channels, 1, bias=True),
                            )
                        )
                    aggr = AttentionalAggregation(
                        gate_nn=nn.Sequential(
                            nn.LayerNorm(source_channels),
                            nn.Linear(source_channels, 1, bias=True),
                        )
                    )
                else:
                    aggr = self.aggr
                root_weight = bool(i) or e[2] != "genes"  # or True
                conv_dict[e] = SAGEConvMod(
                    (source_channels, target_channels),
                    out_channels,
                    normalize=(not skip_last),
                    bias=True,
                    root_weight=root_weight,
                    project=True,
                    project_out=True,
                    full_bias=True,
                    aggr=aggr,
                )
            aggr = None if self.aggr == "attn" else self.aggr
            aggr = "mean"
            conv = HeteroConv(conv_dict, aggr=aggr)
            print("mod sageconv")
            if self.aggr == "attn":
                self.hetero_aggrs.append(aggr_dict)
            prev_c = c
            self.layers.append(conv)

    def init_edge_dicts(self, embeddings, edge_types):
        raise NotImplementedError()

class HeteroGNNCustom(th.nn.Module):
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

    def forward(self, x_dict, edge_index_dict):
        for conv in self.layers:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x for key, x in x_dict.items()}
        return x_dict
    
    def all_embeddings(self, x_dict, edge_index_dict):
        ret_embs = []
        for conv in self.layers:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x for key, x in x_dict.items()}
            ret_embs.append(x_dict)
        return ret_embs 

class HeteroGNNSAGECustom(GNNBaseSAGE):
    def __init__(self, channels, edge_types, embeddings, aggr="attn", skip_last=True):
        print("HeteroGNNCustom")
        super().__init__(
            channels, edge_types, embeddings, aggr=aggr, skip_last=skip_last
        )

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
            else:
                self.es[k] = 0.5
            # elif v / 10000 > 1:
            #    self.es[k] = 0.5
            # else:
            #    self.es[k] = 0.25


class HeteroGNNSAGE(GNNBaseSAGE):
    def __init__(self, channels, edge_types, embeddings, aggr="attn", skip_last=False):
        print("HeteroGNN")
        super().__init__(
            channels, edge_types, embeddings, aggr=aggr, skip_last=skip_last
        )

    def init_edge_dicts(self, embeddings, edge_types):
        self.es = {k: 1 for k in embeddings.keys()}


class OGGNNCustom(th.nn.Module):
    def __init__(self, channels, edge_types, embedding_dims, skip_last=True):
        print("OGGNNCustom")
        super().__init__()
        self.layers = th.nn.ModuleList()
        prev_c = 0
        ed = {k: 0 for k in embedding_dims.keys()}
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
            conv_dict = {}
            for e in edge_types:
                if skip_last and (i == len(channels) - 1 and e[2] != "genes"):
                    continue

                source_channels = int(i == 0) * embedding_dims[e[0]] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[0]])))
                target_channels = int(i == 0) * embedding_dims[e[2]] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[2]])))
                out_channels = max((1, int(c * es[e[2]])))
                root_weight = bool(i) or e[2] != "genes"

                conv_dict[e] = SAGEConv(
                    (source_channels, target_channels),
                    out_channels,
                    normalize=True,
                    root_weight=root_weight,
                    project=True,  # aggr='lstm')
                    aggr="max",
                )
            # layer_sizes = {k: max(1, c // 2) if v / 1000 < 1 else c
            #                for k, v in edge_types.items() if (i != len(channels) - 1) or (e[2] == 'genes')}
            conv = HeteroConv(conv_dict, aggr="mean")
            prev_c = c
            self.layers.append(conv)

    def forward(self, x_dict, edge_index_dict, return_embs=False):
        embs = [x_dict]
        for conv in self.layers:
            x_dict = conv(x_dict, edge_index_dict)
            if return_embs:
                embs.append(x_dict)
        if return_embs:
            return embs
        return x_dict


class OGGNN(th.nn.Module):
    def __init__(self, channels, edge_types, embedding_dims, skip_last=False):
        super().__init__()
        self.layers = th.nn.ModuleList()
        prev_c = 0
        es = {k: 1 for k in embedding_dims.keys()}
        for i, c in enumerate(channels):
            conv_dict = {}
            for e in edge_types:
                if skip_last and (i == len(channels) - 1 and e[2] != "genes"):
                    continue
                source_channels = int(i == 0) * embedding_dims[e[0]] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[0]])))
                target_channels = int(i == 0) * embedding_dims[e[2]] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[2]])))
                out_channels = max((1, int(c * es[e[2]])))
                root_weight = bool(i) or e[2] != "genes"

                conv_dict[e] = SAGEConv(
                    (source_channels, target_channels),
                    out_channels,
                    normalize=True,
                    root_weight=root_weight,
                    project=True,
                    aggr="max",
                )
            conv = HeteroConv(conv_dict, aggr="mean")
            prev_c = c
            self.layers.append(conv)

    def forward(self, x_dict, edge_index_dict, return_embs=False):
        embs = [x_dict]
        for conv in self.layers:
            x_dict = conv(x_dict, edge_index_dict)
            if return_embs:
                embs.append(x_dict)
        if return_embs:
            return embs
        return x_dict


class Model(th.nn.Module):
    def __init__(
        self,
        gnn_channels: list,
        nn_channels: list,
        meta_data,
        embeddings,
        edge_types=[("genes", "interacts", "genes")],
        save_path=None,
        custom=True,
        inter_temp=0.1,
        aggr="attn",
    ):
        super().__init__()
        # custom = False
        # print('none custom sizes for gnn')

        # if custom:
        # varying sizes of embeddings for different target domains
        # self.gnn = HeteroGNNGATCustom(gnn_channels, edge_types, embeddings, aggr=aggr)
        #self.gnn = HeteroGNNTransformerCustom(gnn_channels, edge_types, embeddings, aggr=aggr, heads=HEADS, skip_last=True)
        if len(gnn_channels) > 0:
            
            #self.gnn = HeteroGNNSAGECustom(gnn_channels, edge_types, embeddings,
            #                                aggr='max', skip_last=True)
            self.dropout = th.nn.Dropout(DROP_OUT)
            emb_dims = {k: v.shape[1] for k,v in embeddings.items()}
            if RANDOM_INIT_EMBS:
                emb_dims = EMBEDDING_DIMS
                self.node_embeddings = th.nn.ModuleDict([[k,
                                    th.nn.Embedding(num_embeddings=v.shape[0],
                                                    embedding_dim=EMBEDDING_DIMS[k])]
                                                    for k,v in embeddings.items()])
            elif ONLY_GENE_BOXES:
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
                self.node_embeddings = th.nn.ModuleDict([[k,
                                    th.nn.Embedding(num_embeddings=v.shape[0],
                                                    embedding_dim=v.shape[1])]
                                                    for k,v in embeddings.items()])
            self.gnn = OGGNNCustom(gnn_channels, edge_types, emb_dims, skip_last=True)
            prev_width = max((1, int(gnn_channels[-1] * self.gnn.es['genes'])))
            self.W = th.nn.Linear(prev_width, prev_width, bias=False)
            self.A = nn.Parameter(th.randn(prev_width, prev_width))
            self.u = nn.Parameter(th.zeros(prev_width))
            self.b = nn.Parameter(th.zeros(1))

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
        self._neighbors_to_sample = {
            "neighbors": neighbors,
            "val_neighbors": val_neighbors,
        }

    def forward(self, data: HeteroData):
        raise NotImplementedError()

    def _forward(self, data: HeteroData, return_embs=False) -> th.Tensor:
        links_to_pred = data[LINKS].edge_label_index
        x_dict = {
            k: self.node_embeddings[k](data[k].node_id) for k in self.node_embeddings
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict, return_embs=return_embs)
        embs = x_dict[-1] if return_embs else x_dict

        g1 = embs[LINKS[0]][links_to_pred[0]]
        g2 = embs[LINKS[2]][links_to_pred[1]]
        

        if GENE_COMBINE == 'bilinear':
            z = g1 * self.W(g2) + g2 * self.W(g1)
        elif GENE_COMBINE == 'concat':
            z = th.cat([g1, g2], dim=-1)
        elif GENE_COMBINE == 'product':
            z = g1 * g2
        elif GENE_COMBINE == 'intersection':
            gene_boxes = (
                MinDeltaBoxTensor.from_vector(embs[LINKS[0]][links_to_pred[0]]),
                MinDeltaBoxTensor.from_vector(embs[LINKS[2]][links_to_pred[1]]),
            )
            intersects = self.intersect(gene_boxes[0], gene_boxes[1])
            g1 = intersects.z
            g2 = intersects.z
            z = th.cat([intersects.z, intersects.Z], dim=-1)
        else:
            raise ValueError(f"Unknown gene combine method: {GENE_COMBINE}")
        

        if self.lin_layers:
            for i, l in enumerate(self.lin_layers):
                z = l(z).relu()
                z = self.dropout(z)
        else:
            if ONLY_BILINEAR:
                linear = 0
            else:
                linear = (g1 @ self.u) + (g2 @ self.u)
            W = 0.5 * (self.A + self.A.t())
            bilinear = th.sum(g1 * (g2 @ W), dim=-1)
            z = bilinear + linear + self.b

        if return_embs:
            return z, x_dict
        else:
            return z

    def gene_embedding(self, data: HeteroData) -> th.Tensor:
        x_dict = {
            k: self.node_embeddings[k](data[k].node_id) for k in self.node_embeddings
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        return x_dict["genes"]


class OntologyGNN(th.nn.Module):
    def __init__(self, channels, edge_types, embeddings, aggr="attn"):
        super().__init__()

        self.node_embeddings = th.nn.ModuleDict(
                [[k, th.nn.Embedding.from_pretrained(v.clone(), freeze=False)]
                 for k,v in embeddings.items()])
        self.gnn = OGGNN(channels, edge_types, embeddings, skip_last=False)
        # self.gnn = HeteroGNNSAGE(channels, edge_types, embeddings,
        #                          aggr='attn', skip_last=False)
        # self.gnn = HeteroGNNTransformer(channels, edge_types, embeddings,
        #                          aggr='attn', skip_last=False, heads=4)
        
        # self.gnn = HeteroGNNTransformer(channels, edge_types, embeddings)

        # self.gnn = HeteroGNNTransformer(channels, edge_types, embeddings)

    def forward(self, data: HeteroData, return_embs=True):
        x_dict = {
            k: self.node_embeddings[k](data[k].node_id) for k in self.node_embeddings
        }
        x_dicts = self.gnn(x_dict, data.edge_index_dict, return_embs=return_embs)
        # print(x_dicts)

        return x_dicts


class Regressor(Model):
    def __init__(
        self,
        gnn_channels: list,
        nn_channels: list,
        meta_data,
        embeddings,
        edge_types,
        save_path=None,
        custom=True,
        inter_temp=0.1,
        aggr="attn",
    ):
        super().__init__(
            gnn_channels,
            nn_channels,
            meta_data,
            embeddings,
            edge_types,
            save_path,
            custom,
            inter_temp,
            aggr=aggr,
        )
        if len(nn_channels) > 0:
            self.lin4 = th.nn.Linear(nn_channels[-1], 1)
        else:
            self.lin4 = th.nn.Linear(1, 1)

    def forward(self, data: HeteroData, return_embs=False):

        if return_embs:
            z, x_dicts = self._forward(data, return_embs=return_embs)
            if self.lin_layers:
                return self.lin4(z).squeeze(), x_dicts
            else:
                return z.squeeze(), x_dicts
            # return z, x_dicts
        else:
            z = self._forward(data)
            if self.lin_layers:
                return self.lin4(z).squeeze()
            else:
                return z.squeeze()
            # return z

    def predict_from_embedding(self, emb):
        if self.lin_layers:
            for l in self.lin_layers:
                z = l(emb).relu()
            
        else:
            z = emb.sum(dim=-1)

        return self.lin4(z).squeeze()


class Classifier(Model):
    def __init__(
        self,
        gnn_channels: list,
        nn_channels: list,
        meta_data,
        embeddings,
        edge_types,
        nbr_classes=2,
        save_path=None,
    ):
        super().__init__(
            gnn_channels, nn_channels, meta_data, embeddings, edge_types, save_path
        )
        self.activation = th.nn.Sigmoid()

    def forward(self, data: HeteroData) -> th.Tensor:

        z = self._forward(data).sum(dim=-1)
        return self.activation(z)
