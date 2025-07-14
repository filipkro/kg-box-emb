# %%
import torch
import os, pickle
from model import HeteroGNNGAT
from box_embeddings.parameterizations import MinDeltaBoxTensor, SigmoidBoxTensor
from box_embeddings.modules.intersection import GumbelIntersection
from box_embeddings.modules.volume import BesselApproxVolume

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# %%
def box_loss(embeddings, gci0, loss_type='distance', box_transform='mindelta',
             inter='gumbel', inter_temp=0.1, vol='bessel', vol_temp=0.1,
             gamma=0.0, neg_data=None, neg=False, **kwargs):
    match box_transform:
        case 'mindelta':
            box = MinDeltaBoxTensor
        case 'sigmoid':
            box = SigmoidBoxTensor
        case _:
            raise NotImplementedError()
    if loss_type == 'inclusion':
        return box_loss_inclusion(embeddings, gci0, box=box, inter=inter,
                                  inter_temp=inter_temp, vol=vol,
                                  vol_temp=vol_temp, neg_data=neg_data, neg=neg)
    if loss_type == 'distance':
        return box_loss_distance(embeddings, gci0, box=box, gamma=gamma,
                                 neg_data=neg_data, neg=neg)
    pass

def box_loss_inclusion(embeddings, gci0, box=MinDeltaBoxTensor, inter='gumbel',
             inter_temp=0.1, vol='bessel', vol_temp=0.1, neg_data=None,
             neg=False, **kwargs):
    if neg or neg_data:
        raise NotImplementedError("Negative loss not yet implemented "
                                  "for inclusion loss")
    match inter:
        case 'gumbel':
            intersect = GumbelIntersection(intersection_temperature=inter_temp)
        case _:
            raise NotImplementedError()
        
    match vol:
        case 'bessel':
            volume = BesselApproxVolume(intersection_temperature=inter_temp,
                                        volume_temperature=vol_temp, 
                                        log_scale=False)
    loss = 0
    
    for x_dict in embeddings:
        for k, emb in x_dict.items():
            
            if k == 'genes':
                continue
            box_emb = box.from_vector(emb)
            
            subclasses = box_emb[gci0[k][:,0], ...]
            supclasses = box_emb[gci0[k][:,1], ...]

            loss -= (volume(intersect(subclasses, supclasses)) /
                     volume(subclasses)).clamp(min=1e-9, max=1).log().sum()

    return loss


def box_loss_distance(embeddings, gci0, box=MinDeltaBoxTensor, gamma=0.0,
                      neg_data=None, neg=False):

    def dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False):
        n = -1 if neg else 1
        return torch.relu(n*(torch.abs(sub_c - sup_c) + sub_o - sup_o -
                          gamma)).norm(dim=-1).sum()
    loss = 0
    neg_loss = 0
    for x_dict in embeddings:
        for k, emb in x_dict.items():
            if k == 'genes':
                continue
            box_emb = box.from_vector(emb)
            
            subclasses = box_emb[gci0[k][:,0], ...]
            sub_c, sub_o = subclasses.centre, subclasses.Z - subclasses.centre
            supclasses = box_emb[gci0[k][:,1], ...]
            sup_c, sup_o = supclasses.centre, supclasses.Z - supclasses.centre

            loss += dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False)
            
            
            if neg:
                max_i = len(emb)

                rand_classes = torch.randint(low=0, high=max_i,
                                             size=(len(gci0[k]),),
                                             device=gci0[k].device)
                nsub = box_emb[rand_classes, ...]
                # print(nsub.box_shape)
                nsub_c, nsub_o = nsub.centre, nsub.Z - nsub.centre
                # print(nsub_c.shape)
                # print(nsub_o.shape)
                # print(sup_c.shape)
                # print(sup_o.shape)
                neg_loss += dist_inclusion(nsub_c, nsub_o, sup_c, sup_o,
                                           neg=True)

                rand_classes = torch.randint(low=0, high=max_i,
                                             size=(len(gci0[k]),),
                                             device=gci0[k].device)
                nsup = box_emb[rand_classes, ...]
                nsup_c, nsup_o = nsup.centre, nsup.Z - nsup.centre
                neg_loss += dist_inclusion(sub_c, sub_o, nsup_c, nsup_o,
                                           neg=True)

                rand_classes = torch.randint(low=0, high=max_i,
                                             size=(len(gci0[k]),2),
                                             device=gci0[k].device)
                nsub = box_emb[rand_classes[:,0], ...]
                nsub_c, nsub_o = nsub.centre, nsub.Z - nsub.centre
                nsup = box_emb[rand_classes[:,1], ...]
                nsup_c, nsup_o = nsup.centre, nsup.Z - nsup.centre
                neg_loss += dist_inclusion(nsub_c, nsub_o, nsup_c, nsup_o,
                                           neg=True)
                
            if neg_data:
                subclasses = box_emb[neg_data[k][:,0], ...]
                sub_c = subclasses.centre
                sub_o = subclasses.Z - subclasses.centre
                supclasses = box_emb[neg_data[k][:,1], ...]
                sup_c = supclasses.centre
                sup_o = supclasses.Z - supclasses.centre

                neg_loss += dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=True)

    return loss, neg_loss
# %%
GNN_CHANNELS = [2*2]
LR = 1e-3
REGULARIZATION = 1e-2
EPOCHS = 10
NEG_WEIGHT = 1e7
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(BASE, 'datasets/box_graph.pkl'), 'rb') as fi:
    data = pickle.load(fi)
graph = data['graph'].to(device)
gci = data['gci']
gci = {k: {kk: vv.to(device) for kk, vv in v.items()} for k,v in gci.items()}
# %%

model = HeteroGNNGAT(GNN_CHANNELS, graph.edge_types, graph.x_dict)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                             weight_decay=REGULARIZATION)
# %%
model.requires_grad_(True)

for epoch in range(100*EPOCHS):
    optimizer.zero_grad()

    x_dicts = model(graph.x_dict, graph.edge_index_dict, return_embs=True)

    pos_loss, neg_loss = box_loss(x_dicts, gci['gci0'], neg_data=gci['gci1_bot'], neg=True)
    loss = pos_loss + NEG_WEIGHT * neg_loss
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch}, loss: {loss.detach().item()}, neg loss: {neg_loss}")
# %%
model.to('cpu')
with open('box_model.pkl', 'wb') as fo:
    pickle.dump(model, fo)
# %%
