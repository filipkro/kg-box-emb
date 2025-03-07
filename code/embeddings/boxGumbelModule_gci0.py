from torch import nn
# from embeddings import losses as L
import torch as th

from box_embeddings.modules import BoxEmbedding
from box_embeddings.modules.intersection import GumbelIntersection, HardIntersection
from box_embeddings.modules.volume import BesselApproxVolume, HardVolume

class BoxGumbelModule(nn.Module):

    def __init__(self, nbr_classes,  embed_dims=50,
                 intersect_temp=1.0, vol_temp=1.0, device='cpu') -> None:
        super(BoxGumbelModule, self).__init__()

        self.nbr_classes = nbr_classes

        self.embed_dim = embed_dims
        self.device = device

        self.class_boxes = BoxEmbedding(num_embeddings=self.nbr_classes,
                                        embedding_dim=self.embed_dim)


        # create some hard intersection thingy to get `val` loss

        self.intersect = GumbelIntersection(intersection_temperature=intersect_temp)
        self.volume = BesselApproxVolume(intersection_temperature=intersect_temp,
                                         volume_temperature=vol_temp)
        self.vol_temp = vol_temp
        self.intersect_temp = intersect_temp
        self.val_intersect = HardIntersection()
        self.val_volume = HardVolume()

    def box_distance(self, box_a, box_b):
        center_a, offset_a = box_a
        center_b, offset_b = box_b
        dist = th.abs(center_a - center_b) - offset_a - offset_b
        return dist

    def inclusion_loss(self, box_a, box_b, gamma):
        dist_a_b = self.box_distance(box_a, box_b)
        _, offset_a = box_a
        loss = th.linalg.norm(th.relu(dist_a_b + 2*offset_a - gamma), dim=1)
        return loss
    
    def box_dist_gci0_loss(self, data, neg=False):
        gamma = -0.1
        embeddings = self.class_boxes(data)
        box_sub = self.get_center_offset(embeddings[...,0,:])
        box_sup = self.get_center_offset(embeddings[...,1,:])

        if neg:
            box_dist = self.box_distance(box_sub, box_sup)
            return th.linalg.norm(th.relu(-box_dist - gamma), dim=1) 
        else:
            return self.inclusion_loss(box_sub, box_sup, gamma=gamma)
    
    def get_center_offset(self, embedding):
        centers = (embedding.z + embedding.Z) / 2
        offsets = embedding.Z - centers
        return (centers, offsets)
    
    def forward(self, gci, gci_name, val=False):
        """get embedding instead???"""
        if gci_name != 'gci0':
            raise ValueError("_gci0 module only works with gci0 data")

        # P(p|c)
        embeddings = self.class_boxes(gci)
        sub_classes = embeddings[...,0,:]
        sup_classes = embeddings[...,1,:]
        # vol(intersect(box(p), box(c)))/vol(box(c))
        if val:
            x = th.exp(self.val_volume(self.val_intersect(sub_classes,
                                                          sup_classes)) -
                       self.val_volume(sub_classes)).clamp(min=0.0,max=1.0)
        else:    
            x = th.exp(self.volume(self.intersect(sub_classes, sup_classes)) -
                       self.volume(sub_classes)).clamp(min=0.0,max=1.0)

        return x
