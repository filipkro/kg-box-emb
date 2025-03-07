import torch as th
import pickle
from embeddings.boxRelel import BoxRelELModel


def boxRel_to_cpu(fp):
    model = BoxRelELModel(None, from_file=fp)
    model.module.set_device('cpu')
    with open(fp + '/model.pkl', 'wb') as fo:
        pickle.dump(model.module, fo)