EPOCHS = 300
LR = 1e-4
NEIGHBOR_SAMPLE_SIZE = 20
GNN_CHANNELS = [64, 64]
#NN_CHANNELS = [64, 8]
NN_CHANNELS = [64]
#NN_CHANNELS = []
LR_DECAY = 1
SCHEDULE_RATE = 10
TRAIN_EMBEDDING_EPOCH = 10000#900*int(EPOCHS / 4)
BOX_EMBEDDINGS = True
TRAIN_GENES = False
ONLY_GENE_BOXES = False
BOX_WEIGHT = 0.0
REGULARIZATION = 1e-1
#DATASET = 'pyg_graph_c_DMA30_fitness'
DATASET = 'pyg_graph_box_interactions_DMA30'
#DATASET = 'pyg_graph_box_interactions'
LINKS = ('genes', 'interacts', 'genes')
SPLIT = 'nodes'
MIN_NBR_EDGES = 1000
SEMANTIC_WEIGHT = 1e-1
NUM_BATCHES = 10
NEG_WEIGHT = 5e-2
DROP_OUT = 0.0
SEMANTIC_MEASURE = 'distance'
HEADS = 2
INTER_TYPE = 'gumbel'
VOL_TYPE = 'bessel'

RANDOM_INIT_EMBS = False
EMBEDDING_DIMS = {
    'genes': 8,
    'mat_ent': 20,
    'quality': 10,
    'reactions': 10,
    'cell_comp': 10,
    'reguls': 10,
    'mol_func': 10,
    'bio_proc': 10
}

<<<<<<< HEAD
ONLY_BILINEAR = False
=======
ONLY_BILINEAR = True
GENE_COMBINE = 'product'  # product, bilinear, concat, intersection
>>>>>>> e9f8d97ac29ab38c7b1ef678219a3c85f75f502f
