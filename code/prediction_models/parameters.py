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
    'genes': 20,
    'mat_ent': 20,
    'quality': 20,
    'reactions': 20,
    'cell_comp': 20,
    'reguls': 20,
    'mol_func': 20,
    'bio_proc': 20
}

ONLY_BILINEAR = True
#GENE_COMBINE = 'intersection'  # product, bilinear, concat, intersection
GENE_COMBINE = 'bilinear'  # product, bilinear, concat, intersection
#GENE_COMBINE = 'product'  # product, bilinear, concat, intersection

CUSTOM_OGGNN = False
TRANSFORMER = True