# %%
from boxGumbel_gci0 import BoxGumbelModel
import os, pickle
from argparse import ArgumentParser, Namespace
import torch.cuda as cuda

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE)
# %%
args = Namespace()
args.epochs = 10
args.batch_size = 2**20
args.embed_dims = 10
args.neg_ratio = 0.5
args.lr = 0.5
args.margin = 0
args.reg_norm = 0.1
args.offset_reg = 0.01
from_file = None
args.dataset = 'full'
args.delta=2
args.delta_focus = 2
args.neg_ratio_focus = 3
args.bump_weight = 1
args.focus_factor = 1.0
args.copy_dest = ''
args.volume_temp = 0.5
args.intersect_temp = 0.5
args.reg_factor = 1e-5
args.params_from_file = 0
# %%
# from embeddings.datasets import CollectedDatasets
parser = ArgumentParser()
parser.add_argument('--epochs', default=400, type=int)
parser.add_argument('--batch_size', default=2**15, type=int)
parser.add_argument('--embed_dims', default=25, type=int)
parser.add_argument('--neg_ratio', default=0.5, type=float)
parser.add_argument('--intersect_temp', default=0.5, type=float)
parser.add_argument('--volume_temp', default=0.5, type=float)
parser.add_argument('--reg_factor', default=1e-5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--from_file', default='', type=str)
parser.add_argument('--dataset', default='full', type=str)
parser.add_argument('--neg_ratio_focus', default=0.5, type=float)
parser.add_argument('--focus_factor', default=1.0, type=float)
parser.add_argument('--copy_dest', default='', type=str)
parser.add_argument('--bot_factor', default=1.0, type=float)
parser.add_argument('--params_from_file', default=0, type=int)
args = parser.parse_args()

from_file = None if args.from_file == '' else os.path.join(BASE, 'trained_models', args.from_file)
# %%
if args.dataset == 'full':
    ds = 'collected_gci0_dataset_w_bot.pkl'
elif args.dataset =='mat_ent':
    ds = 'split_datasets/collected_mat_ent_w_bot.pkl'
elif args.dataset =='process':
    ds = 'split_datasets/collected_process_w_bot.pkl'
elif args.dataset =='gene_prod':
    ds = 'split_datasets/collected_gene_prod_w_bot.pkl'
elif args.dataset =='quality':
    ds = 'split_datasets/collected_quality_w_bot.pkl'
elif args.dataset =='role':
    ds = 'split_datasets/collected_role_w_bot.pkl'
elif args.dataset =='root':
    ds = 'split_datasets/collected_root_w_bot.pkl'
elif args.dataset =='bio_proc':
    ds = 'split_datasets/collected_bio_proc_w_bot.pkl'
elif args.dataset =='cell_comp':
    ds = 'split_datasets/collected_cell_comp_w_bot.pkl'
elif args.dataset =='mol_func':
    ds = 'split_datasets/collected_mol_func_w_bot.pkl'
elif args.dataset =='reguls':
    ds = 'split_datasets/collected_reguls_w_bot.pkl'
elif args.dataset =='reactions':
    ds = 'split_datasets/collected_reactions_w_bot.pkl'
elif args.dataset =='genes':
    ds = 'split_datasets/collected_genes_w_bot.pkl'
else:
    ds = 'collected_gci0_dataset_w_bot.pkl'
    print(f"for gci0 data full dataset is used instead of {args.dataset}")
    args.dataset = 'full'
# ds = 'collected_dataset.pkl' if args.dataset == 'go' else 'collected_dataset_cat.pkl'
# ds = 'collected_gci0_dataset.pkl'
# ds = 'collected_gci0_dataset_w_bot.pkl'
print(f'loading dataset {ds}')
with open(os.path.join(BASE, f'datasets/{ds}'), 'rb') as fi:
    dataset = pickle.load(fi)
# %%
device = 'cuda' if cuda.is_available() else 'cpu'
dataset.set_device(device)
# model = Box2ELModel(dataset, from_file=from_file)
# %%
model = BoxGumbelModel(dataset, epochs=args.epochs, batch_size=args.batch_size,
                    embed_dims=args.embed_dims, neg_ratio=args.neg_ratio,
                    learning_rate=args.lr, vol_temp=args.volume_temp,
                    intersect_temp=args.intersect_temp, device=device,
                    from_file=from_file, copy_dest=args.copy_dest,
                    bot_factor=args.bot_factor,
                    reg_factor=args.reg_factor, dataset_label=args.dataset,
                    model_filepath=os.path.join(BASE, 'trained_models'),
                    params_from_file=args.params_from_file)

# %%
model.train()
