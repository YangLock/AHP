'''
Utilities functions for the framework.
'''
import numpy as np
import argparse
import torch
import os
import time
import logging
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from torchmetrics import AveragePrecision

def parse_args():
    parser = argparse.ArgumentParser()
    
    ##### training hyperparameter #####
    parser.add_argument("--dataset_name", type=str, default='cora', help='dataset name: cora, coraA, citeseer, pubmed, dblp, dblpA')
    parser.add_argument("--dataset_path", type=str, default="/home/hefangzhou/work/hefangzhou/heprediction_data", help="The directory path of dataset")
    parser.add_argument("--ckpts_dir", type=str, default="./checkpoints", help="The director path of checkpoints")
    parser.add_argument("--logs_dir", type=str, default="./logs", help="The directory path of logs")
    parser.add_argument("--exp_name", type=str, default='', help="The name of current experiment")
    parser.add_argument("--seed", dest='fix_seed', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--gpu", type=int, default=4, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument("--exp_num", default=1, type=int, help='number of experiments')
    parser.add_argument("--epochs", default=200, type=int, help='number of epochs')
    parser.add_argument("--bs", default=32, type=int, help='batch size')
    parser.add_argument("--train_DG", default="epoch1:1", type=str, help='update ratio in epochs (D updates:G updates)')
    parser.add_argument("--testns", type=str, default='SMCA', help='test negative sampler')
    parser.add_argument("--clip", type=float, default='0.01', help='weight clipping')
    parser.add_argument("--training", type=str, default='wgan', help='loss objective: wgan, none')
    parser.add_argument("--D_lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--G_lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--memory_size", default=32, type=int, help="The size of memory bank")
    
    
    ##### Discriminator architecture #####
    parser.add_argument("--model", default='hnhn', type=str, help='discriminator: hnhn')
    parser.add_argument("--n_layers", default=1, type=int, help='number of layers')
    parser.add_argument("--alpha_e", default=0, type=float, help='normalization term for hnhn')
    parser.add_argument("--alpha_v", default=0, type=float, help='normalization term for hnhn')
    parser.add_argument("--dim_hidden", default=400, type=int, help='dimension of hidden vector')
    parser.add_argument("--dim_vertex", default=400, type=int, help='dimension of vertex hidden vector')
    parser.add_argument("--dim_edge", default=400, type=int, help='dimension of edge hidden vector')
    
    ##### Generator architecture #####
    parser.add_argument("--gen", type=str, default='MLP', help='generator: MLP')
    
    opt = parser.parse_args()
    return opt


def print_summary(args, logger: logging.Logger):
    # Summary of training information
    logger.info('========================================== Training Summary ==========================================')
    logger.info('    - DATASET = %s' % (args.dataset_name))
    logger.info('    - DATASET PATH = %s' % (args.dataset_path))
    logger.info('    - CKPTS DIR = %s' % (args.ckpts_dir))
    logger.info('    - LOGS DIR = %s' % (args.logs_dir))
    logger.info('    - EXP NAME = %s' % (args.exp_name))
    logger.info('    - IS FIX SEED = %s' % (args.fix_seed))
    logger.info('    - GPU INDEX = %s' % (args.gpu))
    logger.info('    - EXP NUM = %s' % (args.exp_num))
    logger.info('    - EPOCHS = %s' % (args.epochs))
    logger.info('    - BATCH SIZE = %s' % (args.bs))
    logger.info('    - DG UPDATE RATIO = %s' % (args.train_DG))
    logger.info('    - TEST NEGATIVE SAMPLER = %s' % (args.testns))
    logger.info('    - GRADIENT CLIP = ' + str(args.clip))
    logger.info('    - LOSS OBJECTIVE = %s' % (args.training))
    logger.info('    - D LEARNING RATE = ' + str(args.D_lr))
    logger.info('    - G LEARNING RATE = ' + str(args.G_lr))
    logger.info(' ')
    logger.info('    - DISCRIMINATOR = %s' % (args.model))
    logger.info('    - NUM LAYERS = ' + str(args.n_layers))
    logger.info('    - HIDDEN DIM = %s' % (args.dim_hidden))
    logger.info('    - VERTEX HIDDEN DIM = %s' % (args.dim_vertex))
    logger.info('    - EDGE HIDDEN DIM = %s' % (args.dim_edge))
    logger.info('    - ALPHA E = %s' % (args.alpha_e))
    logger.info('    - ALPHA V = %s' % (args.alpha_v))
    logger.info(' ')
    logger.info('    - GENERATOR = %s' % (args.gen))
    logger.info(' ')


def get_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    time_stamp = time.strftime('%m%d-%H:%M')
    
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
        
    fHandler = logging.FileHandler(os.path.join(args.logs_dir, f"{args.exp_name}_{args.dataset_name}_train_{time_stamp}.log"), mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    return logger
    
     
def gen_size_dist(hyperedges):
    size_dist = {}
    for edge in hyperedges:
        leng = len(edge)
        if leng not in size_dist :
            size_dist[leng] = 0
        size_dist[leng] += 1
    if 1 in size_dist:
        del size_dist[1]
    if 2 in size_dist:
        del size_dist[2]
    total = sum(v for k, v in size_dist.items())
    for i in size_dist:
        size_dist[i] = float(size_dist[i]) / total
    return size_dist  
 
 
def unsqueeze_onehot(onehot):
    edge_size = max(int(onehot.sum().item()), 1)
    onehot_shape = onehot.shape[0]
    unsqueeze = torch.zeros([edge_size, onehot_shape], device=onehot.device)
    nonzero_idx = onehot.nonzero()
    for i, idx in enumerate(nonzero_idx) :
        unsqueeze[i][idx]=1
    return unsqueeze 


def measure(label, pred):
    average_precision = AveragePrecision()
    auc_roc = metrics.roc_auc_score(np.array(label), np.array(pred))
    ap = average_precision(torch.tensor(pred), torch.tensor(label))
    return auc_roc, ap
