import torch 
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torchmetrics import AveragePrecision
import os
import time
import logging
from tqdm import tqdm

import utils
from data_load import gen_data, gen_DGLGraph, load_train, load_val, load_test
import models
from sampler import *
from aggregator import *
from generator import MLPgenerator
from training import model_train, model_eval


def train(args, logger: logging.Logger, time_stamp):
    ckpt_path = os.path.join(args.ckpts_dir, f"{args.exp_name}_{args.dataset_name}_{time_stamp}")
    
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        
    
    if args.fix_seed:
        np.random.seed(0)
        torch.manual_seed(0)
    train_DG = args.train_DG.split(":")
    args.train_DG = [int(train_DG[0][5:]), int(train_DG[1]), int(train_DG[0][5:])+int(train_DG[1])]    
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    

    for j in tqdm(range(args.exp_num)):    
        # Load data
        args = gen_data(args, args.dataset_name)
        dataset_split_path = os.path.join(args.dataset_path, f"splits/{args.dataset_name}split{j}.pt")
        data_dict = torch.load(dataset_split_path)
        ground = data_dict["ground_train"] + data_dict["ground_valid"]
        g = gen_DGLGraph(args, ground)  
        train_batchloader = load_train(data_dict, args.bs, device) # only positives
        val_batchloader_pos = load_val(data_dict, args.bs, device, label="pos")
        val_batchloader_sns = load_val(data_dict, args.bs, device, label="sns")
        val_batchloader_mns = load_val(data_dict, args.bs, device, label="mns")
        val_batchloader_cns = load_val(data_dict, args.bs, device, label="cns")
    
        # Initialize models
        # Hypergraph neural network (HNHN)
        model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge], \
                        args.n_layers, memory_dim=args.nv, K=args.memory_size)
        model.to(device)     
        
        # Aggregator   
        Aggregator = None    
        cls_layers = [args.dim_vertex, 128, 8, 1]
        Aggregator = MaxminAggregator(args.dim_vertex, cls_layers)
        Aggregator.to(device)  
        
        # Generator
        size_dist = utils.gen_size_dist(ground)
        if args.gen == "MLP":
            dim = [64, 256, 256, args.nv]
            if args.dataset_name == "pubmed":
                dim = [128, 512, 512, args.nv]
            elif "dblp" in args.dataset_name:
                dim = [256, 1024, 2048, args.nv]
            logger.info(f"{args.dataset_name} generator dimension: "+str(dim))
            Generator =  MLPgenerator(dim, args.nv, device, size_dist)
        Generator.to(device)
        
        # Initialize AP metric
        average_precision = AveragePrecision(task="binary")

        best_roc = 0
        best_epoch = 0 
        
        # Initialize optimizer for descrimintor and generator
        optim_D = torch.optim.RMSprop(list(model.parameters())+list(Aggregator.parameters()), lr=args.D_lr)
        optim_G = torch.optim.RMSprop(Generator.parameters(), lr=args.G_lr)
        
        logger.info(f'============================================ Training (split:{j}) ==================================================')
        logger.info('#Epoch \t [Train ROC] \t [Train AP] \t [Train Loss] D | G \t [ROC] SNS | MNS | CNS | Mixed | Average \t [AP] SNS | MNS | CNS | Mixed | Average')

        patience_epoch = 0
        
        for epoch in range(args.epochs):            
            train_pred, train_label = [], []
            d_loss_sum, g_loss_sum, count  = 0.0, 0.0, 0
            
            # Train
            while True :
                # Iterate each hyperedge batch
                pos_hedges, pos_labels, is_last = train_batchloader.next()
                d_loss, g_loss, train_pred, train_label = model_train(args, g, model, Aggregator, Generator, optim_D, optim_G, pos_hedges, pos_labels, train_pred, train_label, device, epoch)
                d_loss_sum += d_loss
                g_loss_sum += g_loss
                count += 1
                if is_last :
                    break
                
            train_pred = torch.stack(train_pred)
            train_pred = train_pred.squeeze()
            train_label = torch.round(torch.cat(train_label, dim=0))        
            train_roc = metrics.roc_auc_score(np.array(train_label.cpu()), np.array(train_pred.cpu()))
            train_ap = average_precision(torch.tensor(train_pred), torch.tensor(train_label, dtype=torch.long))            
            
            train_d_loss = d_loss_sum / count
            train_g_loss = g_loss_sum / count
    
            # Evaluation phase           
            # 1. positive dataset + four negative datasets (SNS, MNS, CNS, and Mixed)
            val_pred_pos, total_label_pos = model_eval(args, val_batchloader_pos, g, model, Aggregator)  # POS
            val_pred_sns, total_label_sns = model_eval(args, val_batchloader_sns, g, model, Aggregator)  # SNS
            val_pred_mns, total_label_mns = model_eval(args, val_batchloader_mns, g, model, Aggregator)  # MNS
            val_pred_cns, total_label_cns = model_eval(args, val_batchloader_cns, g, model, Aggregator)  # CNS
            
            # POS + SNS validation set
            auc_roc_sns, ap_sns = utils.measure(total_label_pos+total_label_sns, val_pred_pos+val_pred_sns)
            
            # POS + MNS validation set
            auc_roc_mns, ap_mns = utils.measure(total_label_pos+total_label_mns, val_pred_pos+val_pred_mns)
            
            # POS + CNS validation set
            auc_roc_cns, ap_cns = utils.measure(total_label_pos+total_label_cns, val_pred_pos+val_pred_cns)

            # Mixed(POS + M + S + C) validation set
            l = len(val_pred_pos) // 3
            val_pred_mixed = val_pred_pos + val_pred_sns[0:l] + val_pred_mns[0:l] + val_pred_cns[0:l]
            total_label_mixed = total_label_pos + total_label_sns[0:l] + total_label_mns[0:l] + total_label_cns[0:l]
            auc_roc_mixed, ap_mixed = utils.measure(total_label_mixed, val_pred_mixed)
            
            # Calculate average AP and AUROC over four test set
            auc_roc_average = (auc_roc_sns + auc_roc_mns + auc_roc_cns + auc_roc_mixed) / 4
            ap_average = (ap_sns + ap_mns + ap_cns + ap_mixed) / 4
            
            # Log all info
            logger.info(f" {epoch}: \t {train_roc:.4f} \t     {train_ap:.4f} \t      {train_d_loss:.4f} {train_g_loss:.4f} \t     {auc_roc_sns:.4f} {auc_roc_mns:.4f} {auc_roc_cns:.4f} {auc_roc_mixed:.4f} {auc_roc_average:.4f} \t     {ap_sns:.4f} {ap_mns:.4f} {ap_cns:.4f} {ap_mixed:.4f} {ap_average:.4f}")
            
            # Save best checkpoint on auc_roc_average
            if best_roc < auc_roc_average:
                best_roc = auc_roc_average
                best_epoch = epoch
                patience_epoch = 0
                
                torch.save(model.state_dict(), os.path.join(ckpt_path, f"model_{j}.pkt"))
                torch.save(Aggregator.state_dict(), os.path.join(ckpt_path, f"Aggregator_{j}.pkt"))
                torch.save(Generator.state_dict(), os.path.join(ckpt_path, f"Generator_{j}.pkt"))
            else:
                patience_epoch += 1
                if patience_epoch >= args.patience:
                    logger.info("=== Early Stopping")
                    break
    
    logger.info(' ')
    logger.info(f"=====\t Split: {j} \t Best AUROC: {best_roc:.4f} Best Epoch: {best_epoch} \t=====")
    logger.info(' ')
    
    return args


def test(args, j, logger: logging.Logger, time_stamp): 
    # Get the correct path of checkpoints   
    ckpt_path = os.path.join(args.ckpts_dir, f"{args.exp_name}_{args.dataset_name}_{time_stamp}")

    logger.info(' ')
    logger.info(f'=========================================== Test (Split: {j}) ================================================')
    logger.info('[ROC] SNS | MNS | CNS | Mixed | Average \t [AP] SNS | MNS | CNS | Mixed | Average')
    
    # Load data
    dataset_split_path = os.path.join(args.dataset_path, f"splits/{args.dataset_name}split{j}.pt")
    data_dict = torch.load(dataset_split_path)
    args = gen_data(args, args.dataset_name)
    ground = data_dict["ground_train"] + data_dict["ground_valid"]
    g = gen_DGLGraph(args, ground)
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'

    # test set    
    test_batchloader_pos = load_test(data_dict, args.bs, device, label="pos")
    test_batchloader_sns = load_test(data_dict, args.bs, device, label="sns")
    test_batchloader_mns = load_test(data_dict, args.bs, device, label="mns")
    test_batchloader_cns = load_test(data_dict, args.bs, device, label="cns")
   
    # Initialize models
    model = models.multilayers(models.HNHN, [args.input_dim, args.dim_vertex, args.dim_edge], \
                    args.n_layers, memory_dim=args.nv, K=args.memory_size)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, f"model_{j}.pkt")))
    cls_layers = [args.dim_vertex, 128, 8, 1]
    Aggregator = MaxminAggregator(args.dim_vertex, cls_layers)
    Aggregator.to(device)
    Aggregator.load_state_dict(torch.load(os.path.join(ckpt_path, f"Aggregator_{j}.pkt")))
    
    model.eval()
    Aggregator.eval()

    # Test phase
    with torch.no_grad():
        # 1. positive dataset + four negative datasets (SNS, MNS, CNS, and MIXED)
        test_pred_pos, total_label_pos = model_eval(args, test_batchloader_pos, g, model, Aggregator)  # POS
        test_pred_sns, total_label_sns = model_eval(args, test_batchloader_sns, g, model, Aggregator)  # SNS
        test_pred_mns, total_label_mns = model_eval(args, test_batchloader_mns, g, model, Aggregator)  # MNS
        test_pred_cns, total_label_cns = model_eval(args, test_batchloader_cns, g, model, Aggregator)  # CNS
        
        # POS + SNS test set
        auc_roc_sns, ap_sns = utils.measure(total_label_pos+total_label_sns, test_pred_pos+test_pred_sns)
        
        # POS + MNS test set
        auc_roc_mns, ap_mns = utils.measure(total_label_pos+total_label_mns, test_pred_pos+test_pred_mns)
        
        # POS + CNS test set
        auc_roc_cns, ap_cns = utils.measure(total_label_pos+total_label_cns, test_pred_pos+test_pred_cns)
        
        # Mixed(POS + M + S + C) test set
        l = len(test_pred_pos)//3
        test_pred_mixed = test_pred_pos + test_pred_sns[0:l] + test_pred_mns[0:l] + test_pred_cns[0:l]
        total_label_mixed = total_label_pos + total_label_sns[0:l] + total_label_mns[0:l] + total_label_cns[0:l]
        auc_roc_mixed, ap_mixed = utils.measure(total_label_mixed, test_pred_mixed)
        
        auc_roc_average = (auc_roc_sns + auc_roc_mns + auc_roc_cns + auc_roc_mixed) / 4
        ap_average = (ap_sns + ap_mns + ap_cns + ap_mixed) / 4
        
        logger.info(f"    {auc_roc_sns:.4f} {auc_roc_mns:.4f} {auc_roc_cns:.4f} {auc_roc_mixed:.4f} {auc_roc_average:.4f} \t     {ap_sns:.4f} {ap_mns:.4f} {ap_cns:.4f} {ap_mixed:.4f} {ap_average:.4f}")

        
if __name__ == "__main__":
    time_stamp = time.strftime('%m%d-%H:%M')
    exp_name = "exp1"
    
    # train
    args = utils.parse_args()
    args.exp_name = exp_name
    logger = utils.get_logger(args)
    utils.print_summary(args, logger)
    train(args, logger, time_stamp)
    
    # test
    args = utils.parse_args()
    args.exp_name = exp_name
    for j in range(args.exp_num):
        test(args, j, logger, time_stamp)
