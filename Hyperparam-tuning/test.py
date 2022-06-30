import sys
import math
import time
import pickle
import datetime
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


from pdbbind_utils import *
from CPI_model import *

def test(net, test_data, batch_size):
    net.eval()
    output_list = []
    label_list = []
    pairwise_auc_list = []
    
    criterion1 = nn.MSELoss()
    criterion2 = Masked_BCELoss() 

    total_loss = 0
    affinity_loss = 0
    pairwise_loss = 0

    for i in range(int(math.ceil(len(test_data[0])/float(batch_size)))):
        #loss_perf = []
        #loss_perf_1= []

        input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, aff_label, pairwise_mask, pairwise_label_orig = \
            [ test_data[data_idx][i*batch_size:(i+1)*batch_size] for data_idx in range(9)]
        
        inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq]
        vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process(inputs)
        
        aff_label = torch.FloatTensor(aff_label)#.to(device)
        pairwise_mask = torch.FloatTensor(pairwise_mask)#.to(device)
        pairwise_label = torch.FloatTensor(pad_label_2d(pairwise_label_orig, vertex, sequence))#.to(device)
        
        affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
        
        loss_aff = criterion1(affinity_pred, aff_label)
        loss_pairwise = criterion2(pairwise_pred, pairwise_label, pairwise_mask, vertex_mask, seq_mask)
        #loss = loss_aff + 0.1*loss_pairwise
        #loss_perf_1 = [loss_aff, loss_pairwise,loss]

        #total_loss += float(loss.data*batch_size)
        affinity_loss += float(loss_aff.data*batch_size)
        pairwise_loss += float(loss_pairwise.data*batch_size)
        #loss_perf = [total_loss, affinity_loss, pairwise_loss]
        for j in range(len(pairwise_mask)):
            if pairwise_mask[j]:
                num_vertex = int(torch.sum(vertex_mask[j,:]))
                num_residue = int(torch.sum(seq_mask[j,:]))
                pairwise_pred_i = pairwise_pred[j, :num_vertex, :num_residue].detach().cpu().numpy().reshape(-1)
                pairwise_label_i = pairwise_label_orig[j].reshape(-1)
                #pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i)).cpu().detach().numpy().reshape(-1)
                pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i))
        output_list += affinity_pred.cpu().detach().numpy().reshape(-1).tolist()
        label_list += aff_label.reshape(-1).tolist()
    output_list = np.array(output_list)
    label_list = np.array(label_list)
    rmse_value, pearson_value, spearman_value = reg_scores(label_list, output_list)  
    average_pairwise_auc = np.mean(pairwise_auc_list)
    
    #writer1.add_graph(net,batch_data_process(inputs))
    test_performance = [rmse_value, pearson_value, spearman_value, average_pairwise_auc]
    auc = test_performance[3]
    return test_performance,label_list, output_list, auc, loss_aff, loss_pairwise #affinity_loss, pairwise_loss   