__auteurs__ = "ODJE Floriane"
__date__ = "2022-05"

import sys
import math
import time
import pickle
import optuna 
import datetime
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms



from pdbbind_utils import *
from CPI_model import *
from test import *

# Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy and loss
def objective(trial):

    #generate optimizers
    params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'GNN_depth': trial.suggest_int("GNN_depth", 2 , 10), 
            'inner_CNN_depth': trial.suggest_int("inner_CNN_depth", 2 , 10), 
            'DMA_depth': trial.suggest_int("DMA_depth", 2 , 10),
            'k_head': trial.suggest_int("k_head", 2 , 6),
            'hidden_size1': trial.suggest_int("hidden_size1", 80 , 180),
            'hidden_size2': trial.suggest_int("hidden_size2", 80 , 180),
            'num_epoch': trial.suggest_int("num_epoch", 20, 45)
            }
    
    #generate model
    init_A, init_B, init_W = loading_emb(measure)
    net = Net(init_A, init_B, init_W, kernel_size, params)
    
    rep_all_list = []
    rep_avg_list = []
    data_pack, train_idx_list, valid_idx_list, test_idx_list = load_data(measure, setting, clu_thre, n_fold) 
    fold_score_list = []
    all_auc, all_loss_aff, all_loss_pairwise = [], [], []
    for a_fold in range(n_fold):
        print('fold', a_fold+1, 'begin')
        train_idx, valid_idx, test_idx = train_idx_list[a_fold], valid_idx_list[a_fold], test_idx_list[a_fold]
        print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))
        
        train_data = data_from_index(data_pack, train_idx)
        valid_data = data_from_index(data_pack, valid_idx)
        test_data = data_from_index(data_pack, test_idx)
        
        test_performance,test_label, test_output,auc, loss_aff_test, loss_pair_test= train_and_eval(train_data, valid_data, test_data, kernel_size, batch_size,params,net)
        
        perf=[ auc, loss_aff_test, loss_pair_test]
        
        auc=perf[0]
        loss_aff=perf[1]
        loss_pairwise=perf[2]
        all_auc.append(auc)
        all_loss_aff.append(loss_aff)
        all_loss_pairwise.append(loss_pairwise)
        
        rep_all_list.append(test_performance)
        fold_score_list.append(test_performance)
        print('-'*30)
    print(f"fold avg performance , {np.mean(fold_score_list,axis=0)}")
    rep_avg_list.append(np.mean(fold_score_list,axis=0))
    
    auc_avg = sum(all_auc) / len(all_auc)
    loss_aff_avg = sum(all_loss_aff) / len(all_loss_aff)
    loss_pairwise_avg = sum(all_loss_pairwise) / len(all_loss_pairwise)
    #return float(auc), float(loss_aff), float(loss_pairwise)
    return float(auc_avg), float(loss_aff_avg), float(loss_pairwise_avg)
    
def train_and_eval(train_data, valid_data, test_data, kernel_size, batch_size,params,net):
    #net.apply(weights_init)
    #net.train()
    criterion1 = nn.MSELoss()
    criterion2 = Masked_BCELoss()  
    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"])
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.25) #GAMMA=0.5
    
    shuffle_index = np.arange(len(train_data[0]))
    for epoch in range(params["num_epoch"]):
        np.random.shuffle(shuffle_index)
        optimizer.step()
        scheduler.step()
        for param_group in optimizer.param_groups: 
            print(f"Learning rate: {param_group['lr']}")
        
        train_output_list = []
        train_label_list = []
        total_loss = 0
        affinity_loss = 0
        pairwise_loss = 0
            

        for i in range(int(len(train_data[0])/batch_size)):
            if i % 100 == 0:
                print(f"Epoch: {params['num_epoch']} \nBatch: {i}")
                    
            input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, affinity_label, pairwise_mask, pairwise_label = \
                [ train_data[data_idx][shuffle_index[i*batch_size:(i+1)*batch_size]] for data_idx in range(9)]
                
            inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq]
            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process(inputs)
            
            affinity_label = torch.FloatTensor(affinity_label)#.to(device)
            pairwise_mask = torch.FloatTensor(pairwise_mask)#.to(device)
            pairwise_label = torch.FloatTensor(pad_label_2d(pairwise_label, vertex, sequence))#.to(device)
            
            optimizer.zero_grad()
            affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence) #plente
            
            loss_aff = criterion1(affinity_pred, affinity_label)
            loss_pairwise = criterion2(pairwise_pred, pairwise_label, pairwise_mask, vertex_mask, seq_mask)
            loss = loss_aff + 0.1*loss_pairwise 

            total_loss += float(loss.data*batch_size)
            affinity_loss += float(loss_aff.data*batch_size)
            pairwise_loss += float(loss_pairwise.data*batch_size)

            loss.backward()
            #nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

        loss_list = [total_loss, affinity_loss, pairwise_loss]
        loss_name = ['total loss', 'affinity loss', 'pairwise loss']
        print_loss = [loss_name[i]+' '+str(round(loss_list[i]/float(len(train_data[0])), 6)) for i in range(len(loss_name))]
        print('epoch:',params['num_epoch'], ' '.join(print_loss))
        
        perf_name = ['RMSE', 'Pearson', 'Spearman', 'avg pairwise AUC']

        train_performance,train_label, train_output, auc ,  loss_perf, loss_perf_1 = test(net, train_data, batch_size)
        print_perf = [perf_name[i]+' '+str(round(train_performance[i], 6)) for i in range(len(perf_name))]
        print('train', len(train_output), ' '.join(print_perf))
        
        valid_performance, valid_label, valid_output, auc , loss_perf_valid, loss_perf_1_valid = test(net, valid_data, batch_size)
        print_perf = [perf_name[i]+' '+str(round(valid_performance[i], 6)) for i in range(len(perf_name))]             
        print('valid', len(valid_output), ' '.join(print_perf))

        test_performance,test_label, test_output, auc, loss_aff, loss_pair = test(net, test_data, batch_size)
        print_perf = [perf_name[i]+' '+str(round(test_performance[i], 6)) for i in range(len(perf_name))]
        print('test ', len(test_output), ' '.join(print_perf))

        print('Finished Training')
        perfs = [test_performance, auc, loss_aff, loss_pair ]
        #auc=perfs[1]
        #loss_affinity=perfs[2]
        #loss_pairwises=perfs[3]
        return  test_performance,test_label, test_output,auc, loss_aff, loss_pair

if __name__ == "__main__":
    measure = "KIKD"
    setting = "new_compound"
    clu_thre = 0.3
    
    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:1"if use_cuda else "cpu")
    
    print("Processing the main part of the code ....")
    if setting == 'new_compound':
        n_fold = 5
        batch_size = 32
        kernel_size = 7

    study = optuna.create_study(directions=["maximize", "minimize","minimize"],sampler=optuna.samplers.NSGAIISampler(),study_name='HP-tuning Monn-2')
    study.optimize(objective, n_trials=30)
    #study.add_trial(study.trials)
    
    print("Number of finished trials: ", len(study.trials))
    
    # Get best trial based on val accuracy and loss.
    best_val_accuracy, best_val_loss_aff, best_val_loss_pair = None, None, None
    best_val_accuracy_trial, best_val_loss_aff_trial, best_val_loss_pair_trial = None, None,None

    for t, vals in enumerate(study.trials):
        print(f'trial: {t}, params: {vals.params}, values: {vals.values} \n')
        accuracy, loss_aff, loss_pair = vals.values[0], vals.values[1], vals.values[2]
        if best_val_accuracy is None or accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_val_accuracy_trial = t

        if best_val_loss_aff is None or loss_aff < best_val_loss_aff:
            best_val_loss_aff = loss_aff
            best_val_loss_aff_trial = t

        if best_val_loss_pair is None or loss_pair < best_val_loss_pair : 
           best_val_loss_pair = loss_pair
           best_val_loss_pair_trial = t 

        print(f'\nbest trial by accuracy: {best_val_accuracy_trial}')
        print(f'best trial by affinity loss    : {best_val_loss_aff_trial}')
        print(f'best trial by pairwise loss: {best_val_loss_pair_trial}\n')

    # re-evaluate code to re-use best trials 
    #best_trial = study.best_trials
    #best_trial_copy = copy.deepcopy(best_trial)
    #objective(best_trial)
    
    # the user attribute is overwritten by re-evaluation
    #assert best_trial.user_attrs != best_trial_copy.user_attrs
    
    
    #for  val in best_trial: 
        #print(f"Best trials : {study.best_trials[val].params} \n")
    
  
