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

# no RNN
#train and evaluate

def train_and_eval(train_data, valid_data, test_data, params, batch_size=32,num_epoch=200):
    init_A, init_B, init_W = loading_emb(measure)
    net = Net(init_A, init_B, init_W, params)
    net.to(device)
    net.apply(weights_init)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"total num params: {pytorch_total_params}")
    net.train()
    criterion1 = nn.MSELoss()
    criterion2 = Masked_BCELoss() 
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) #GAMMA=0.5
    
    shuffle_index = np.arange(len(train_data[0]))
    for epoch in range(num_epoch):
        np.random.shuffle(shuffle_index)
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
                print(f"Epoch: {epoch} \nBatch: {i}")
                 
            input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, affinity_label, pairwise_mask, pairwise_label = \
                [ train_data[data_idx][shuffle_index[i*batch_size:(i+1)*batch_size]] for data_idx in range(9)]
             
            inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq]
            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process(inputs)
            
            affinity_label = torch.FloatTensor(affinity_label).to(device)
            pairwise_mask = torch.FloatTensor(pairwise_mask).to(device)
            pairwise_label = torch.FloatTensor(pad_label_2d(pairwise_label, vertex, sequence)).to(device)
            
            optimizer.zero_grad()
            affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence) #plente
            
            loss_aff = criterion1(affinity_pred, affinity_label)
            loss_pairwise = criterion2(pairwise_pred, pairwise_label, pairwise_mask, vertex_mask, seq_mask)
            loss = loss_aff + 0.1*loss_pairwise 
            writer.add_scalar("Total Loss 1 /train", loss.item(), epoch)
            writer.add_scalar("Affinity Loss 1 /train", loss_aff.item(), epoch)
            writer.add_scalar("Pairwise Loss 1 /train", loss_pairwise.item(), epoch)
            
            total_loss += float(loss.data*batch_size)
            affinity_loss += float(loss_aff.data*batch_size)
            pairwise_loss += float(loss_pairwise.data*batch_size)
  
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
        
            writer.add_scalar("Total Loss * batch /train", total_loss, epoch)
            writer.add_scalar("Affinity Loss * batch /train", affinity_loss, epoch)
            writer.add_scalar("Pairwise Loss * batch /train", pairwise_loss, epoch)          
        
        loss_list = [total_loss, affinity_loss, pairwise_loss]
        loss_name = ['total loss', 'affinity loss', 'pairwise loss']
        print_loss = [loss_name[i]+' '+str(round(loss_list[i]/float(len(train_data[0])), 6)) for i in range(len(loss_name))]
        print('epoch:',epoch, ' '.join(print_loss))
        
        perf_name = ['RMSE', 'Pearson', 'Spearman', 'avg pairwise AUC']
        
        train_performance, train_label, train_output, loss_perf, loss_perf_1 = test(net, train_data, batch_size)
        print_perf = [perf_name[i]+' '+str(round(train_performance[i], 6)) for i in range(len(perf_name))]
        writer.add_scalar("RMSE/train", train_performance[0], epoch)
        writer.add_scalar("Pearson/train", train_performance[1], epoch)
        writer.add_scalar("Spearman/train", train_performance[2], epoch)
        writer.add_scalar("avg pairwise AUC/train", train_performance[3], epoch)
        print('train', len(train_output), ' '.join(print_perf))
        
        valid_performance, valid_label, valid_output, loss_perf_valid, loss_perf_1_valid = test(net, valid_data, batch_size)
        print_perf = [perf_name[i]+' '+str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
        
        writer.add_scalar("RMSE/valid", valid_performance[0], epoch)
        writer.add_scalar("Pearson/valid", valid_performance[1], epoch)
        writer.add_scalar("Spearman/valid", valid_performance[2], epoch)
        writer.add_scalar("avg pairwise AUC/valid", valid_performance[3], epoch)
       
        writer.add_scalar("Total Loss 1 /valid",loss_perf_1_valid[0],epoch)
        writer.add_scalar("Affinity Loss 1 /valid",loss_perf_1_valid[1],epoch)
        writer.add_scalar("Pairwise Loss 1 /valid",loss_perf_1_valid[2],epoch)

        writer.add_scalar("Total Loss * batch /valid", loss_perf_valid[0], epoch)
        writer.add_scalar("Affinity Loss * batch /valid", loss_perf_valid[1], epoch)
        writer.add_scalar("Pairwise Loss * batch /valid", loss_perf_valid[2], epoch)               
        print('valid', len(valid_output), ' '.join(print_perf))

    
        test_performance, test_label, test_output, loss_perf_test, loss_perf_1_test = test(net, test_data, batch_size)
        print_perf = [perf_name[i]+' '+str(round(test_performance[i], 6)) for i in range(len(perf_name))]
        writer.add_scalar("RMSE/test", test_performance[0], epoch)
        writer.add_scalar("Pearson/test", test_performance[1], epoch)
        writer.add_scalar("Spearman/test", test_performance[2], epoch)
        writer.add_scalar("avg pairwise AUC/test", test_performance[3], epoch)
        
        writer.add_scalar("Total Loss 1 /test",loss_perf_1_test[0],epoch)
        writer.add_scalar("Affinity Loss 1 /test",loss_perf_1_test[1],epoch)
        writer.add_scalar("Pairwise Loss 1 /test",loss_perf_1_test[2],epoch)

        writer.add_scalar("Total Loss * batch /test",loss_perf_test[0], epoch)
        writer.add_scalar("Affinity Loss * batch /test", loss_perf_test[1], epoch)
        writer.add_scalar("Pairwise Loss * batch /test", loss_perf_test[2], epoch)  
        print('test ', len(test_output), ' '.join(print_perf))
    
    print('Finished Training')
    return test_performance, test_label, test_output

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
        loss_perf = []
        loss_perf_1= []

        input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, aff_label, pairwise_mask, pairwise_label_orig = \
            [ test_data[data_idx][i*batch_size:(i+1)*batch_size] for data_idx in range(9)]
        
        inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq]
        vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process(inputs)
        
        aff_label = torch.FloatTensor(aff_label).to(device)
        pairwise_mask = torch.FloatTensor(pairwise_mask).to(device)
        pairwise_label = torch.FloatTensor(pad_label_2d(pairwise_label_orig, vertex, sequence)).to(device)
        
        affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
        
        loss_aff = criterion1(affinity_pred, aff_label)
        loss_pairwise = criterion2(pairwise_pred, pairwise_label, pairwise_mask, vertex_mask, seq_mask)
        loss = loss_aff + 0.1*loss_pairwise
        loss_perf_1 = [loss_aff, loss_pairwise,loss]

        total_loss += float(loss.data*batch_size)
        affinity_loss += float(loss_aff.data*batch_size)
        pairwise_loss += float(loss_pairwise.data*batch_size)
        loss_perf = [total_loss, affinity_loss, pairwise_loss]
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
    return test_performance, label_list, output_list, loss_perf, loss_perf_1   

if __name__ == "__main__":
    
    #evaluate scheme
    measure = "KIKD" #sys.argv[1]  # IC50 or KIKD
    setting = "new compound" #sys.argv[2]   # new_compound, new_protein or new_new
    clu_thre = 0.3 #float(sys.argv[3])  # 0.3, 0.4, 0.5 or 0.6
    n_epoch = 10
    #n_rep = 10
    
    torch.cuda.empty_cache()
    #device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    writer1 = SummaryWriter(log_dir='runs/model')
    
    assert setting in ['new_compound', 'new_protein', 'new_new']
    assert clu_thre in [0.3, 0.4, 0.5, 0.6]
    assert measure in ['IC50', 'KIKD']
    GNN_depth, inner_CNN_depth, DMA_depth =  4,4,4 #4, 2, 2
    
    print("Processing the main part of the code ....")
    if setting == 'new_compound':
        n_fold = 5
        batch_size = 32
        k_head, kernel_size, hidden_size1, hidden_size2 = 2, 7, 80, 80 #2,7,128,128
    elif setting == 'new_protein':
        n_fold = 5
        batch_size = 32
        k_head, kernel_size, hidden_size1, hidden_size2 = 1, 5, 128, 128

    elif setting == 'new_new':
        n_fold = 5
        batch_size = 32
        k_head, kernel_size, hidden_size1, hidden_size2 = 1, 7, 128, 128
    para_names = ['GNN_depth', 'inner_CNN_depth', 'DMA_depth', 'k_head', 'kernel_size', 'hidden_size1', 'hidden_size2']
    
    params = [GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2]
    #params = sys.argv[4].split(',')
    #params = map(int, params)
    
    #print evaluation scheme
    print(f"Dataset: PDBbind v2020 with measurement : {measure}")
    print(f"Clustering threshold : {clu_thre}")
    print(f"Number of epochs : {n_epoch}")
    #print(f"Number of repeats : {n_rep}")
    print('Hyper-parameters :',[para_names[i]+':'+str(params[i]) for i in range(7)])
    
    rep_all_list = []
    rep_avg_list = []
    #for a_rep in range(n_rep):
    #load data
    data_pack, train_idx_list, valid_idx_list, test_idx_list = load_data(measure, setting, clu_thre, n_fold) 
    fold_score_list = []
    
    for a_fold in range(n_fold):
        #write the metrics on n_fold different file
        suffixe_fodler="fold_"+str(a_fold)
        log_folder = "runs/"+measure+"_"+setting+"_"+str(clu_thre)+"_"+suffixe_fodler+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_folder)
        
        print('fold', a_fold+1, 'begin')
        train_idx, valid_idx, test_idx = train_idx_list[a_fold], valid_idx_list[a_fold], test_idx_list[a_fold]
        print('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))
        
        train_data = data_from_index(data_pack, train_idx)
        valid_data = data_from_index(data_pack, valid_idx)
        test_data = data_from_index(data_pack, test_idx)
        
        test_performance, test_label, test_output = train_and_eval(train_data, valid_data, test_data, params, batch_size, n_epoch)
        rep_all_list.append(test_performance)
        fold_score_list.append(test_performance)
        print('-'*30)
    print(f"fold avg performance , {np.mean(fold_score_list,axis=0)}")
    rep_avg_list.append(np.mean(fold_score_list,axis=0))
    np.save('MONN_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre), rep_all_list)
#writer.flush()
#writer.close()

print('All repetitions done')
print('Print all stats: RMSE, Pearson, Spearman, avg pairwise AUC')
print(f"mean, {np.mean(rep_all_list, axis=0)}")
print(f"std, {np.std(rep_all_list, axis=0)}")
print('='*20)
print('print avg stats:  RMSE, Pearson, Spearman, avg pairwise AUC')
print(f"mean, {np.mean(rep_avg_list, axis=0)}")
print(f"std , {np.std(rep_avg_list, axis=0, dtype=np.float32)}")
print('Hyper-parameters:', [para_names[i]+':'+str(params[i]) for i in range(7)])
#np.save('CPI_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre)+'_'+'_'.join(map(str,params)), rep_all_list)
np.save('MONN_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre), rep_all_list)
