#!/usr/bin/python3
__auteurs__ = "ODJE Floriane"
__data__ = "2022-06"

import torch 
import numpy as np
import pandas 
from tabulate import tabulate
from collections import defaultdict
from itertools import pairwise
from CPI_model import Net
#from CPI_model import *
from processing_input import generate_input , loading_emb

def predict(MONN,processed_input):
    vertex_mask = processed_input[0]
    vertex = processed_input[1]
    edge = processed_input[2]
    atom_adj = processed_input[3]
    bond_adj = processed_input[4]
    nbs_mask = processed_input[5]
    seq_mask = processed_input[6]
    sequence = processed_input[7]
    
    MONN.eval()
    with torch.no_grad():
        #Pred
        nbs_mask = nbs_mask.unsqueeze(0) #ADD A DIMENSION BC OF THE BATCH 
        affinity_pred, pairwise_pred = MONN(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
        # predictions -> affinity_pred et pairwise pred (-> tensor dim(sizeofsample,numberofclassetobepredicted))
    return affinity_pred, pairwise_pred

def filing_output_file(name_file,sequence,molecule,pw_frame,affinity_pred):
        with open(name_file,'w') as f:
            f.write(f"Proteins : {sequence.upper()}\n")
            f.write(f"Molecule : {molecule} \n")
            f.write('------------------------------------- \n')
            f.write(f"The predicted affinity (KI or KD) is {affinity_pred} in -log[M] \n ")
            f.write('Pairwise matrix : \n')
            print(tabulate(pw_frame,headers=list(sequence.upper()),showindex=molecule,tablefmt="grid"),file=f)

if __name__ == "__main__":

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    
    #re instantiate the model (backbone)
    init_atom_features,init_bond_features,init_word_features=loading_emb()
    k_head, kernel_size, hidden_size1, hidden_size2, GNN_depth, inner_CNN_depth, DMA_depth = 1, 7, 128, 128, 4, 2, 2
    params = [GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2]
    MONN = Net(init_atom_features, init_bond_features, init_word_features,params)
    
    #loading the dictionnary that containings learnt model params
    state_model = torch.load("MONN-2.pth")
    MONN.load_state_dict(state_model)

    #load validation data to get sample or get sample from input  for inference
    processed_input,mol, seq = generate_input(atom_dict,bond_dict,word_dict)
    
    #make inference
    affinity_pred, pairwise_pred  = predict(MONN,processed_input)  
    #print(f'{pairwise_pred} \n {affinity_pred}')
    
    #convert raw output KI
    KIKD = affinity_pred.numpy().item()
        #KIKD = 10*np.log10(KIKD) 
    print(f"The predicted affinity (KI or KD) is {affinity_pred.numpy().item()} in -log[M]")
    #print(f"The predicted affinity (KI or KD) is {KIKD} in [M]")
    
    #output file 
    print("Named your output file : ",end='')
    nfile = input()
    
    #convert raw output non covalent bound
    P_M = pairwise_pred.numpy()
    print(f"Dim of pairwise pred are {P_M.shape}")
    print(f'Predicted pairwise matrix ')
    pm_reshaped = P_M.reshape(P_M.shape[1],P_M.shape[2])
    np.save('PM',pm_reshaped)
    filing_output_file(nfile,seq,mol,pm_reshaped,KIKD)
    pw_frame = pandas.DataFrame(pm_reshaped,index=mol,columns=list(seq.upper()))
    print(pw_frame)
    #print(tabulate(pw_frame,headers='keys',showindex=mol,tablefmt="fancy_grid"))
    print(f'You have saved the data into the file {nfile}')
