#!/usr/bin/python3
__auteurs__ = "ODJE Floriane"
__data__ = "2022-06"

import torch 
import numpy as np
import pandas 
from math import log as ln 
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
            f.write(f"The predicted affinity (pKI or pKD) is {affinity_pred :.3f} in -log[M] \n ")
            f.write('Pairwise matrix : \n')
            print(tabulate(pw_frame,headers=list(sequence.upper()),showindex=molecule,tablefmt="grid"),file=f)

def convert_unit(pKIKD,unit):
    assert unit in ['nM','uM','mM','pM','fM']
    if unit == 'nM':
        pvalue = 10**(-pKIKD)*10**9
    elif unit == 'uM':
        pvalue = 10**(-pKIKD)*10**6
    elif unit == 'mM':
        pvalue = 10**(-pKIKD)*10**3
    elif unit == 'pM':
        pvalue = 10**(-pKIKD)*10**12
    elif unit == 'fM':
        pvalue = 10**(-pKIKD)*10**15
    return pvalue 

def transform_matrix(pairwise_mat):
    with np.nditer(pairwise_mat, op_flags=['readwrite']) as p:
        for val in p:
            if float(val)==0.0:
                val[...] =  int(99)
            else :
                val[...] = int(ln(val/(1-val)))
    return pairwise_mat

def treshold(pairwise_mat):
    with np.nditer(pairwise_mat, flags=['multi_index']) as p:
        for val in p:
            if val > 10e-2:
                print("%d [%s]" % val,( p.multi_index),end=' ')


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
    
    #convert raw pkikd output in KI
    pKIKD = affinity_pred.numpy().item()
    print(f"The predicted affinity (pKI or pKD) is {affinity_pred.numpy().item():.3f} in -log[M]")
    print("Desired unit ['nM','uM','mM','pM','fM'] : ",end="")
    unit=input()
    kikd= convert_unit(pKIKD,unit)
    print(f"The predicted affinity is {kikd:.3f} {unit}.")
    
    #output file 
    print("Named your output file : ",end='')
    nfile = input()      

    #Print Pairwise matrix output into a dataframe
    P_M = pairwise_pred.numpy()
    print(f"Dim of pairwise pred are {P_M.shape}")
    print(f'Predicted pairwise matrix ')
    pm_reshaped = P_M.reshape(P_M.shape[1],P_M.shape[2])
    np.save('PM',pm_reshaped)
    print(f"Max value {np.amax(pm_reshaped)}")
    treshold (pm_reshaped)
    pw_frame = pandas.DataFrame(pm_reshaped,index=mol,columns=list(seq.upper()))
    print(f"{pw_frame}")
    
    pm=transform_matrix(pm_reshaped)
    print(pandas.DataFrame(pm,index=mol,columns=list(seq.upper())))
    filing_output_file(nfile,seq,mol,pm,pKIKD)


    print(f'You have saved the data into the file {nfile}')
    #print(tabulate(pw_frame,headers='keys',showindex=mol,tablefmt="fancy_grid"))
    

