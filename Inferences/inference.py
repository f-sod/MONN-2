from collections import defaultdict
from itertools import pairwise
import torch 
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
        nbs_mask = nbs_mask.unsqueeze(0)
        affinity_pred, pairwise_pred = MONN(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
        # predictions -> affinity_pred et pairwise pred (-> tensor dim(sizeofsample,numberofclassetobepredicted))
    return affinity_pred, pairwise_pred

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

    #load validation data to get sample or get sample for inference
    processed_input = generate_input(atom_dict,bond_dict,word_dict)
    
    #make inference
    pairwise_pred , affinity_pred =predict(MONN,processed_input)  # pairwise_expected, affinity_expected

