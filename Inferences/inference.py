from itertools import pairwise
import torch 
from CPI_model import *
from processing_input import generate_input

def predict():
    MONN.eval()
    with torch.no_grad():
        #predictions=MONN()
        affinity_pred, pairwise_pred = MONN(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)

if __name__ == "__main__":
    vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence=generate_input()
   
    #re instantiate the model (backbone)
    MONN= Net() #Net(init_A,init_B,init_W, params)
    state_model= torch.load("MONN-2.pth")
    #loading the dictionnary that containings model params
    MONN.load_state_dict(state_model)

    #load validation data to get sample or get sample for inference

    #make inference
    pairwise_pred , affinity_pred , pairwise_expected, affinity_expected=predict()  