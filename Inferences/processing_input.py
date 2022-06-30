from collections import defaultdict
import os
import pickle
import time 
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6


def onek_encoding_unk(x, allowable_set):
    """
    Retrieve the position of the atom currently being read ,
    Within the list given as an argument,
    in the so called function
    Return 
    ------
    list : The list is a boolean list containing TRUE or 1 at the right 
           position in the list    
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set)) #lambda arguements(variable taken): expression(def of the Function)  

def atom_features(atom):
    """ 
    Using the one-hot encoding function enables to transform all categorical data into numbers, 
    A bunch of binomial/binary variables 
    This function returns an atom features arrays of 0 and 1 for each element of the ligands.  
    1 list encodes 5 features of an atom 
    
    Return
    ------
    Numpy array :  N lists , each one characterising an atom by a vector of length 82 
                the first 63 bits embeds the symbol of the atoms, 
                the following 6 elements embeds the atom degree’s, 
                the next 6 the explicit valence, 
                the next 6 the implicit valence 
                the last bits : if aromatic or not
    """
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)

def bond_features(bond):
    """
    Retrieve the caracteristic bonds features for each element
    within a 6 bits vectors 
    Return
    -------
    Numpy array : A float vector of 6 elements where each positions 
                encode for a type of bonds. 
    """
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
    bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)

def Mol2Graph(mol):
    """
    Generate and store a large amount of data for the ligand
    Convert molecule into the main GNN input
    
    Return 
    ------
    Numpy array :   fatoms : Is an array related to the atoms features / one hot encoding
                             it is an array of [number_atoms x 1] dimension
                    fbonds : Same principle as fatoms ; dim [number_bonds x 1 ] 
                    atom_nb : It encodes as a matrix the neighbours of the atoms; dim [n_atoms , max_nb]
                    bond_nb : Same principle as atom_nb but for edges, dim [n_atoms x max_nb]
                    num_nbs : This list , 1D matrix of size [n_atoms x 1], 
                             represents the number of neighbours of the corresponding atom 
                    num_nbs_mat : same as num_nbs with bool instead of numbers 
    """
    
    idxfunc=lambda x:x.GetIdx()

    n_atoms = mol.GetNumAtoms()
    assert mol.GetNumBonds() >= 0

    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms,), dtype=np.int32) #to encode atom feature ID
    fbonds = np.zeros((n_bonds,), dtype=np.int32) #bond feature ID
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32) #to encode atom features 2d
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    num_nbs_mat = np.zeros((n_atoms,max_nb), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        fatoms[idx] = atom_dict[''.join(str(x) for x in atom_features(atom).astype(int).tolist())] 
    #print(fatoms)
    
    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom()) #retrieve idx of the atoms where the bond start
        a2 = idxfunc(bond.GetEndAtom()) #retrieve idx of the atoms where the bond end 
        idx = bond.GetIdx() #retrieve bond idx 
        fbonds[idx] = bond_dict[''.join(str(x) for x in bond_features(bond).astype(int).tolist())] 
        try:
            atom_nb[a1,num_nbs[a1]] = a2
            atom_nb[a2,num_nbs[a2]] = a1
        except:
            return [], [], [], [], []
        bond_nb[a1,num_nbs[a1]] = idx
        bond_nb[a2,num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
        
    for i in range(len(num_nbs)):
        num_nbs_mat[i,:num_nbs[i]] = 1
    print(fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat)
    return fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat

def Protein2Sequence(sequence, ngram=1):
    """ 
    Convert sequence to CNN input
    Basically you get an array instead of a sequence, 
    Amino acids are encoded by a random value, 
    each amino acid gets an assigned value word_dict that is used to encode the protein sequence. 
    
    Return
    ------
    Numpy array : [len(sequences) x 1 ]
    """
    sequence = sequence.upper()
    word_list = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]
    output = []
    for word in word_list:
        if word not in aa_list:
            output.append(word_dict['X'])
        else:
            output.append(word_dict[word])
    if ngram == 3:
        output = [-1]+output+[-1] # pad A QUOI CA SERT CE PAD
    print(output)
    print(np.array(output, np.int32))
    return np.array(output, np.int32)




def pad_label_2d(label, vertex, sequence):
	dim1 = vertex.size(1) #size usually get the number of element of an arr or whaterver put arg 1 idk 
	dim2 = sequence.size(1)
	a = np.zeros((len(label), dim1, dim2))
	for i, arr in enumerate(label):
		a[i, :arr.shape[0], :arr.shape[1]] = arr 
	#print(a)
	return a

def pack2D(arr_list):
	M = max_nb#max([x.shape[1] for x in arr_list])
	a = np.zeros((len(arr_list), M))
	for i, arr in enumerate(arr_list):
		n = arr.shape[0]
		a[i,0:n] = arr 
	#print(a)
	return a

def pack1D(arr_list):
    #when batch sixe diff from 1 it make all the input of same size, each input is re wrote in a vector of len = max(element of the batch) and zero are added to complete
	#N = max([x.shape[0] for x in arr_list])
	N= 1
	a = np.zeros((N,len(arr_list)))
	for i, arr in enumerate(arr_list):
		n = arr #.shape[0]
		a[0:n,i] = arr
	#print(a)
	return a

def pack1D_seq(arr_list):
	N = max([x.shape[0] for x in arr_list])
	a = np.zeros((len(arr_list),N))
	for i,arr in enumerate(arr_list[0]):
		n = len(arr_list)
		a[0:n,i] = arr
	print(a)
	return a
    
def get_mask(arr_list):
    #same as pack1d but instead of re-writing the seq or mol or infos in general it is only 1 with zero to complete the size 
	N=len(arr_list)
	a = np.zeros((len(arr_list), N))
	for i, arr in enumerate(arr_list):
		a[i,:arr] = 1
	#print(a)
	return a

def get_mask_seq(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        a[i,:arr.shape[0]] = 1
    return a

#embedding selection function
def add_index(input_array, ebd_size):
	n_vertex, n_nbs = np.shape(input_array) #batch_size
	n = n_nbs*n_vertex
	add_idx = np.array(list(range(0,ebd_size*1,ebd_size))*n) 
	add_idx = np.transpose(add_idx.reshape(-1,1))
	add_idx = add_idx.reshape(-1)
	new_array = input_array.reshape(-1)+add_idx
	return new_array

#function for generating processed data
def data_process(data):
	vertex, edge, atom_adj, bond_adj, nbs, sequence = data
	vertex_mask = get_mask(vertex)
	vertex = pack1D(vertex)
	edge = pack1D(edge)
	atom_adj = pack2D(atom_adj)
	bond_adj = pack2D(bond_adj)
	nbs_mask = pack2D(nbs)
	
	#pad proteins and make masks
	seq_mask = get_mask_seq(sequence)
	sequence = pack1D_seq(sequence)
	
	#add index
	atom_adj = add_index(atom_adj, np.shape(atom_adj)[1])
	bond_adj = add_index(bond_adj, np.shape(edge)[1])
	
    #convert to torch cuda data type
	vertex_mask = Variable(torch.FloatTensor(vertex))#.to(device)
	vertex = Variable(torch.LongTensor(np.array(vertex)))#.to(device)
	edge = Variable(torch.LongTensor(np.array(edge)))#.to(device)
	atom_adj = Variable(torch.LongTensor(np.array(atom_adj)))#.to(device)
	bond_adj = Variable(torch.LongTensor(np.array(bond_adj)))#.to(device)
	nbs_mask = Variable(torch.FloatTensor(nbs_mask))#.to(device)
	
	seq_mask = Variable(torch.FloatTensor(seq_mask))#.to(device)
	sequence = Variable(torch.LongTensor(np.array(sequence)))#.to(device)
	
	return vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence #vertex, edge, atom_adj, bond_adj, sequence


def load_blosum62():
	#from txt to dict for blossum matrix (similRITY MATRIX)
	blosum_dict = {}
	with open("blosum62.txt", "r") as f:
		f.readline()   # Skip first line
		for line in f:
			parsed = line.strip().split()
			blosum_dict[parsed[0]] = np.array(parsed[1:]).astype(float)	
	return blosum_dict

def loading_emb():
    measure = "KIKD"
	#load intial atom and bond features (i.e., embeddings)
    with open('../preprocessing/pdbbind_all_atom_dict_'+measure , 'rb') as f : 
        atom_dict = pickle.load(f)
    with open('../preprocessing/pdbbind_all_bond_dict_'+measure, 'rb') as f : 
        bond_dict = pickle.load(f)
    with  open('../preprocessing/pdbbind_all_word_dict_'+measure, 'rb') as f:
        word_dict = pickle.load(f)
    
    print(f"atom dict size: {len(atom_dict)}, bond dict size: {len(bond_dict)}, word dict size: {len(word_dict)}")
    init_atom_features = np.zeros((len(atom_dict), atom_fdim))
    init_bond_features = np.zeros((len(bond_dict), bond_fdim))
    init_word_features = np.zeros((len(word_dict), 20))
    
    for key,value in atom_dict.items():
        init_atom_features[value] = np.array(list(map(int, key)))
    
    for key,value in bond_dict.items():
        init_bond_features[value] = np.array(list(map(int, key)))
    
    blosum_dict = load_blosum62()
    for key, value in word_dict.items():
        if key not in blosum_dict:
            continue
    
    init_word_features[value] = blosum_dict[key] # float(np.sum(blosum_dict[key]))
    init_atom_features = (torch.FloatTensor(init_atom_features))#.to(device)
    init_bond_features = Variable(torch.FloatTensor(init_bond_features))#.to(device)
    init_word_features = Variable(torch.cat((torch.zeros(1,20),torch.FloatTensor(init_word_features)),dim=0))#.to(device)
    
    return init_atom_features, init_bond_features, init_word_features
    
def generate_input():

    print("Generating embeddings ... \n")
    #input_mol= input("Enter smile of the molecule:")
    #mol=Chem.MolFromSmiles(input_mol)
    caffeine = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    mol = Chem.MolFromSmiles(caffeine)
    #seq =input("Enter fasta seqence of your protein:")
    seq="METKGYHSLPEGLDMERRWGQVSQAVERSSLGPTERTDENNYMEIVNVSCVSGAIPNNSTQGSSKEKQELLPCLQQDNNRPGILTSDIKTELESKELSATVAES"
    # MGLYMDSVRDADYSYEQQNQQGSMSPAKIYQNVEQLVKFYKGNGHRPSTLSCVNTPLRSFMSDSGSSVNGGVMRAVVKSPIMCHEKSPSVCSPLNMTS
    # SVCSPAGINSVSSTTASFGSFPVHSPITQGTPLTCSPN "
    #caffeine = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    mol_inputs, seq_inputs = [], []
    fa, fb, anb, bnb, nbs_mat = Mol2Graph(mol)
    
    #Function to generate inputs call
    mol_inputs.append([fa, fb, anb, bnb, nbs_mat])
    seq_inputs.append(Protein2Sequence(seq, ngram=1))
   

    inputs = [fa,fb,anb,bnb,nbs_mat,seq_inputs]
    vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = data_process(inputs)
    processed_input=[vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence]
    #vertex, edge, atom_adj, bond_adj, sequence = data_process(inputs)
    return processed_input


if __name__ == "__main__":

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    
    """ 
    Create a dictionary with the 
    key the aa and assigns default values
    """ 
    for aa in aa_list:
        word_dict[aa]
    word_dict['X']
    
    #input processed
    t=generate_input()
    print(t)
    print("DOne")