from collections import defaultdict
import os
import pickle
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist
import time


elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6


def onek_encoding_unk(x, allowable_set):
    """A PARTIR DE CE QUE CHERCHE RECUPER LA FONCTION (ATOM,DEGRS,VALENCE),\
     AJOUTE DE NVLLES VALEURS SI NON PRESENTE DANS KE SET INITIALE"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set)) #lambda arguements(variable taken): expression(def of the Function)  


def atom_features(atom):
    """ 
    Based on the previous function, will return a numpy array features containing \
    (degree,implicit-explit valence, if aromatic or not)of each atom 
    """
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    """
    Selon les tupes de liaison retourne un array ou 
    dtype=np.floatTurn the bool type into float true=0.0 
    """
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
    bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)



def Mol2Graph(mol):
    # convert molecule to GNN input
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
    print(f"{fatoms},{cid}")
    
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
    #print(fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat)
    return fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat


def Batch_Mol2Graph(mol_list):
    """
    GENERER UNE GRANDE QUANTITE DE DONNEE (BATCH); GRAPH FEATURES DES MOLECULE A PARTIR DE LA LISTE DE MOLECULE
    OBTENTION DE X TENSORS DE FEATURES SOUS FORME DE LISTE : RES
    UNZIP DE CHAQUE ELEMENT POUR AFFECTATION MLTIPLE  
    """
    res = list(map(lambda x:Mol2Graph(x), mol_list))
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)
    return fatom_list, fbond_list, gatom_list, gbond_list, nb_list


def Protein2Sequence(sequence, ngram=1):
    # convert sequence to CNN input
    ''' 
    en gros on obtient un array a la place de la sequence, dans lequel les lettre sont remplace par les valeurs par defaut 
    des key(AA) du dico 
    '''
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


def Batch_Protein2Sequence(sequence_list, ngram=3): #3 weird output with 3 
    res = list(map(lambda x:Protein2Sequence(x,ngram), sequence_list))
    return res


def get_mol_dict():
    """ 
    CONVERSION DES SDF EN OBJET 
    """
    if os.path.exists('../data/mol_dict'):
        with open('../data/mol_dict', 'rb') as f:
            mol_dict = pickle.load(f)
    else:
        mol_dict = {}
        mols = Chem.SDMolSupplier('../data/Components-pub.sdf')
        for m in mols:
            if m is None:
                continue
            name = m.GetProp("_Name")
            mol_dict[name] = m
        with open('../data/mol_dict', 'wb') as f:
            pickle.dump(mol_dict, f)
    #print('mol_dict',len(mol_dict))
    return mol_dict


def get_pairwise_label(pdbid, interaction_dict):
    if pdbid in interaction_dict and pdbid not in ['3rxj','3rme','1gbt']:
        sdf_element = np.array([atom.GetSymbol().upper() for atom in mol.GetAtoms()])
        atom_element = np.array(interaction_dict[pdbid]['atom_element'], dtype=str)
        atom_name_list = np.array(interaction_dict[pdbid]['atom_name'], dtype=str)
        atom_interact = np.array(interaction_dict[pdbid]['atom_interact'], dtype=int)
        nonH_position = np.where(atom_element != ('H'))[0]
        assert sum(atom_element[nonH_position] != sdf_element) == 0 
        #print(atom_name_list)
        
        atom_name_list = atom_name_list[nonH_position].tolist()
        pairwise_mat = np.zeros((len(nonH_position), len(interaction_dict[pdbid]['uniprot_seq'])), dtype=np.int32)
        for atom_name, bond_type in interaction_dict[pdbid]['atom_bond_type']:
            atom_idx = atom_name_list.index(str(atom_name))
            assert atom_idx < len(nonH_position)
            
            seq_idx_list = []
            for seq_idx, bond_type_seq in interaction_dict[pdbid]['residue_bond_type']:
                if bond_type == bond_type_seq:
                    seq_idx_list.append(seq_idx)
                    pairwise_mat[atom_idx, seq_idx] = 1
        if len(np.where(pairwise_mat != 0)[0]) != 0:
            pairwise_mask = True
            print(pairwise_mat)
            return True, pairwise_mat
    return False, np.zeros((1,1))


def get_fps(mol_list):
    fps = []
    for mol in mol_list:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useChirality=True)
        fps.append(fp)
    #print('fingerprint list',len(fps))
    return fps


def calculate_sims(fps1,fps2,simtype='tanimoto'):
    """
    compute similaryt and distance matrix for ligand  
    with tanimoto and dice 
    """
    sim_mat = np.zeros((len(fps1),len(fps2))) #,dtype=np.float32)
    for i in range(len(fps1)):
        fp_i = fps1[i]
        if simtype == 'tanimoto':
            sims = DataStructs.BulkTanimotoSimilarity(fp_i,fps2)
        elif simtype == 'dice':
            sims = DataStructs.BulkDiceSimilarity(fp_i,fps2)
        sim_mat[i,:] = sims
    return sim_mat


def compound_clustering(ligand_list, mol_list):
    """
    Clustering compounds based on their similarity 
    Clustering based on jaccard index
    """
    print ('start compound clustering...')
    fps = get_fps(mol_list)
    sim_mat = calculate_sims(fps, fps)
    #np.save('../preprocessing/'+MEASURE+'_compound_sim_mat.npy', sim_mat)
    print(f'compound sim mat, {sim_mat.shape}')
    C_dist = pdist(fps, 'jaccard')
    C_link = single(C_dist)
    for thre in [0.3, 0.4, 0.5, 0.6]:
        C_clusters = fcluster(C_link, thre, 'distance')
        len_list = []
        for i in range(1,max(C_clusters)+1):
            len_list.append(C_clusters.tolist().count(i))
        print( 'thre', thre, 'total num of compounds', len(ligand_list), 'num of clusters', max(C_clusters), 'max length', max(len_list))
        C_cluster_dict = {ligand_list[i]:C_clusters[i] for i in range(len(ligand_list))}
        with open('../preprocessing/'+MEASURE+'_compound_cluster_dict_'+str(thre),'wb') as f:
            pickle.dump(C_cluster_dict, f, protocol=0)


def protein_clustering(protein_list, idx_list):
    print ('start protein clustering...')
    protein_sim_mat = np.load('../data/pdbbind_protein_sim_mat.npy').astype(np.float32)
    sim_mat = protein_sim_mat[idx_list, :]
    sim_mat = sim_mat[:, idx_list]
    print ('original protein sim_mat', protein_sim_mat.shape, 'subset sim_mat', sim_mat.shape)
    #np.save('../preprocessing/'+MEASURE+'_protein_sim_mat.npy', sim_mat)
    P_dist = []
    for i in range(sim_mat.shape[0]):
        P_dist += (1-sim_mat[i,(i+1):]).tolist()
    print("Check P_dist - P_Link")
    P_dist = np.array(P_dist)
    print(P_dist)
    P_link = single(P_dist)
    print(P_link)
    for thre in [0.3, 0.4, 0.5, 0.6]:
        P_clusters = fcluster(P_link, thre,'distance')
        len_list = []
        for i in range(1,max(P_clusters)+1):
            len_list.append(P_clusters.tolist().count(i))
        print ('thre', thre, 'total num of proteins', len(protein_list), 'num of clusters', max(P_clusters), 'max length', max(len_list))
        P_cluster_dict = {protein_list[i]:P_clusters[i] for i in range(len(protein_list))} #initialement P_cluster[i] but seems to work with any value from 0-2 haven't tested over 2
        with open('../preprocessing/'+MEASURE+'_protein_cluster_dict_'+str(thre),'wb') as f:
            pickle.dump(P_cluster_dict, f, protocol=0)
    print("Pclus \n", P_clusters)
    print(P_cluster_dict)

def pickle_dump(dictionary, file_name):
    pickle.dump(dict(dictionary), open(file_name, 'wb'), protocol=0)

if __name__ == "__main__":
#    if len(sys.argv) != 2:
 #       sys.exit("ERREUR : il faut exactement un argument.")
 #   else : 
  #      assert sys.argv == 'KIKD' or 'IC50'
   #     print(f"Your preprocessing the data for the measure  {sys.argv[1]}")
    MEASURE = 'IC50'#sys.argv[1]    
    
    print ('Create dataset for measurement:', MEASURE)
    print ('Step 1/5, loading dict...')
    # load label dicts
    mol_dict = get_mol_dict()
    with open('../data/out7_final_pairwise_interaction_dict','rb') as f:
        interaction_dict = pickle.load(f)
    
    # initialize feature dicts
    wlnn_train_list = []
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    for aa in aa_list:
        word_dict[aa]
    word_dict['X']
    ''' 
    cree un dictionnaire avec comme key les aa et assigne des valeurs par defaut
    '''
    # get labels
    i = 0
    pair_info_dict = {}
    with open('../data/pdbbind_all_datafile.tsv','r') as f:
        print ('Step 2/5, generating labels...')
        for line in f.readlines():
            i += 1
            if i % 1000 == 0:
                print ('processed sample num', i)
            pdbid, pid, cid, inchi, seq, measure, label = line.strip().split('\t')           
            # filter interaction type and invalid molecules
            if MEASURE == 'All':
                pass
            elif MEASURE == 'KIKD':
                if measure not in ['Ki', 'Kd']:
                    continue
            elif measure != MEASURE:
                continue
            if cid not in mol_dict:
                print ('ligand not in mol_dict')
                continue
            mol = mol_dict[cid]
            
            # get labels
            value = float(label)
            pairwise_mask, pairwise_mat = get_pairwise_label(pdbid, interaction_dict)
            
            # handle the condition when multiple PDB entries have the same Uniprot ID and Inchi
            if inchi+' '+pid not in pair_info_dict:
                pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]
            else:
                if pair_info_dict[inchi+' '+pid][6]:
                    if pairwise_mask and pair_info_dict[inchi+' '+pid][3] < value:
                        pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]
                else:
                    if pair_info_dict[inchi+' '+pid][3] < value:
                        pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]
    
    print ('Step 3/5, generating inputs...')
    valid_value_list = []
    valid_cid_list = []
    valid_pid_list = []
    valid_pairwise_mask_list = []
    valid_pairwise_mat_list = []
    mol_inputs, seq_inputs = [], []
    
    # get inputs
    #caffeine = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    #mol=Chem.MolFromSmiles(caffeine)
    #fa, fb, anb, bnb, nbs_mat = Mol2Graph(mol)
    
    #get inputs
    for item in pair_info_dict:
        pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat = pair_info_dict[item]
        fa, fb, anb, bnb, nbs_mat = Mol2Graph(mol)
        print(f"{fa} ,{cid}")
        if fa == []:
            print ('num of neighbor > 6, ')
            continue
        mol_inputs.append([fa, fb, anb, bnb, nbs_mat])
        seq_inputs.append(Protein2Sequence(seq,ngram=1))
        valid_value_list.append(value)
        valid_cid_list.append(cid)
        valid_pid_list.append(pid)
        valid_pairwise_mask_list.append(pairwise_mask)
        valid_pairwise_mat_list.append(pairwise_mat)
        wlnn_train_list.append(pdbid)
                
    print ('Step 4/5, saving data...')
    # get data pack
    fa_list, fb_list, anb_list, bnb_list, nbs_mat_list = zip(*mol_inputs)
    data_pack = [np.array(fa_list,dtype=object), np.array(fb_list,dtype=object), np.array(anb_list,dtype=object), np.array(bnb_list,dtype=object), \
    np.array(nbs_mat_list,dtype=object), np.array(seq_inputs,dtype=object ), \
    np.array(valid_value_list,dtype=object), np.array(valid_cid_list,dtype=object), np.array(valid_pid_list,dtype=object), np.array(valid_pairwise_mask_list,dtype=object), np.array(valid_pairwise_mat_list,dtype=object)]
    
    # save data
    with open('../preprocessing/pdbbind_all_combined_input_'+MEASURE, 'wb') as f:
        pickle.dump(data_pack, f, protocol=0)
    
    np.save('../preprocessing/wlnn_train_list_'+MEASURE, wlnn_train_list)
    
    pickle_dump(atom_dict, '../preprocessing/pdbbind_all_atom_dict_'+MEASURE)
    pickle_dump(bond_dict, '../preprocessing/pdbbind_all_bond_dict_'+MEASURE)
    pickle_dump(word_dict, '../preprocessing/pdbbind_all_word_dict_'+MEASURE)
    
    print ('Step 5/5, clustering...')
    compound_list = list(set(valid_cid_list))
    protein_list = list(set(valid_pid_list))
    # compound clustering
    mol_list = [mol_dict[ligand] for ligand in compound_list]
    compound_clustering(compound_list, mol_list)
    
    # protein clustering
    ori_protein_list = np.load('../data/pdbbind_protein_list.npy')
    ori_protein_list = [str(id).replace('b','').strip("'") for id in ori_protein_list]
    idx_list=[]
    to_remove=[]
    for i,pid in enumerate(protein_list):
        if pid not in ori_protein_list:
            to_remove.append(pid)
            #to_remove.append(ori_protein_list.index(pid))
            continue
        else:
            idx_list.append(ori_protein_list.index(pid))
    for i in to_remove:
        protein_list.remove(i) 
        
    print(f"protein_list:{len(protein_list)}")
    print(f"idx_list:{len(idx_list)},{idx_list}")
    protein_clustering(protein_list, idx_list)



    print ('='*50)
    print (f"Finish generating dataset for measurement , {MEASURE}")
    print (f"Number of valid samples, {len(valid_value_list)}")
    print (f"Number of unique compounds,{len(compound_list)}")
    print (f"Number of unique proteins, {len(protein_list)}")
