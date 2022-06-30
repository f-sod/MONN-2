#!/usr/bin/python3

__auteurs__ = ""
__update_by__="ODJE Floriane"
__data__ = "2022-03"

from rdkit import Chem
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one
import os
import pickle


def get_pdbid_list():
	""" 
	Read and Remove the end character of each line 
	of the file out1.2_pdbid_list.txt, 
	which contain all the pdb identifier as a column.

	Return
	------
		List : pdbid_list of 19443 elements 
	"""
	pdbid_list = []
	with open('out1.2_pdbid_list.txt') as f:
		for line in f.readlines():
			pdbid_list.append(line.strip())
	print(f"pdbid_list,{len(pdbid_list)}")
	return pdbid_list

def get_bonds(pdbid, ligand, atom_idx_list):
	""" 
	Parses the PLIP output file. 
	Retrives identifiers/name of each field 
	As well as the type of the bond 
	The indexes of the elements/atoms interacting

	Return
	------
		List : bond_list (huge_list : size depends on the pair ligand-prot)
	"""
	bond_list = []
	f = open('./plip_result/'+pdbid+'_output.txt')
	isheader = False
	for line in f.readlines():
		if line[0] == '*':
			bond_type = line.strip().replace('*','')
			isheader = True
		if line[0] == '|':
			if isheader:
				header = line.replace(' ','').split('|')
				isheader = False
				continue
			lines = line.replace(' ','').split('|')
			if ligand not in lines[5]:
				continue
			aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], lines[4], lines[5], lines[6] # initialement int(lines[4])
				#continue
			if bond_type in ['Hydrogen Bonds', 'Water Bridges'] :
				atom_idx1, atom_idx2 = int(lines[12]), int(lines[14])
				if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:   # discard ligand-ligand interaction
					continue
				if atom_idx1 in atom_idx_list:
					atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
				elif atom_idx2 in atom_idx_list:
					atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
				else:
					print(f"{pdbid},{ligand},{bond_type}, error: atom index in plip result not in atom_idx_list")
					print(atom_idx1, atom_idx2)
					return None
				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
			elif bond_type == 'Hydrophobic Interactions':
				atom_idx_ligand, atom_idx_protein = int(lines[8]), int(lines[9])
				if  atom_idx_ligand not in atom_idx_list: 
					continue
				elif atom_idx_ligand not in atom_idx_list:
					print('error: atom index in plip result not in atom_idx_list')
					print(f"Hydrophobic Interactions, {atom_idx_ligand}, {atom_idx_protein}")
					return None
				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
			elif bond_type in ['pi-Stacking', 'pi-Cation Interactions']:
				atom_idx_ligand_list = list(map(int, lines[12].split(','))) #initalement lines[11]
				if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
					print(f"{bond_type}, error: atom index in plip result not in atom_idx_list")
					print(atom_idx_ligand_list)
					return None
				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain, ligand_name, ligand_id, atom_idx_ligand_list))
			elif bond_type == 'Salt Bridges':
				print("salt Bridges")
				atom_idx_ligand_list = list(set(map(int, lines[11].split(',')))) #initialement lines[10]
				if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
					print('error: atom index in plip result not in atom_idx_list')
					print('Salt Bridges', atom_idx_ligand_list, set(atom_idx_ligand_list).intersection(set(atom_idx_list)))
					return None
				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain, ligand_name, ligand_id, atom_idx_ligand_list))
			elif bond_type == 'Halogen Bonds':
				atom_idx1, atom_idx2 = int(lines[11]), int(lines[13])
				if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:   # discard ligand-ligand interaction
					continue
				if atom_idx1 in atom_idx_list:
					atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
				elif atom_idx2 in atom_idx_list:
					atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
				else:
					print('error: atom index in plip result not in atom_idx_list')
					print(f"Halogen Bonds, {atom_idx1},{atom_idx2}")
					return None
				bond_list.append((bond_type+'_'+str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein], ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
			else:
				print('bond_type',bond_type)
				print(header)
				print(lines)
				return None
	f.close()
	if len(bond_list) != 0:
		return bond_list

def get_atoms_from_pdb(ligand, pdbid): 
	"""
	From pdb protein structure, get a ligand index list,
	for bond extraction using Function from module biophython.

	Return
	------
		List : atom_idx_list [2439, 2440, 2441, ...]
		List : atom_name_list ['OH1', 'N1', 'C1', ...]
	"""  
	p = PDBParser()
	atom_idx_list = []
	atom_name_list = []
	structure = p.get_structure(pdbid, './pdb_files/'+pdbid+'.pdb')
	for model in structure:
		for chain in model:
			chain_id = chain.get_id()
			id_list = []
			for res in chain:
				if ligand == res.get_resname():
					if res.get_id()[0] == ' ':
						continue
					for atom in res:
						atom_idx_list.append(atom.get_serial_number())
						atom_name_list.append(atom.get_id())
	if len(atom_idx_list) != 0:
		return atom_idx_list, atom_name_list
	else:
		return None, None

def get_mol_from_ligandpdb(ligand):
	"""
	Retrieve the ligand index and the name or their atom,
	Using function from module biophython.

	Return
	------
		List : name_order_list ['OH1', 'N1', 'C1', ...]
		Dict : name_to_idx_dict {'OH1': 0, 'N1': 1, 'C1': 2}
		Dict : name_to_element_dict {'OH1': 'O', 'N1': 'N', ...}
	"""  
	if not os.path.exists('./pdb_files/'+ligand+'_ideal.pdb') or os.stat('./pdb_files/'+ligand+'_ideal.pdb').st_size == 0 :
		print(f"File of {ligand} is empty or doesn't exists")
		return None, None, None
	name_order_list = []
	name_to_idx_dict, name_to_element_dict = {}, {}
	p = PDBParser()
	structure = p.get_structure(ligand,'./pdb_files/'+ligand+'_ideal.pdb')
	for model in structure:
		for chain in model:
			chain_id = chain.get_id()
			for res in chain:
				if ligand == res.get_resname():
					#print(ligand,res.get_resname(),res.get_full_id())
					for atom in res:
						name_order_list.append(atom.get_id())
						name_to_element_dict[atom.get_id()] = atom.element
						name_to_idx_dict[atom.get_id()] = atom.get_serial_number()-1
	#print('check', name_to_idx_dict.items())
	if len(name_to_idx_dict) == 0:
		return None, None, None
	return name_order_list, name_to_idx_dict, name_to_element_dict


def get_interact_atom_name(atom_idx_list, atom_name_list,bond_list):
	"""
	Based on the previous created list 
	This code retrieves, first, the names of the atons involved in interactions
	Then, the type of interaction and the names of the associated atoms 
	
	Return 
	------
		List : interact_atom_name_list ['CD2', 'N', ..]
		List : interact_bond_type_list [('CD2', 'Hydrophobic Interactions_0'), ('N', 'Hydrogen Bonds_1') ..]
	"""
	interact_atom_name_list = []
	interact_bond_type_list = []
	interact_atom_name_set = set()
	assert len(atom_idx_list) == len(atom_name_list)
	for bond in bond_list:
		for atom_idx in bond[-1]:
			atom_name = atom_name_list[atom_idx_list.index(atom_idx)]
			#if atom_name not in interact_atom_name_set:
			interact_atom_name_set.add(atom_name)
			interact_atom_name_list.append(atom_name)
			interact_bond_type_list.append((atom_name, bond[0]))
	return interact_atom_name_list, interact_bond_type_list

def get_interact_atom_list(name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict, interact_atom_name_list):
	"""
	Same as the previous function but return more elements like
	a binary list telling if the correspond atom is interacting 
	or not. 

	Return
	------
		List : atom_idx_list [2439, 2440, 2441, ..]
		List : atom_name_list ['N', 'CA', 'C', ..]
		List : atom_element_list ['O', 'N', 'C', ..]
		List : atom_interact_list [0, 0, 0, 0 , .. ]
	"""
	atom_idx_list = []
	atom_name_list = []
	atom_element_list = []
	atom_interact_list = []
	for name in name_order_list:
		idx = atom_name_to_idx_dict[name]
		atom_idx_list.append(idx)
		atom_name_list.append(name)
		atom_element_list.append(atom_name_to_element_dict[name])
		atom_interact_list.append(int(name in interact_atom_name_list))
	return atom_idx_list, atom_name_list, atom_element_list, atom_interact_list

def get_seq(pdbid):
	"""
	Parse the file, transform the 3 letters code 
	Into the one letter code 
	Return it as dictionnary where keys are the chain identifier,
	Values are the sequences  
	The hetatom part of the pdb file is no considered 

	Return 
	------
		Dict : seq_dic {'A':('ITGTSTVGVG....')}
		Dict : idx_to_aa_dict {'A1': 'I', 'A2': 'T', 'A3': 'G'...')}
	"""
	p = PDBParser()
	structure = p.get_structure(pdbid, './pdb_files/'+pdbid+'.pdb')
	seq_dict = {}
	idx_to_aa_dict = {}
	for model in structure:
		for chain in model:
			chain_id = chain.get_id()
			if chain_id == ' ':
				continue
			seq = ''
			id_list = []
			for res in chain:
				if res.get_id()[0] != ' ' or res.get_id()[2] != ' ':   # remove HETATM
					continue
				try:
					seq+=three_to_one(res.get_resname())
					idx_to_aa_dict[chain_id+str(res.get_id()[1])+res.get_id()[2].strip()] = three_to_one(res.get_resname())
				except:
					print(f"unexpected aa name, {res.get_resname()}")
				id_list.append(res.get_id()[1])
			seq_dict[chain_id] = (seq,id_list)
	return seq_dict, idx_to_aa_dict

def get_interact_residue(idx_to_aa_dict, bond_list):
	"""
	Same as interact_atom_name function but for the amino acids
	Meaning that , here , we returns the atoms of the AA, its number 
	on the proetic chain and the type of interaction which it's involved in 
	
	Return : 
	------
		List : interact_residue [('A242', 'N', 'Hydrophobic Interactions_0'), ...]
	"""
	interact_residue = []
	for bond in bond_list:
		if bond[1]+str(bond[3]) not in idx_to_aa_dict:
			continue
		aa = idx_to_aa_dict[bond[1]+str(bond[3])]
		assert three_to_one(bond[2]) == aa
		interact_residue.append((bond[1]+str(bond[3]), aa, bond[0]))
	if len(interact_residue) != 0:
		return interact_residue
	else:
		return None

if __name__ == "__main__":
	"""
	The code aims to generate a huge dictionary called interaction_dict, 
	Which , has , as keys the PDB id combined with the ligand 
	And give access to the bond list, the atom_idx_list, atom_name_list, 
	atom_element_list, atom_interact_list, interact_bond_type_list, 
	the sequence_dict and the interact_residue_list. 
	"""
	pdbid_to_ligand_file = open("pdbid_to_ligand_dict",'rb')
	pdbid_to_ligand = pickle.load(pdbid_to_ligand_file)
	
	no_valid_ligand = 0
	no_such_ligand_in_pdb_error = 0
	no_interaction_detected_error = 0
	no_ideal_pdb_error = 0
	empty_atom_interact_list = 0
	protein_seq_error = 0
	
	i = 0
	interaction_dict = {}
	pdbid_list = get_pdbid_list()
	for pdbid in pdbid_list:
		i += 1
		print(i, pdbid)
		if pdbid not in pdbid_to_ligand:
			no_valid_ligand += 1
			continue
		ligand = pdbid_to_ligand[pdbid]
		
		# get bond
		atom_idx_list, atom_name_list =  get_atoms_from_pdb(ligand, pdbid)  
		if atom_idx_list is None:
			no_such_ligand_in_pdb_error += 1
			print(f"no such ligand in pdb, pdbid : {pdbid} and ligand :{ligand}")
			continue
		bond_list = get_bonds(pdbid, ligand, atom_idx_list)
		if bond_list is None:
			print(f"empty bond list: pdbid: {pdbid},ligand: {ligand}, atom_idx_list: {len(atom_idx_list)}")
			no_interaction_detected_error += 1
			continue
		interact_atom_name_list, interact_bond_type_list = get_interact_atom_name(atom_idx_list, atom_name_list,bond_list)
	
		name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict = get_mol_from_ligandpdb(ligand)
		if atom_name_to_idx_dict == None:
			no_ideal_pdb_error+=1
			continue
		atom_idx_list, atom_name_list, atom_element_list, atom_interact_list \
		= get_interact_atom_list(name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict, interact_atom_name_list)
		if len(atom_idx_list) == 0:
			empty_atom_interact_list+=1
			continue
		
		# get sequence interaction information
		seq_dict, idx_to_aa_dict = get_seq(pdbid)
		interact_residue_list = get_interact_residue(idx_to_aa_dict, bond_list)
		if interact_residue_list is None:
			protein_seq_error += 1
			continue
		
		interaction_dict[pdbid+'_'+ligand] = {}
		interaction_dict[pdbid+'_'+ligand]['bond'] = bond_list
		interaction_dict[pdbid+'_'+ligand]['atom_idx'] = atom_idx_list
		interaction_dict[pdbid+'_'+ligand]['atom_name'] = atom_name_list
		interaction_dict[pdbid+'_'+ligand]['atom_element'] = atom_element_list
		interaction_dict[pdbid+'_'+ligand]['atom_interact'] = atom_interact_list
		interaction_dict[pdbid+'_'+ligand]['atom_bond_type'] = interact_bond_type_list
		
		interaction_dict[pdbid+'_'+ligand]['sequence'] = seq_dict
		interaction_dict[pdbid+'_'+ligand]['residue_interact'] = interact_residue_list
		
	
	print(f"interaction_dict, {len(interaction_dict)}")
	print(f"no_valid_ligand error, {no_valid_ligand}")
	print(f"no_such_ligand_in_pdb_error, {no_such_ligand_in_pdb_error}")
	print(f"no_interaction_detected_error, {no_interaction_detected_error}")
	print(f"no_ideal_pdb_error', {no_ideal_pdb_error}")
	print(f"empty_atom_interact_list,{empty_atom_interact_list}")
	print(f"protein_seq_error',{protein_seq_error}")
	
	with open('out4_interaction_dict', 'wb') as f:
		pickle.dump(interaction_dict, f, protocol=0)
