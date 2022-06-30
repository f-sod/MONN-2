#!/usr/bin/python3

__auteurs__ = ""
__update_by__="ODJE Floriane"
__data__ = "2022-03"

import os 
import pickle
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one


def get_seq_dict(pdbid, file_type):
	"""
	Parse the file, transform the 3 letters code 
	Into the one letter code 
	Return it as dictionnary where keys are the chain identifier,
	Values are the sequences  
	The hetatom part of the pdb file is no considered 

	Return 
	------
		Dict : seq_dict {'A':('ITGTSTVGVG....')}
	"""
	p = PDBParser()
	if os.path.exists('./pdb_files/'+pdbid+'.pdb'):
		structure = p.get_structure(pdbid, './pdb_files/'+pdbid+'.pdb')
	#elif os.path.exists('../pdbbind/refined-set/'+pdbid+'/'+pdbid+'_'+file_type+'.pdb'):
		#structure = p.get_structure(pdbid, '../pdbbind/refined-set/'+pdbid+'/'+pdbid+'_'+file_type+'.pdb')
	else:
		print(pdbid)
		return None
	seq_dict = {}
	for model in structure:
		for chain in model:
			chain_id = chain.get_id()
			if chain_id == ' ':
				continue
			seq = ''
			id_list = []
			for res in chain:
				if res.get_id()[0] != ' ':  
					continue
				try:
					seq+=three_to_one(res.get_resname()) 
				except:
					print(f"unexpected aa name, {res.get_resname()}")
				id_list.append(str(res.get_id()[1])+str(res.get_id()[2]))
			seq_dict[chain_id] = (seq,id_list)
	return seq_dict

if __name__ == "__main__":
	with open('out1.2_pdbid_list.txt') as f:
		pdbid_list = [line.strip() for line in f.readlines()]

	pdb_seq_dict = {}
	count_no_seq = 0
	count_no_pocket = 0
	i = 0
	for pdbid in pdbid_list:
		print (f"{i},{pdbid}")
		i += 1
		pdb_seq_dict[pdbid] = {}
		full_seq_dict = get_seq_dict(pdbid,'protein')
		pocket_seq_dict = get_seq_dict(pdbid,'pocket')
		if full_seq_dict == None:
			count_no_seq += 1
			continue
		if pocket_seq_dict == None:
			assert 0
			count_no_pocket += 1
			continue
		pdb_seq_dict[pdbid]['protein'] = full_seq_dict
		pdb_seq_dict[pdbid]['pocket'] = pocket_seq_dict

	print (f"count_no_seq, {count_no_seq}")
	print (f"count_no_pocket,{count_no_pocket}")
	print (f"seq_dict length ,{len(pdb_seq_dict)}")
	with open('out5_pocket_dict','wb') as f:
		pickle.dump(pdb_seq_dict, f, protocol=0)


