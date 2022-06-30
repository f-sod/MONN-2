#!/usr/bin/python3

__auteurs__ = ""
__update_by__="ODJE Floriane"
__data__ = "2022-03"


def get_pdbid_list():
	""" 
	Extract PDB IDs of all protein-ligand complex
	While parsing INDEX_general_PL.2020

	Return
	------
	List : pdbid_list of 19443 elements 
	"""
	pdbid_list = []
	with open('./pdbbind_index/INDEX_general_PL.2020') as f1:
		for line in f1.readlines():
			if line[0] != '#':
				pdbid_list.append(line.strip().split()[0])
	print(f"pdbid_list,{len(pdbid_list)}")
	return pdbid_list




def get_pdbid_to_ligand():
	"""
	Extract ligand IDs of all protein-ligand complex
	While parsing INDEX_general_PL.2020

	Return
	------
	List: pdbid_to_ligand of 16726 elements 
	"""
	pdbid_to_ligand = {}
	with open('./pdbbind_index/INDEX_general_PL.2020') as f:
		for line in f.readlines():
			if line[0] != '#':
				ligand = line.strip().split('(')[1].split(')')[0]
				if '-mer' in ligand:
					continue
				elif '/' in ligand:
					ligand = ligand.split('/')[0]
				if len(ligand) != 3:
					continue
				pdbid_to_ligand[line[:4]] = ligand
	print(f"pdbid_to_ligand,{len(pdbid_to_ligand)}")
	return pdbid_to_ligand

if __name__ == "__main__":
	
	pdbid_list = get_pdbid_list()
	pdbid_to_ligand = get_pdbid_to_ligand()
	#Creation of downloadable files containing ligand URLs and pdb file URLs
	with open('out1.2_pdbbind_wget_ligand.txt','w') as fw : 
		for pdbid in pdbid_list:
			if pdbid in pdbid_to_ligand:
				ligand = pdbid_to_ligand[pdbid]
				fw.write('https://files.rcsb.org/ligands/download/'+ligand+'_ideal.pdb\n')

	with open('out1.2_pdbbind_wget_complex.txt','w') as f1w:
		for pdbid in pdbid_list:
			f1w.write('https://files.rcsb.org/download/'+pdbid+'.pdb\n')
	
	#PDB code list file creation 
	with open('out1.2_pdbid_list.txt','w') as f1w:
		for pdbid in pdbid_list:
			f1w.write(pdbid+'\n')
