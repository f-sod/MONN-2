#!/usr/bin/python3

__auteurs__ = ""
__update_by__="ODJE Floriane"
__data__ = "2022-03" 

import pickle
	
def get_pdbid_to_uniprot():
	"""
	Parse the out2_pdbind_all_datafile.tsv to retrieve the UniProt id 
	And its sequences in a dictionary where pdb id are the keys 
	The values are the uniprotid and the sequences. 
	
	Returns
	-------
		Dict : pdbid_to_uniprot : 16051 elements 
	"""
	pdbid_to_uniprot = {}
	with open('out2_pdbbind_all_datafile.tsv') as f:
		for line in f.readlines():
			try:
				pdbid, uniprotid, ligand, inchi, seq, measure, value = line.strip().split('\t')
			except:
				print(line.strip().split('\t'))
				assert 0
			pdbid_to_uniprot[pdbid] = (uniprotid, seq)
	print(f"pdbid_to_uniprot,{len(pdbid_to_uniprot)}")
	return pdbid_to_uniprot



if __name__ == "__main__":
	"""
	The main code is decomposed into 2 parts. 
	The first one focuses on the interaction the second on the pocket but proceed the same way. 
	The first one aims to write all the protein fasta sequences of the pdb chain within a file and 
	on the second file their corresponding fasta  sequences with UniProt id. 
	The second part is similar but with the specification of the protein and pocket terms.
	Those file are called query and target because they will be later aligned. 
	"""
	pdbid_to_uniprot = get_pdbid_to_uniprot()
	
 
	with open('out4_interaction_dict','rb') as f:
		interaction_dict = pickle.load(f)
	with open('out6.1_query_pdb.fasta', 'w') as fw1, open('out6.1_target_uniprot_pdb.fasta', 'w') as fw2:
		for name in interaction_dict:
			pdbid = name.split('_')[0]
			if pdbid not in pdbid_to_uniprot:
				continue
			chain_dict = interaction_dict[name]['sequence']
			uniprotid, uniprotseq = pdbid_to_uniprot[pdbid]
			#print('chain_dict', len(chain_dict))
			for chain_id in chain_dict:
				if len(chain_dict[chain_id][0]) == 0:
					continue
				fw1.write('>'+pdbid+'_'+chain_id+'\n')
				fw1.write(chain_dict[chain_id][0]+'\n')
				fw2.write('>'+pdbid+'_'+chain_id+'_'+uniprotid+'\n')
				fw2.write(uniprotseq+'\n')
	print('1/2 Done writing files out6.1_query_pdb.fasta and out6.1_target_uniprot_pdb.fasta ')
	

	with open('out5_pocket_dict','rb') as f: 
		interaction_dict = pickle.load(f)
	with open('out6.2_query_pdbbind.fasta', 'w') as fw1,open('out6.2_target_uniprot_pdbbind.fasta', 'w') as fw2:
		for pdbid in interaction_dict:
			if pdbid not in pdbid_to_uniprot:
				#print(pdbid)
				continue
			try:
				chain_dict = interaction_dict[pdbid]
			except:
				continue
			uniprotid, uniprotseq = pdbid_to_uniprot[pdbid]
			for chain_id in chain_dict:
				if len(list(chain_dict[chain_id].keys())) == 0: 
					continue
				fw1.write('>'+pdbid+'_'+chain_id+'\n')
				fw1.write(str(list(chain_dict[chain_id].keys()))+'\n')
				fw2.write('>'+pdbid+'_'+chain_id+'_'+uniprotid+'\n')
				fw2.write(uniprotseq+'\n')
	print('2/2 Done writing files out6.2_query_pdbbind.fasta and out6.2_target_uniprot_pdbbind.fasta')
