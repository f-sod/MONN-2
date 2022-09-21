
Inference.py : 
  The code take as inputs molecules (inchi key) and proteins (fasta sequence without header).
  The code output a table with the affinity of interaction beetween each atoms and amino acids residues as well as an estimated pki or pKD.
 
To run this code you need the following codes in the same directory :
    - CPI_model.py
    - processing_input.py 

TRAIN.py :
  This file train - test and valid the model
  The different datasets are saved evrytimes this files is run 
  The parameters of the trained model are saved in MONN-2.pth   
--> MONN-2.pth :
    This file contains the weight and information of the model.
    For better result of inferences train and save new parameters wwithin this file .

Processing_input.py : 
  cf. documentation of the function within the codes 
