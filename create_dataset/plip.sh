#!/bin/bash
for file in `ls *.pdb`
do 
    id=`basename $file`
    b=${id%.pdb}
    echo Processing id $b
    #python /home/floriane/plip/plip/plipcmd.py -f ${b}.pdb -t --name ${b}_output
    plip -f ${b}.pdb -t --name ${b}_output
    echo $b
done