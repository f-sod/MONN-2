#!/bin/bash
empty_list=()
for file in `find -empty`;
do
    name=`basename $file`
    id=${name%_ideal.pdb*}
    empty_list+=( $id )
done
echo ${empty_list[@]}