#!/usr/bin/env bash
# generated discritized adult.data
# INPUT: adult.test in adult@UCI Repository
# OUTPUT: adultd.arff, adultd.data adult.bindata

#SET to location of directory
w=${HOME}/Desktop/CS_Thesis/2012ecmlpkdd/data-adultd

o=./
d=./

tmp=chotto-$$
input=adult.test

echo "convert to arff format"
python adult_arff.py $o/${input} $d/${tmp}.arff

echo "discritize by the Calders and Verwer's procedure"
python adult_discritize.py $d/${tmp}.arff $d/adultd.arff

echo "convert to the space separated format"
python $w/arff2txt.py -m 1 $d/adultd.arff $d/${tmp}.data
python $w/arff2txt.py -m 3 $d/adultd.arff $d/${tmp}.bindata

echo "move the senstitive attribute to the last position"
python select_sensitive.py -n -r -f 9 $d/${tmp}.data $d/adultd.data
python select_sensitive.py -n -r -f 67 $d/${tmp}.bindata $d/adultd.bindata

echo "clearn-up tmporary files"
/bin/rm -f $d/${tmp}*
