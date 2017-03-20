#!/usr/bin/env bash

### constants ###

# input result dir and ext
resdir=00RESULT
resext=result

# output summary dir and ext
sumdir=00SUMMARY
sumext=txt

# statistics script
stat_script="python fai_bin_bin.py --raw"

### functions ###

# list of data names in $resdir
get_data_list() {
  ls -1 $resdir | cut -d@ -f 1 | sort | uniq
}

# list of trial names in $resdir
get_trial_list() {
  ls -1 $resdir | perl -pe "s/^.*@([^@]*)\.${resext}$/\1/" | sort | uniq
}

# list of method names of the $data and $trial in $resdir
get_method_list() {
  ls -1 $resdir | perl -pe "s/^.*\/([^\/]*)\.${resext}$/\1/" | \
    cut -d@ -f2 | sort | uniq
}

### setup ###

# check directories

if [ ! -d $resdir ]; then
  echo "No input directory: $resdir"
  exit 1
fi

if [ ! -d $sumdir ]; then
  mkdir -p $sumdir
fi

### main ###

for data in `get_data_list`; do
for trial in `get_trial_list`; do

outfile=${data}@${trial}.${sumext}
out=${sumdir}/${outfile}
if [ -f $out ]; then
  /bin/rm -f $out
fi

#---------------

echo "Now processing ${data}@${trial}"

for method in `get_method_list`; do
  infile=${resdir}/${data}@${method}@${trial}.${resext}
  #For example
  #infile = 00RESULT/adultd@method=PR4-reg=1-eta=5.0-ltype=4-itype=3-try=1@t.result
  # -f checks if file exists, and is a normal file
  # perl -pe executes perl command
  if [ -f $infile ]; then
    echo -n "$method " | perl -pe 's/\-nfv=[0-9\:]+//' | perl -pe 's/ /\t/g' >> $out
    echo $out
    $stat_script < $infile | perl -pe 's/ /\t/g' >> $out
  fi
done

#---------------

done
done
