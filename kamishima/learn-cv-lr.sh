#!/usr/bin/env bash
# learning by a cross-varidation process
# SCRIPT <file_stem> <method> <learning_script> <test_script> *<parameters>
# SCRIPT iris LR train_lr predict_lr reg=0.1 eta=1.0

#Bash command to run script
#bash "learn-cv-lr.sh" "adultd" "PR4" "train_pr.py" "predict_lr.py" reg=1 eta=30.0 ltype=4 itype=3 try=1



### parse command line ###

fstem=$1            # file stem (adultd)
shift
method=$1           # method abbreviation (PR4)
shift
lscript=${1}        # learning script (train_pr.py)
shift
tscript=${1}        # test script (predict_lr.py)
shift

### constants ####

# learning constants

nosFold=5
rseed=1234

# directories

datadir=00DATA
dataext=bindata

moddir=00MODEL
modext=model

resdir=00RESULT
resext=result

### prepeare for learning ###

# set learning parameters

opt="method=${method}"
cmdopt=""

#This adds (reg=1 eta=30.0 ltype=4 itype=3 try=1) to the opt and cmdopt variables
for i in $@; do
  opt="${opt}-${i}"

  if [ "x${cmdopt}" = "x" ]; then
    cmdopt="--${i}"
  else
    cmdopt="${cmdopt} --${i}"
  fi
done


# generate trial list
#Just a list of (0 1 2 3 4) presumably for the number of trials
tlist=`perl -e '
  for($i = 0; $i < $ARGV[0]; $i++) {
    push(@T, $i)
  }
  print join(" ",@T);
' $nosFold`
### make directories if they don't alreay exist ###

if [ ! -d ${moddir} ]; then
    mkdir -p ${moddir}
fi

if [ ! -d ${resdir} ]; then
    mkdir -p ${resdir}
fi

#I added this to make data directory
if [ ! -d ${datadir} ]; then
    mkdir -p ${datadir}
fi

### set output stems ###

datastem=${datadir}/${fstem}
modstem=${moddir}/${fstem}@${opt}
resstem=${resdir}/${fstem}@${opt}

### init results ###

result=${resstem}@t.${resext}

if [ -f ${result} ]; then
  rm -f ${result}
fi

### main loop ###

for t in ${tlist}; do
  ldata=${datastem}@${t}l.${dataext}
  tdata=${datastem}@${t}t.${dataext}
  model=${modstem}@${t}l.${modext}
  printf "\n\n\n\n\n"
  python ${lscript} -q --rseed=${rseed} ${cmdopt} -i $ldata -o $model
  python ${tscript} -q -m $model -i $tdata >> $result
  echo learning script: $lscript
  echo "test script: " + $tscript
  echo result file: $result
  echo model: $model
  echo ${fstem}@$opt@${t}
done

echo "###" ${fstem}@$opt
