#!/usr/bin/env bash
# learning by a cross-varidation process
#
# SCRIPT <file_stem> <method> <learning_script> <test_script> *<parameters>
# SCRIPT iris LR train_lr predict_lr reg=0.1 eta=1.0

### parse command line ###

fstem=$1            # file stem
shift
method=$1           # method abbreviation
shift
lscript=${1}        # learning script
shift
tscript=${1}        # test script
shift

### constants ####

# learning constants

nosFold=5
rseed=1234

# directories

datadir=00DATA
dataext=data

moddir=00MODEL
modext=model

resdir=00RESULT
resext=result

### prepeare for learning ###

# set learning parameters

opt="method=${method}"
cmdopt=""

for i in $@; do
  opt="${opt}-${i}"

  if [ "x${cmdopt}" = "x" ]; then
    cmdopt="--${i}"
  else
    cmdopt="${cmdopt} --${i}"
  fi
done

# generate trial list

tlist=`perl -e '
  for($i = 0; $i < $ARGV[0]; $i++) {
    push(@T, $i)
  }
  print join(" ",@T);
' $nosFold`

### make directories ###

if [ ! -d ${moddir} ]; then
    mkdir -p ${moddir}
fi

if [ ! -d ${resdir} ]; then
    mkdir -p ${resdir}
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

  python ${lscript} -q --rseed=${rseed} ${cmdopt} -i $ldata -o $model
  python ${tscript} -q -m $model -i $tdata >> $result

  echo ${fstem}@$opt@${t}
done

echo "###" ${fstem}@$opt
