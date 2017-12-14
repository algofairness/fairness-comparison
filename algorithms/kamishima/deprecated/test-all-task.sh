#!/usr/bin/env bash

#mymail=your@mail.address

### set log file ###

log=LOG-${0##*/}
host=`uname -n`

date > $log
##############################

go() {
  bash "$script" "$stem" "$method" "$lscript" "$tscript" $1 >> $log 2>&1
}

##############################

# include task-list

stem=adultd
nfv=4:7:4:16:4:7:14:6:5:2:2:3:8
. all-task.sh

##############################

date >> $log
cat $log
#cat $log | Mail -s ${0##*/}@${host} ${mymail}
