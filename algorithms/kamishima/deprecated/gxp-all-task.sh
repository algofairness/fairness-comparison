#!/usr/bin/env bash

##############################

jno=0
job=JOB`date +'%Y%m%d%H%M'`
go() {
  jno=`expr $jno + 1`
  perl -e 'printf("%s-%03d ", $ARGV[0], $ARGV[1]);' $job $jno
  echo bash \"$script\" \"$stem\" \"$method\" \"$lscript\" \"$tscript\" $1
}

##############################

# include task-list

stem=adultd
nfv=4:7:4:16:4:7:14:6:5:2:2:3:8
. all-task.sh

##############################
