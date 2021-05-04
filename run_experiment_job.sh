#!/bin/bash

file_name=$1
num_runs=$2

while read line;
do
  for (( c=1; c<=$num_runs; c++ ))
  do
    echo "Experiment run $c"
    python run_experiment.py -p $line
  done
done < $file_name
