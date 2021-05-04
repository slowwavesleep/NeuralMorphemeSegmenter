#!/bin/bash

file_name=$1
num_runs=$2
visible_devices=$3

while read line;
do
  for (( c=1; c<=$num_runs; c++ ))
  do
    echo "Experiment run $c"
    echo $visible_devices
    CUDA_VISIBLE_DEVICES=$visible_devices python run_experiment.py -p $line
  done
done < $file_name
