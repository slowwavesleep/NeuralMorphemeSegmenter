#!/bin/bash

file_name=$1

while read line;
do
  cat "$line" | grep "No such file"
done < "$file_name"
