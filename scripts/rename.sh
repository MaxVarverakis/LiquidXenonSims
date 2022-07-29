#!/bin/bash
for file in build/out{0..15}Dep.csv
do
    mv $file ${file//Dep.csv/.csv}
done

for i in {0..15}
do
    mv build/out$i.csv data/TaDep$i.csv
done