#!/bin/bash
for file in build/out{0..15}_nt_Data.csv
do
    mv $file ${file//_nt_Data.csv/.csv}
done

for i in {0..15}
do
    mv data/XeDep$i.csv data/OldXeDep$i.csv
    mv build/out$i.csv data/XeDep$i.csv
done