#!/bin/bash

for file in build/out{0..15}_nt_e+.csv
do
    mv $file ${file//_nt_e+.csv/.csv}
done

for i in {0..15}
do
    mv build/out$i.csv data/Xe_1mm_$i.csv
done