#!/bin/bash

# for file in *
# do
#     mv $file ${file//Xe/Ta}
# done

for i in {0..15}
do
    mv ../build/out${i}_nt_e+.csv Xe$i.csv
    mv ../build/out${i}_nt_Data.csv XeDep$i.csv
done