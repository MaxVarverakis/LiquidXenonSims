#!/bin/bash

# for file in *
# do
#     mv $file ${file//Xe/Ta}
# done

for i in {0..15}
do
    mv ../build/out${i}_nt_e+.csv Ta$i.csv
    mv ../build/out${i}_nt_Data.csv TaDep$i.csv
done