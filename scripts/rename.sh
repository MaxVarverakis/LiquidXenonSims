#!/bin/bash

for file in build/out{0..15}_nt_Data.csv
do
    mv $file ${file//_nt_Data/Dep}
done

for file in build/out{0..15}_nt_e+.csv
do
    mv $file ${file//_nt_e+/}
done

for i in {0..15}
do
    mv data/XeDep$i.csv data/OldXeDep$i.csv
    mv data/Xe$i.csv data/OldXe$i.csv
    
    mv build/out$i.csv data/Xe$i.csv
done

for file in build/out{0..15}Dep.csv
do
    mv $file ${file//Dep/}
done

for i in {0..15}
do
    mv build/out$i.csv data/XeDep$i.csv
done