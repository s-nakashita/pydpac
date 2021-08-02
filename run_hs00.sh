#!/bin/sh
# This is a run script for Lorenz96 experiment
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic cubic"
perturbations="etkf"
#perturbations="rep"
na=1000 # Number of assimilation cycle
aostype="MB" # Adaptive Observation Strategy (AOS) type
vt=8 # verification time range (x6hour)
#linf="T" # "T":Apply inflation "F":Not apply
#lloc="F" # "T":Apply localization "F":Not apply
#ltlm="F" # "T":Use tangent linear approximation "F":Not use
#a_window=
exp="hsmb"
echo ${exp}
#rm -rf ${exp}
#mkdir ${exp}
cd ${exp}
cp ../logging_config.ini .
for op in ${operators}; do
  for pt in ${perturbations}; do
    echo ${op} ${pt} ${na} ${aostype} ${vt}
    python ../hs00.py ${op} ${pt} ${na} ${aostype} ${vt} > hs00_${op}_${pt}.log 2>&1
    wait
    #python ../plot/plotpf.py ${op} l96 ${na} ${pt} > plotpf_${op}_${pt}.log 2>&1
    #python ../plot/plotlpf.py ${op} l96 ${na} ${pt} > plotlpf_${op}_${pt}.log 2>&1
  done
  python ../plot/plothov.py ${op} hs00 ${na} ${aostype}
  #python ../plot/plotchi.py ${op} l96 ${na}
  #python ../plot/plotinnv.py ${op} l96 ${na} > innv_${op}.log
  #python ../plot/plotxa.py ${op} l96 ${na}
  #python ../plot/plotdof.py ${op} l96 ${na}
  
  #rm obs*.npy
done