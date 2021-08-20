#!/bin/sh
# This is a run script for Lorenz96 experiment
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic cubic"
#perturbations="mlef etkf po srf letkf kf var"
perturbations="mlefb mlefr mlef"
#perturbations="var"
na=100 # Number of assimilation cycle
linf="T" # "T":Apply inflation "F":Not apply
lloc="F" # "T":Apply localization "F":Not apply
ltlm="F" # "T":Use tangent linear approximation "F":Not use
a_window=5
#L="-1.0 0.5 1.0 2.0"
exp="mlef_loc"
echo ${exp}
rm -rf ${exp}
mkdir ${exp}
cd ${exp}
cp ../logging_config.ini .
touch timer
for op in ${operators}; do
  for pert in ${perturbations}; do
    pt=${pert:0:4}
    if [ "${pt:0:2}" = "4d" ]; then
      a_window=5
    else
      a_window=1
    fi
    if [ "${pert:4:1}" = "b" ]; then
      lloc="T"
      bloc="T"
      rloc="F"
    elif [ "${pert:4:1}" = "r" ]; then
      lloc="T"
      bloc="F"
      rloc="T"
    else
      lloc="F"
      bloc="F"
      rloc="F"
    fi
    #for lb in $L; do
    echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window} ${rloc} ${bloc}
    #echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${lb}
    start_time=$(gdate +"%s.%5N")
    python ../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window} ${rloc} ${bloc} > l96_${op}_${pert}.log 2>&1
    #python ../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${lb} > l96_${op}_${pt}_${lb}.log 2>&1
    wait
    #mv l96_e_${op}_${pt}.txt e_${op}_${pt}_${lb}.txt
    mv l96_e_${op}_${pt}.txt l96_e_${op}_${pert}.txt
    mv l96_pa_${op}_${pt}.npy l96_pa_${op}_${pert}.npy
    end_time=$(gdate +"%s.%5N")
    echo ${pert} "time (sec)" >> timer
    echo "scale=1; ${end_time}-${start_time}" | bc >> timer
    #python ../plot/plotpf.py ${op} l96 ${na} ${pt} > plotpf_${op}_${pt}.log 2>&1
    #python ../plot/plotlpf.py ${op} l96 ${na} ${pt} > plotlpf_${op}_${pt}.log 2>&1
    #done
  done
  python ../plot/plote.py ${op} l96 ${na}
  #python ../plot/plotchi.py ${op} l96 ${na}
  #python ../plot/plotinnv.py ${op} l96 ${na} > innv_${op}.log
  #python ../plot/plotxa.py ${op} l96 ${na}
  #python ../plot/plotdof.py ${op} l96 ${na}
  
  #rm obs*.npy
done
rm l96*.txt 
rm l96*.npy 