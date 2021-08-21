#!/bin/sh
# This is a run script for Lorenz96 experiment
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic cubic"
#perturbations="mlef etkf po srf letkf" # kf var"
perturbations="mlefb mlefr mlef"
#perturbations="var"
na=100 # Number of assimilation cycle
linf="T" # "T":Apply inflation "F":Not apply
lloc="T" # "T":Apply localization "F":Not apply
ltlm="F" # "T":Use tangent linear approximation "F":Not use
a_window=5
#L="-1.0 0.5 1.0 2.0"
exp="mlef_loc"
echo ${exp}
#rm -rf ${exp}
#mkdir -p ${exp}
cd ${exp}
cp ../logging_config.ini .
rm *.npy
rm *.log
rm -rf timer
touch timer
for op in ${operators}; do
  for pert in ${perturbations}; do
    echo $pert
    #pt=${pert}
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
    end_time=$(gdate +"%s.%5N")
    echo ${pert} "time (sec)" >> timer
    echo "scale=1; ${end_time}-${start_time}" | bc >> timer
    #mv l96_e_${op}_${pt}.txt e_${op}_${pt}_${lb}.txt
    mv l96_e_${op}_${pt}.txt l96_e_${op}_${pert}.txt
    mv l96_pa_${op}_${pt}.npy l96_pa_${op}_${pert}.npy
    #if [ "${pert:4:1}" = "b" ]; then
    #mv l96_rho_${op}_${pt}.npy l96_rho_${op}_${pert}.npy
    #fi
    #for icycle in $(seq 0 4); do
    #  mv l96_pa_${op}_${pt}_cycle$icycle.npy l96_pa_${op}_${pert}_cycle$icycle.npy
    #  mv l96_pf_${op}_${pt}_cycle$icycle.npy l96_pf_${op}_${pert}_cycle$icycle.npy
    #  mv l96_spf_${op}_${pt}_cycle$icycle.npy l96_spf_${op}_${pert}_cycle$icycle.npy
    #  if [ "${pert:4:1}" = "b" ]; then
    #  mv l96_lpf_${op}_${pt}_cycle$icycle.npy l96_lpf_${op}_${pert}_cycle$icycle.npy
    #  mv l96_lspf_${op}_${pt}_cycle$icycle.npy l96_lspf_${op}_${pert}_cycle$icycle.npy
    #  fi
    #done
    #python ../plot/plotpf.py ${op} l96 ${na} ${pert}
    #python ../plot/plotlpf.py ${op} l96 ${na} ${pert} 
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