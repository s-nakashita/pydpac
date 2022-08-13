#!/bin/sh
# This is a run script for QG experiment (Sakov and Oke, 2008)
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic cubic"
#perturbations="mlef 4dmlef" # etkf po srf letkf" # kf var"
perturbations="letkf"
#perturbations="var"
na=5 # Number of assimilation cycle
linf="T" # "T":Apply inflation "F":Not apply
lloc="T" # "T":Apply localization "F":Not apply
ltlm="F" # "T":Use tangent linear approximation "F":Not use
a_window=1
#L="-1.0 0.5 1.0 2.0"
exp="qgtest"
#exp="${datype}_loc_hint"
echo ${exp}
rm -rf work/${exp}
mkdir -p work/${exp}
cd work/${exp}
cp ../../logging_config.ini .
rm -rf *.npy
rm -rf *.log
rm -rf timer
touch timer
for op in ${operators}; do
  for pert in ${perturbations}; do
    echo $pert
    pt=${pert}
    echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} #${a_window} ${iloc}
    #echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${lb}
    start_time=$(gdate +"%s.%5N")
    python ../../so08_qg.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} #> qg_${op}_${pert}.log 2>&1
    wait
    end_time=$(gdate +"%s.%5N")
    echo ${pert} >> timer
    echo "scale=1; ${end_time}-${start_time}" | bc >> timer
  done
  #python ../../plot/plote.py ${op} l96 ${na} ${datype}
  #python ../../plot/plotchi.py ${op} l96 ${na}
  #python ../../plot/plotinnv.py ${op} l96 ${na} > innv_${op}.log
  #python ../../plot/plotxa.py ${op} l96 ${na}
  #python ../../plot/plotdof.py ${op} l96 ${na}
  
  #rm obs*.npy
done
