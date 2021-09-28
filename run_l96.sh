#!/bin/sh
# This is a run script for Lorenz96 experiment
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic cubic"
#perturbations="mlef 4dmlef" # etkf po srf letkf" # kf var"
datype="mlef"
perturbations="${datype}be ${datype}bm l${datype} ${datype}"
#perturbations="var"
na=100 # Number of assimilation cycle
linf="T" # "T":Apply inflation "F":Not apply
lloc="T" # "T":Apply localization "F":Not apply
ltlm="F" # "T":Use tangent linear approximation "F":Not use
a_window=3
#L="-1.0 0.5 1.0 2.0"
exp="mlef_loc"
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
    #pt=${pert}
    pt=${datype}
    if [ "${pt:0:2}" = "4d" ]; then
      a_window=5
    else
      a_window=1
    fi
    loctype=$(echo "${pert}" | sed s/"${pt}"//g)
    echo $loctype
    if [ "$loctype" = "be" ]; then
    #if [ $pert = "mlef" ]; then
      lloc="T"
      iloc=1
    elif [ "$loctype" = "bm" ]; then
      lloc="T"
      iloc=2
    elif [ "$loctype" = "l" ]; then
    #elif [ $pert = "etkf" ]; then
      lloc="T"
      iloc=0
    else
      lloc="F"
      iloc=
    fi
    #for lb in $L; do
    echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window} ${iloc}
    #echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${lb}
    start_time=$(gdate +"%s.%5N")
    python ../../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window} ${iloc} > l96_${op}_${pert}.log 2>&1
    #python ../../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${lb} > l96_${op}_${pt}_${lb}.log 2>&1
    wait
    end_time=$(gdate +"%s.%5N")
    echo ${pert} >> timer
    echo "scale=1; ${end_time}-${start_time}" | bc >> timer
    #mv l96_e_${op}_${pt}.txt e_${op}_${pt}_${lb}.txt
    mv l96_e_${op}_${pt}.txt l96_e_${op}_${pert}.txt
    mv l96_pa_${op}_${pt}.npy l96_pa_${op}_${pert}.npy
    #if [ "${pert:4:1}" = "b" ]; then
    #mv l96_rho_${op}_${pt}.npy l96_rho_${op}_${pert}.npy
    #fi
    #for icycle in $(seq 0 4); do
    #  mv l96_K_${op}_${pt}_cycle$icycle.npy l96_K_${op}_${pert}_cycle$icycle.npy
    #  mv l96_dxaorig_${op}_${pt}_cycle$icycle.npy l96_dxaorig_${op}_${pert}_cycle$icycle.npy
    #  mv l96_dxa_${op}_${pt}_cycle$icycle.npy l96_dxa_${op}_${pert}_cycle$icycle.npy
    #  mv l96_pa_${op}_${pt}_cycle$icycle.npy l96_pa_${op}_${pert}_cycle$icycle.npy
    #  mv l96_pf_${op}_${pt}_cycle$icycle.npy l96_pf_${op}_${pert}_cycle$icycle.npy
    #  mv l96_spf_${op}_${pt}_cycle$icycle.npy l96_spf_${op}_${pert}_cycle$icycle.npy
    #  if [ "${pert:4:1}" = "b" ]; then
    #  mv l96_lpf_${op}_${pt}_cycle$icycle.npy l96_lpf_${op}_${pert}_cycle$icycle.npy
    #  mv l96_lspf_${op}_${pt}_cycle$icycle.npy l96_lspf_${op}_${pert}_cycle$icycle.npy
    #  fi
    #done
    #python ../../plot/plotk.py ${op} l96 ${na} ${pert}
    #python ../../plot/plotdxa.py ${op} l96 ${na} ${pert}
    #python ../../plot/plotpf.py ${op} l96 ${na} ${pert}
    #python ../../plot/plotlpf.py ${op} l96 ${na} ${pert} 
    #done
  done
  python ../../plot/plote.py ${op} l96 ${na} ${datype}
  #python ../../plot/plotchi.py ${op} l96 ${na}
  #python ../../plot/plotinnv.py ${op} l96 ${na} > innv_${op}.log
  #python ../../plot/plotxa.py ${op} l96 ${na}
  #python ../../plot/plotdof.py ${op} l96 ${na}
  
  #rm obs*.npy
done
rm l96*.txt 
rm l96*.npy 