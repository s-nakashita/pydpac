#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
perturbations="etkf-jh etkf-fh mlef var"
#perturbations="etkf po srf letkf mlef var"
na=20 # Number of assimilation cycle
linf="F" # "T"->Apply inflation "F"->Not apply
lloc="F" # "T"->Apply localization "F"->Not apply
ltlm="F" # "T"->Use tangent linear approximation "F"->Not use
#a_window=5
exp="znoIL"
echo ${exp}
rm -rf ${exp}
mkdir ${exp}
cd ${exp}
cp ../logging_config.ini .
for pert in ${perturbations}; do
#  pt=${pert}
  pt=${pert:0:4}
  if test "${pert:5:2}" = "jh" ; then
    ltlm="T"
  elif test "${pert:5:2}" = "fh" ; then
    ltlm="F"
  fi
  for op in ${operators}; do
    echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm}
    python ../z08.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} > z08_${op}_${pert}.log 2>&1
    wait
    mv z08_e_${op}_${pt}.txt z08_e_${op}_${pert}.txt
    #mv z08_dof_${op}_${pt}.txt z08_dof_${op}_${pert}.txt
    mv z08_pa_${op}_${pt}.npy z08_pa_${op}_${pert}.npy
    mv z08_ua_${op}_${pt}.npy z08_ua_${op}_${pert}.npy
    mv z08_uf_${op}_${pt}.npy z08_uf_${op}_${pert}.npy
    for i in $(seq 0 3); do
      mv z08_dh_${op}_${pt}_cycle${i}.npy z08_dh_${op}_${pert}_cycle${i}.npy
      mv z08_dxf_${op}_${pt}_cycle${i}.npy z08_dxf_${op}_${pert}_cycle${i}.npy
    done
  done
done
#pt="kf"
#for op in ${operators}; do
#  echo ${op} ${pt} ${linf} ${lloc} ${ltlm}
#  python ../z08.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} > z08_${op}_${pt}.log 2>&1
#  wait
#done
for op in ${operators}; do
  python ../plot/plote.py ${op} z08 ${na}
  #python ../plot/plotdh.py ${op} z08 ${na}
  #python ../plot/plotua.py ${op} z08 ${na}
  #for pt in ${perturbations}; do
  #  convert -delay 40 -loop 0 z08_ua_${op}_${pt}_cycle*.png z08_ua_${op}_${pt}.gif
  #done
done
rm z08*.txt 
rm z08*.npy 