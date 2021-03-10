#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="quadratic cubic"
perturbations="etkf-jh etkf-fh mlef"
#perturbations="etkf po srf letkf var"
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
    mv z08_dof_${op}_${pt}.txt z08_dof_${op}_${pert}.txt
    mv z08_pa_${op}_${pt}.npy z08_pa_${op}_${pert}.npy
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
  python ../plot/plotdof.py ${op} z08 ${na}
done
rm z08*.txt 
rm z08*.npy 