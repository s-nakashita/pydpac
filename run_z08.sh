#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="quadratic cubic"
perturbations="etkf-jh etkf-fh mlef grad"
#perturbations="etkf po srf letkf var"
na=20 # Number of assimilation cycle
linf="F" # "T"->Apply inflation "F"->Not apply
lloc="F" # "T"->Apply localization "F"->Not apply
ltlm="T" # "T"->Use tangent linear approximation "F"->Not use
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
    echo ${op} ${pt} ${linf} ${lloc} ${ltlm}
    python ../z08.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} > z08_${op}_${pert}.log 2>&1
    wait
    mv z08_e_${op}_${pt}.txt z08_e_${op}_${pert}.txt
    mv z08_dof_${op}_${pt}.txt z08_dof_${op}_${pert}.txt
  done
done
#pt="kf"
#for op in ${operators}; do
#  echo ${op} ${pt} ${linf} ${lloc} ${ltlm}
#  python ../z08.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} > z08_${op}_${pt}.log 2>&1
#  wait
#done
for op in ${operators}; do
  python ../plote.py ${op} z08 ${na}
  python ../plotdof.py ${op} z08 ${na}
done
#mv l96_e_${operators}.png l96_e_${operators}_${exp}.png
rm z08*.txt 
rm z08*.npy 