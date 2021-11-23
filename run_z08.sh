#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear quadratic cubic quartic quadratic-nodiff cubic-nodiff quartic-nodiff"
#perturbations="etkf-jh etkf-fh mlef-fh mlef-jh"
perturbations="mlef-fh mlef-jh mlefw-fh mlefw-jh"
#perturbations="etkf po srf letkf mlef var"
na=20 # Number of assimilation cycle
linf="F" # "T"->Apply inflation "F"->Not apply
lloc="F" # "T"->Apply localization "F"->Not apply
ltlm="F" # "T"->Use tangent linear approximation "F"->Not use
#a_window=5
exp="znoIL_w"
echo ${exp}
rm -rf work/${exp}
mkdir -p work/${exp}
cd work/${exp}
cp ../../logging_config.ini .
#Ntest=50 # Number of test
#for itest in $(seq 1 $Ntest); do
#echo "test:"${itest}
for pert in ${perturbations}; do
#  pt=${pert}
  pt=${pert%-*}
  if test "${pert#*-}" = "jh" ; then
    ltlm="T"
  elif test "${pert#*-}" = "fh" ; then
    ltlm="F"
  fi
  for op in ${operators}; do
    echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm}
    python ../../z08.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} > z08_${op}_${pert}.log 2>&1
    wait
    mv z08_e_${op}_${pt}.txt z08_e_${op}_${pert}.txt
    #mv z08_dof_${op}_${pt}.txt z08_dof_${op}_${pert}.txt
    mv z08_pa_${op}_${pt}.npy z08_pa_${op}_${pert}.npy
    #mv z08_e_${op}_${pt}.txt e_${op}_${pert}_${itest}.txt
    #mv z08_pa_${op}_${pt}.npy pa_${op}_${pert}_${itest}.npy
    #mv z08_ua_${op}_${pt}.npy ua_${op}_${pert}_${itest}.npy
    #mv z08_uf_${op}_${pt}.npy uf_${op}_${pert}_${itest}.npy
    #for i in $(seq 0 3); do
    #  mv z08_dh_${op}_${pt}_cycle${i}.npy z08_dh_${op}_${pert}_cycle${i}.npy
    #  mv z08_dxf_${op}_${pt}_cycle${i}.npy z08_dxf_${op}_${pert}_cycle${i}.npy
    #done
  done
done
#pt="kf"
#for op in ${operators}; do
#  echo ${op} ${pt} ${linf} ${lloc} ${ltlm}
#  python ../z08.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} > z08_${op}_${pt}.log 2>&1
#  wait
#done
for op in ${operators}; do
  python ../../plot/plote.py ${op} z08 ${na}
#  #python ../plot/plotdh.py ${op} z08 ${na}
#  #python ../plot/plotua.py ${op} z08 ${na}
#  #for pt in ${perturbations}; do
#  #  convert -delay 40 -loop 0 z08_ua_${op}_${pt}_cycle*.png z08_ua_${op}_${pt}.gif
#  #done
done
rm z08*.txt 
rm z08*.npy 
rm obs*.npy
#done