#!/bin/sh
# This is a run script for Lorenz96 experiment
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic cubic"
#perturbations="mlef etkf po srf letkf kf var"
perturbations="mlef"
na=100 # Number of assimilation cycle
linf="T" # "T":Apply inflation "F":Not apply
lloc="T" # "T":Apply localization "F":Not apply
ltlm="F" # "T":Use tangent linear approximation "F":Not use
v_time=4
lplot="F"
exp="testensvi"
echo ${exp}
rm -rf ${exp}
mkdir ${exp}
cd ${exp}
cp ../logging_config.ini .
for op in ${operators}; do
  for pt in ${perturbations}; do
  for v_time in 1 2 4 8; do
  hh=$((6 * $v_time))
  for count in $(seq 1 50); do
    if [ $count = 1 ]; then
      lplot=T 
    else
      lplot=F
    fi
    echo ${count} ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${v_time} ${lplot}
    #python ../sva.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${v_time} ${lplot} > l96_${op}_${pt}.log 2>&1
    python ../ensvsa.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${v_time} ${lplot} > l96_${op}_${pt}.log 2>&1
    wait
    mv erandom.txt er_${op}_${pt}_${count}.txt 
    mv esv${hh}h.txt es_${op}_${pt}_${count}.txt
    mv initialSVA_${pt}_${hh}h.npy isv_${pt}_${hh}h_${count}.npy
    mv finalSVA_${pt}_${hh}h.npy fsv_${pt}_${hh}h_${count}.npy
  done
  python ../plot/calc_mean.py ${op} sv${hh}h 10 ${count} es ${pt}
  python ../plot/calc_mean.py ${op} random 10 ${count} er ${pt}
  #python ../plot/calc_mean.py ${hh} l96 10 ${count} sv ${pt}
  python ../plot/plotsv_sp.py ${pt} ${v_time} ${count}
  mv sv${hh}h_es_${op}_${pt}.txt sv${hh}_${pt}.txt
  mv random_er_${op}_${pt}.txt random_${pt}.txt 
  rm er*.txt 
  rm es*.txt
  rm isv*.npy
  rm fsv*.npy
  done
  python ../plot/plotsv.py ${pt}
  done
done