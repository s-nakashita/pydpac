#!/bin/sh
# This is a run script for parameter sensitivity experiment
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic cubic"
perturbations="mlef etkf po srf letkf kf var" # var4d"
na=500 # Number of assimilation cycle
linf="T" # "T":Apply inflation "F":Not apply
lloc="T" # "T":Apply localization "F":Not apply
ltlm="F" # "T":Use tangent linear approximation "F":Not use
a_window=5
exp="l96_nt"
echo ${exp}
rm -rf ${exp}
mkdir ${exp}
cd ${exp}
#sig="1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0"
#inf="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9"
#Nobs="40 35 30 25 20 15 10"
#Nmem="40 35 30 25 20 15 10 5"
Nt="1 2 3 4 5 6 7 8"
cp ../logging_config.ini .
for op in ${operators}; do
  for pt in ${perturbations}; do
    #for lsig in ${sig}; do
    #for infl_parm in ${inf}; do
    #for nobs in ${Nobs}; do
    #for nmem in ${Nmem}; do
    for nt in ${Nt}; do
      #echo ${op} ${pt} ${na} ${linf} ${lsig} ${ltlm} ${a_window}
      #echo ${op} ${pt} ${na} ${infl_parm} ${lloc} ${ltlm} ${a_window}
      #echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${nobs}
      #echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${nmem}
      echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${nt}
      for count in $(seq 1 50); do
        #python ../l96.py ${op} ${pt} ${na} ${linf} ${lsig} ${ltlm} ${a_window} > l96_${op}_${pt}.log 2>&1
        #python ../l96.py ${op} ${pt} ${na} ${infl_parm} ${lloc} ${ltlm} ${a_window} > l96_${op}_${pt}.log 2>&1
        #python ../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${nobs} > l96_${op}_${pt}.log 2>&1
        #python ../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${nmem} > l96_${op}_${pt}.log 2>&1
        python ../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${nt} > l96_${op}_${pt}.log 2>&1
        wait
        echo ${count}
        cp l96_e_${op}_${pt}.txt e_${op}_${pt}_${count}.txt
        rm obs*.npy
      done
      python ../plot/calc_mean.py ${op} l96 ${na} ${count} e ${pt}
      rm e_${op}_${pt}_*.txt
      #cp l96_e_${op}_${pt}.txt l96_e_${op}_${pt}_${lsig}.txt
      #cp l96_e_${op}_${pt}.txt l96_e_${op}_${pt}_${infl_parm}.txt
      #cp l96_e_${op}_${pt}.txt l96_e_${op}_${pt}_${nobs}.txt
      #cp l96_e_${op}_${pt}.txt l96_e_${op}_${pt}_${nmem}.txt
      cp l96_e_${op}_${pt}.txt l96_e_${op}_${pt}_${nt}.txt
    done
  done
  #python ../plot/ploteparam.py ${op} l96 ${na} loc
  #python ../plot/ploteparam.py ${op} l96 ${na} infl
  #python ../plot/ploteparam.py ${op} l96 ${na} nobs
  #python ../plot/ploteparam.py ${op} l96 ${na} nmem
  python ../plot/ploteparam.py ${op} l96 ${na} nt
  rm obs*.npy
done
rm l96*.txt 
rm l96*.npy 