#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic cubic"
#perturbations="mlef grad etkf po srf letkf" # kf var var4d"
perturbations="etkf po srf letkf kf var mlef grad"
na=500 # Number of assimilation cycle
linf="T" # "T":Apply inflation "F":Not apply
lloc="T" # "T":Apply localization "F":Not apply
ltlm="F" # "T":Use tangent linear approximation "F":Not use
a_window=
exp="obstest"
echo ${exp}
rm -rf ${exp}
mkdir ${exp}
cd ${exp}
cp ../logging_config.ini .
for op in ${operators}; do
  #if [ ${linf} = "T" ]; then
  #  if [ ${op} = "linear" ]; then
  #    infl_parm=1.2
  #  elif [ ${op} = "quadratic" ]; then
  #    infl_parm=1.3
  #  elif [ ${op} = "cubic" ]; then
  #    infl_parm=1.6
  #  elif [ ${op} = "test" ]; then
  #    infl_parm=1.25
  #  fi
  #else
  #  infl_parm=-1.0
  #fi
  for pt in ${perturbations}; do
    echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window}
    #echo ${op} ${pt} ${na} ${infl_parm} ${lloc} ${ltlm} ${a_window}
    python ../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window} > l96_${op}_${pt}.log 2>&1
    #python ../l96.py ${op} ${pt} ${na} ${infl_parm} ${lloc} ${ltlm} ${a_window} > l96_${op}_${pt}.log 2>&1
    wait
    #python ../plotpf.py ${op} l96 ${na} ${pt} > plotpf_${op}_${pt}.log 2>&1
    #python ../plotlpf.py ${op} l96 ${na} ${pt} > plotlpf_${op}_${pt}.log 2>&1
  done
  python ../plote.py ${op} l96 ${na}
  python ../plotchi.py ${op} l96 ${na}
  python ../plotinnv.py ${op} l96 ${na} > innv_${op}.log
  #python ../plotxa.py ${op} l96 ${na}
  #python ../plotdof.py ${op} l96 ${na}
  
  rm obs*.npy
done
#mv l96_e_${operators}.png l96_e_${operators}_${exp}.png
rm l96*.txt 
rm l96*.npy 