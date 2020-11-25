#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear"
#perturbations="mlef grad etkf po srf letkf kf var var4d"
perturbations="etkf"
na=100 # Number of assimilation cycle
linf="T" # "T"->Apply inflation "F"->Not apply
lloc="T" # "T"->Apply localization "F"->Not apply
ltlm="T" # "T"->Use tangent linear approximation "F"->Not use
a_window=5
exp="cltest"
echo ${exp}
rm -rf ${exp}
mkdir ${exp}
cd ${exp}
cp ../logging_config.ini .
for pt in ${perturbations}; do
  for op in ${operators}; do
    echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window}
    python ../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window} > l96_${op}_${pt}.log 2>&1
    wait
  done
done
python ../plote.py ${operators} l96 ${na}
#mv l96_e_${operators}.png l96_e_${operators}_${exp}.png
rm l96*.txt 
rm l96*.npy 