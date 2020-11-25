#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="quadratic"
perturbations="mlef grad kf var"
#perturbations="etkf po srf letkf"
na=20 # Number of assimilation cycle
linf="T" # "T"->Apply inflation "F"->Not apply
lloc="T" # "T"->Apply localization "F"->Not apply
ltlm="T" # "T"->Use tangent linear approximation "F"->Not use
#a_window=5
exp="ztest"
echo ${exp}
rm -rf ${exp}
mkdir ${exp}
cd ${exp}
cp ../logging_config.ini .
for pt in ${perturbations}; do
  for op in ${operators}; do
    echo ${op} ${pt} ${linf} ${lloc} ${ltlm}
    python ../z08.py ${op} ${pt} ${linf} ${lloc} ${ltlm} > z08_${op}_${pt}.log 2>&1
    wait
  done
done
python ../plote.py ${operators} z08 ${na}
#mv l96_e_${operators}.png l96_e_${operators}_${exp}.png
rm z08*.txt 
rm z08*.npy 