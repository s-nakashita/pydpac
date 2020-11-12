#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear"
perturbations="mlef etkf po srf letkf kf var var4d"
#perturbations="var4d"
na=100 # Number of assimilation cycle
linf="T"
lloc="T"
ltlm="T"
#for w in $(seq 1 20); do
a_window=5
exp="comp"
echo ${exp}
#./clean.sh  l96 ${operators}
for pt in ${perturbations}; do
  for op in ${operators}; do
    echo ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window}
    python ../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window} > l96_${op}_${pt}.log 2>&1
    wait
    #./output.sh ${exp} l96 ${op} ${pt}
  done
done
#./plot.sh l96 ${exp} ${operators} ${na}
#./copy.sh l96 ${exp} ${operators}
#done
python plote.py ${operators} l96 ${na}
#python plote-w.py ${operators} l96 ${na}
mv l96_e_${operators}.png l96_e_${operators}_${exp}.png
rm l96*.txt 
rm l96*.npy 