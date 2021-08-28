#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="quadratic"
#perturbations="mlef grad etkf po srf letkf kf var var4d"
perturbations="mlef"
#model=l96 or z08
model=z08
na=20 # Number of assimilation cycle
linf="T" # "T"->Apply inflation "F"->Not apply
lloc="T" # "T"->Apply localization "F"->Not apply
ltlm="T" # "T"->Use tangent linear approximation "F"->Not use
#a_window=5 # for 4dVar (burgers not prepared)
exp="test"
echo ${exp}
rm -rf ${exp}
mkdir ${exp}
cd ${exp}
cp ../logging_config.ini .
for pt in ${perturbations}; do
  for op in ${operators}; do
    echo ${op} ${pt}  ${na} ${linf} ${lloc} ${ltlm}
    python ../${model}.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} > ${model}_${op}_${pt}.log 2>&1
    wait
  done
done
python ../plot/plote.py ${operators} ${model} ${na}
rm ${model}*.txt 
rm ${model}*.npy 