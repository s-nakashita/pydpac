#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear"
#perturbations="mlef grad etkf po srf letkf kf var var4d"
perturbations="mlef etkf kf var"
#model=l96 or z08
model=l96
na=100 # Number of assimilation cycle
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
    mv const.log ../res/${model}_parm.${exp}
    wait
  done
done
python ../plote.py ${operators} ${model} ${na}
mv ${model}_e_${operators}.png ../res/${model}_e_${operators}_${exp}.png
rm ${model}*.txt 
rm ${model}*.npy 