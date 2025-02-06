#!/bin/sh
# This is a run script for conducting multiple cycle experiments
alias python=python3
#model=l96/l05II/l05III/l05IIm/l05IIIm
model="z05"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="quadratic"
perturbations="mlef"
na=200 # Number of assimilation cycle
nmem=10 # ensemble size
nobs=10 # observation volume
linf=False # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
exp="test"
echo ${exp}
cdir=` pwd `
rm -rf work/${model}/${exp}
mkdir -p work/${model}/${exp}
cd work/${model}/${exp}
cp ${cdir}/logging_config.ini .
rm -rf *.npy
rm -rf *.log
rm -rf timer
touch timer
rseed=509
for op in ${operators}; do
  for pert in ${perturbations}; do
    if [ $lloc = True ]; then
      if [ $pert = letkf ]; then
        cp ${cdir}/analysis/config/config_${pert}_sample.py config.py
      else
        cp ${cdir}/analysis/config/config_${pert}kloc_sample.py config.py
      fi
    else
      cp ${cdir}/analysis/config/config_${pert}_sample.py config.py
    fi
    gsed -i -e "2i \ \"op\":\"${op}\"," config.py
    gsed -i -e "2i \ \"na\":${na}," config.py
    gsed -i -e "2i \ \"nobs\":${nobs}," config.py
    gsed -i -e "/nmem/s/40/${nmem}/" config.py
    if [ $linf = True ];then
      gsed -i -e '/linf/s/False/True/' config.py
      gsed -i -e "4i \ \"infl_parm\":1.05," config.py
    else
      gsed -i -e '/linf/s/True/False/' config.py
    fi
    if [ $lloc = True ];then
      gsed -i -e "6i \ \"lsig\":3.0," config.py
    fi
    if [ $ltlm = True ];then
      gsed -i -e '/ltlm/s/False/True/' config.py
    else
      gsed -i -e '/ltlm/s/True/False/' config.py
    fi
    sed -i -e '/ss/s/False/True/' config.py
    sed -i -e '/getkf/s/True/False/' config.py
    if [ ! -z $rseed ]; then
      gsed -i -e "2i \ \"rseed\":${rseed}," config.py
    fi
    cat config.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    start_time=$(gdate +"%s.%5N")
    python ${cdir}/main.py ${model} > ${op}_${pert}.log 2>&1
    wait
    end_time=$(gdate +"%s.%5N")
    echo "${pert}" >> timer
    echo "scale=1; ${end_time}-${start_time}" | bc >> timer
    mv ${model}_e_${op}_${pt}.txt e_${op}_${pert}.txt
    mv ${model}_stda_${op}_${pt}.txt stda_${op}_${pert}.txt
    mv ${model}_ef_${op}_${pt}.txt ef_${op}_${pert}.txt
    mv ${model}_stdf_${op}_${pt}.txt stdf_${op}_${pert}.txt
    mv ${model}_xdmean_${op}_${pt}.txt xdmean_${op}_${pert}.txt
    mv ${model}_xsmean_${op}_${pt}.txt xsmean_${op}_${pert}.txt
    mv ${model}_xdfmean_${op}_${pt}.txt xdfmean_${op}_${pert}.txt
    mv ${model}_xsfmean_${op}_${pt}.txt xsfmean_${op}_${pert}.txt
    mv ${model}_xf_${op}_${pt}.npy xf_${op}_${pert}.npy
    mv ${model}_xa_${op}_${pt}.npy xa_${op}_${pert}.npy
    mv ${model}_xsa_${op}_${pt}.npy xsa_${op}_${pert}.npy
    mkdir -p data/${pert}
    mv ${model}_*_${op}_${pt}_cycle*.npy data/${pert}
  done
  python ${cdir}/plot/plote.py ${op} ${model} ${na}
  python ${cdir}/plot/plotxd.py ${op} ${model} ${na}
  python ${cdir}/plot/plotxa.py ${op} ${model} ${na}
  #rm obs*.npy
done
#rm ${model}*.txt 
#rm ${model}*.npy 
