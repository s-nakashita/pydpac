#!/bin/sh
# This is a run script for Lorenz05 experiment
export OMP_NUM_THREADS=4
#alias python=python3.9
model="l05IIm"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear"
perturbations="mlef envar"
na=240 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=15 # observation volume
linf=False # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
model_error=False
functype=gc5
a=-0.2
exp="mlef+envar_noinfl_mem${nmem}obs${nobs}"
echo ${exp}
cdir=` pwd `
if [ $model_error = True ]; then
wdir=work/${model}_inperfect/${exp}
else
wdir=work/${model}/${exp}
fi
rm -rf $wdir
mkdir -p $wdir
cd $wdir
cp ${cdir}/logging_config.ini .
rm -rf *.npy
rm -rf *.log
rm -rf timer
touch timer
if [ $model_error = True ]; then
if [ $model = l05II ]; then
ln -fs ${cdir}/data/l05III/truth.npy .
elif [ $model = l05IIm ]; then
ln -fs ${cdir}/data/l05IIIm/truth.npy .
fi
else
ln -fs ${cdir}/data/${model}/truth.npy .
fi
rseed=509
roseed=517
for op in ${operators}; do
  for pert in ${perturbations}; do
    echo $pert
    cp ${cdir}/analysis/config/config_${pert}_sample.py config.py
    gsed -i -e "2i \ \"op\":\"${op}\"," config.py
    gsed -i -e "2i \ \"na\":${na}," config.py
    gsed -i -e "2i \ \"nobs\":${nobs}," config.py
    gsed -i -e "/nmem/s/40/${nmem}/" config.py
    gsed -i -e "2i \ \"rseed\":${rseed}," config.py
    gsed -i -e "2i \ \"roseed\":${roseed}," config.py
    if [ $linf = True ];then
    gsed -i -e '/linf/s/False/True/' config.py
    else
    gsed -i -e '/linf/s/True/False/' config.py
    fi
    if [ $ltlm = True ];then
    gsed -i -e '/ltlm/s/False/True/' config.py
    else
    gsed -i -e '/ltlm/s/True/False/' config.py
    fi
    sed -i -e '/ss/s/False/True/' config.py
    sed -i -e '/getkf/s/True/False/' config.py
    if [ ! -z $lsig ]; then
      gsed -i -e "6i \ \"lsig\":${lsig}," config.py
    fi
    #gsed -i -e "6i \ \"functype\":\"${functype}\"," config.py
    #gsed -i -e "6i \ \"a\":${a}," config.py
    gsed -i -e "6i \ \"model_error\":${model_error}," config.py
    cat config.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    start_time=$(date +"%s")
    python ${cdir}/l05.py ${model} > ${model}_${op}_${pert}.log 2>&1
    wait
    end_time=$(date +"%s")
    echo "${op} ${pert}" >> timer
    echo "scale=3; ${end_time}-${start_time}" | bc >> timer
    mv ${model}_e_${op}_${pt}.txt e_${op}_${pert}.txt
    mv ${model}_stda_${op}_${pt}.txt stda_${op}_${pert}.txt
    mv ${model}_xdmean_${op}_${pt}.txt xdmean_${op}_${pert}.txt
    mv ${model}_xsmean_${op}_${pt}.txt xsmean_${op}_${pert}.txt
    mv ${model}_ef_${op}_${pt}.txt ef_${op}_${pert}.txt
    mv ${model}_stdf_${op}_${pt}.txt stdf_${op}_${pert}.txt
    mv ${model}_xdfmean_${op}_${pt}.txt xdfmean_${op}_${pert}.txt
    mv ${model}_xsfmean_${op}_${pt}.txt xsfmean_${op}_${pert}.txt
    #python ${cdir}/plot/plotpf.py ${op} ${model} ${na} ${pert}
  done
  python ${cdir}/plot/plote.py ${op} ${model} ${na} 
  python ${cdir}/plot/plotxd.py ${op} ${model} ${na} 
  #python ${cdir}/plot/plotchi.py ${op} ${model} ${na}
  #python ${cdir}/plot/plotinnv.py ${op} ${model} ${na} > innv_${op}.log
  python ${cdir}/plot/plotxa.py ${op} ${model} ${na} ${model_error}
  #python ${cdir}/plot/plotdof.py ${op} ${model} ${na}
  python ${cdir}/plot/ploterrspectra.py ${op} ${model} ${na} ${model_error}
  if [ ${na} -gt 1000 ]; then python ${cdir}/plot/nmc.py ${op} ${model} ${na}; fi
  python ${cdir}/plot/plotjh+gh.py ${op} ${model} ${na}
  rm ${model}_jh_${op}_*_cycle*.txt ${model}_gh_${op}_*_cycle*.txt ${model}_alpha_${op}_*_cycle*.txt
  
  #rm obs*.npy
done
#rm ${model}*.txt 
#rm ${model}_*_cycle*.npy 
mkdir -p data
mv ${model}_*_cycle*.npy data/
