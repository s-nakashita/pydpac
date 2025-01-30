#!/bin/sh
# This is a run script for Lorenz96 experiment
alias python=python3
model="l96"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear"
datype="enkf"
perturbations="eakf srf etkf po letkf"
na=100 # Number of assimilation cycle
nmem=20 # ensemble size
nobs=40 # observation volume
linf=True # True:Apply inflation False:Not apply
lloc=True # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
#L="-1.0 0.5 1.0 2.0"
iinf=-1
exp="test_eakf_loc"
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
      gsed -i -e "4i \ \"iinf\":${iinf}," config.py
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
    cat config.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    start_time=$(gdate +"%s.%5N")
    python ${cdir}/l96.py > l96_${op}_${pert}.log 2>&1
    wait
    end_time=$(gdate +"%s.%5N")
    echo "${pert}" >> timer
    echo "scale=1; ${end_time}-${start_time}" | bc >> timer
    mv l96_e_${op}_${pt}.txt e_${op}_${pert}.txt
    mv l96_stda_${op}_${pt}.txt stda_${op}_${pert}.txt
    mv l96_ef_${op}_${pt}.txt ef_${op}_${pert}.txt
    mv l96_stdf_${op}_${pt}.txt stdf_${op}_${pert}.txt
    mv l96_xdmean_${op}_${pt}.txt xdmean_${op}_${pert}.txt
    mv l96_xsmean_${op}_${pt}.txt xsmean_${op}_${pert}.txt
    mv l96_xdfmean_${op}_${pt}.txt xdfmean_${op}_${pert}.txt
    mv l96_xsfmean_${op}_${pt}.txt xsfmean_${op}_${pert}.txt
    mv l96_xf_${op}_${pt}.npy xf_${op}_${pert}.npy
    mv l96_xa_${op}_${pt}.npy xa_${op}_${pert}.npy
    if [ $iinf -le -2 ]; then
      mv l96_infl_${op}_${pt}.txt infl_${op}_${pert}.txt
    fi
    mv l96_pdr_${op}_${pt}.txt pdr_${op}_${pert}.txt
    mkdir -p data/${pert}
    mv l96_*_${op}_${pt}_cycle*.npy data/${pert}
  done
  python ${cdir}/plot/plote.py ${op} l96 ${na}
  python ${cdir}/plot/plotxd.py ${op} l96 ${na}
  #python ${cdir}/plot/plotchi.py ${op} l96 ${na}
  #python ${cdir}/plot/plotinnv.py ${op} l96 ${na} > innv_${op}.log
  #python ${cdir}/plot/plotxa.py ${op} l96 ${na}
  #python ${cdir}/plot/nmc.py ${op} l96 ${na}
  #python ${cdir}/plot/plotdof.py ${op} l96 ${na}
  #python ${cdir}/plot/plotinfl.py ${op} l96 ${na}
  
  #rm obs*.npy
done
#rm l96*.txt 
#rm l96*.npy 
