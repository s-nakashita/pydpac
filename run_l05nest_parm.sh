#!/bin/sh
# This is a run script for Nesting Lorenz experiment
model="l05nest"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="mlef"
na=30 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=60 # observation volume
linf=True  # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
#L="-1.0 0.5 1.0 2.0"
exp="test_nmem"
#exp="${datype}_loc_hint"
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
nmemlist="40 80 120 160 200"
lsiglist="5 10 15 20"
nobslist="240 120 60 30 15"
touch params.txt
for op in ${operators}; do
  echo nmem > params.txt
  #echo loc > params.txt
  #echo nobs > params.txt
  for nmem in ${nmemlist}; do
    echo $nmem >> params.txt
    ptmp=$nmem
  #for lsig in ${lsiglist}; do
  #  echo $lsig >> params.txt
  #  ptmp=$lsig
  #for nobs in ${nobslist}; do
  #  echo $nobs >> params.txt
  #  ptmp=$nobs
    for pert in ${perturbations}; do
      echo $pert
      cp ${cdir}/analysis/config/config_${pert}_sample.py config.py
      gsed -i -e "2i \ \"op\":\"${op}\"," config.py
      gsed -i -e "2i \ \"na\":${na}," config.py
      gsed -i -e "2i \ \"nobs\":${nobs}," config.py
      gsed -i -e "/nmem/s/40/${nmem}/" config.py
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
      if [ $ptmp = lsig ]; then
        gsed -i -e "6i \ \"lsig\":${lsig}," config.py
      fi
      cat config.py
      ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
      pt=${ptline#\"*}; pt=${pt%\"*}
      echo $pt
      mv config.py config_gm.py
      cp config_gm.py config_lam.py
      cat config_gm.py
      cat config_lam.py
      for count in $(seq 1 10); do
        start_time=$(gdate +"%s.%5N")
        python ${cdir}/${model}.py > ${model}_${op}_${pert}.log 2>&1
        wait
        end_time=$(gdate +"%s.%5N")
        cputime=`echo "scale=1; ${end_time}-${start_time}" | bc`
        echo "${op} ${pert} ${count} ${cputime}" >> timer
        cp ${model}_e_gm_${op}_${pt}.txt e_gm_${op}_${pt}_${count}.txt
        cp ${model}_stda_gm_${op}_${pt}.txt stda_gm_${op}_${pt}_${count}.txt
        cp ${model}_e_lam_${op}_${pt}.txt e_lam_${op}_${pt}_${count}.txt
        cp ${model}_stda_lam_${op}_${pt}.txt stda_lam_${op}_${pt}_${count}.txt
        rm obs*.npy
      done
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} e_gm ${pt}
      rm e_gm_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} stda_gm ${pt}
      rm stda_gm_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} e_lam ${pt}
      rm e_lam_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} stda_lam ${pt}
      rm stda_lam_${op}_${pt}_*.txt
      mv ${model}_e_gm_${op}_${pt}.txt ${model}_e_gm_${op}_${pert}_${ptmp}.txt
      mv ${model}_stda_gm_${op}_${pt}.txt ${model}_stda_gm_${op}_${pert}_${ptmp}.txt
      mv ${model}_e_lam_${op}_${pt}.txt ${model}_e_lam_${op}_${pert}_${ptmp}.txt
      mv ${model}_stda_lam_${op}_${pt}.txt ${model}_stda_lam_${op}_${pert}_${ptmp}.txt
    done #pert
    #python ${cdir}/plot/plote.py ${op} ${model} ${na} #mlef
  done #nmem
  cat params.txt
  python ${cdir}/plot/ploteparam_nest.py ${op} ${model} ${na} $ptmp
  #rm obs*.npy
done #op
#rm ${model}*.txt 
#rm ${model}*.npy 
