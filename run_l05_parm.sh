#!/bin/sh
# This is a run script for Lorenz05 experiment
model="l05III"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="var"
na=100 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=30 # observation volume
linf=True  # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
#L="-1.0 0.5 1.0 2.0"
ptype=lb
exp="var_${ptype}_obs${nobs}"
#exp="${datype}_loc_hint"
echo ${exp}
cdir=` pwd `
#rm -rf work/${model}/${exp}
mkdir -p work/${model}/${exp}
cd work/${model}/${exp}
cp ${cdir}/logging_config.ini .
ln -fs ${cdir}/data/l05III/truth.npy .
rm -rf obs*.npy
rm -rf *.log
rm -rf timer
touch timer
nmemlist="40 80 120 160 200"
lsiglist="20 40 60 80 100 120 140 160"
nobslist="480 240 120 60 30 15"
sigblist="0.2 0.4 0.6 0.8 1.0 1.2"
sigblist="1.6 2.0 2.4 2.8 3.2 3.6"
lblist="-1.0 10.0 20.0 40.0 60.0 80.0"
touch params.txt
for op in ${operators}; do
  echo $ptype > params.txt
  #for nmem in ${nmemlist}; do
  #  echo $nmem >> params.txt
  #  ptmp=$nmem
  #for lsig in ${lsiglist}; do
  #  echo $lsig >> params.txt
  #  ptmp=$lsig
  #for nobs in ${nobslist}; do
  #  echo $nobs >> params.txt
  #  ptmp=$nobs
  #for sigb in ${sigblist}; do
  #  echo $sigb >> params.txt
  #  ptmp=$sigb
  for lb in ${lblist}; do
    echo $lb >> params.txt
    ptmp=$lb
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
      if [ $ptype = loc ]; then
        gsed -i -e "6i \ \"lsig\":${lsig}," config.py
      fi
      if [ $ptype = sigb ]; then
        gsed -i -e "6i \ \"sigb\":${sigb}," config.py
      fi
      if [ $ptype = lb ]; then
        gsed -i -e "6i \ \"lb\":${lb}," config.py
      fi
      cat config.py
      ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
      pt=${ptline#\"*}; pt=${pt%\"*}
      echo $pt
      for count in $(seq 1 10); do
        echo $count
        start_time=$(date +"%s")
        python3.9 ${cdir}/l05.py ${model} > ${model}_${op}_${pert}.log 2>&1
        wait
        end_time=$(date +"%s")
        cputime=`echo "scale=3; ${end_time}-${start_time}" | bc`
        echo "${op} ${pert} ${count} ${cputime}" >> timer
        mv ${model}_e_${op}_${pt}.txt e_${op}_${pt}_${count}.txt
        mv ${model}_stda_${op}_${pt}.txt stda_${op}_${pt}_${count}.txt
        mv ${model}_xdmean_${op}_${pt}.txt xdmean_${op}_${pt}_${count}.txt
        mv ${model}_xsmean_${op}_${pt}.txt xsmean_${op}_${pt}_${count}.txt
        rm obs*.npy
      done
      python3.9 ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} e ${pt}
      rm e_${op}_${pt}_*.txt
      python3.9 ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} stda ${pt}
      rm stda_${op}_${pt}_*.txt
      python3.9 ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} xdmean ${pt}
      rm xdmean_${op}_${pt}_*.txt
      python3.9 ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} xsmean ${pt}
      rm xsmean_${op}_${pt}_*.txt
      mv ${model}_e_${op}_${pt}.txt ${model}_e_${op}_${pert}_${ptmp}.txt
      mv ${model}_stda_${op}_${pt}.txt ${model}_stda_${op}_${pert}_${ptmp}.txt
      mv ${model}_xdmean_${op}_${pt}.txt ${model}_xdmean_${op}_${pert}_${ptmp}.txt
      mv ${model}_xsmean_${op}_${pt}.txt ${model}_xsmean_${op}_${pert}_${ptmp}.txt
    done #pert
    #python3.9 ${cdir}/plot/plote.py ${op} ${model} ${na} #mlef
  done #nmem
  cat params.txt
  python3.9 ${cdir}/plot/ploteparam.py ${op} ${model} ${na} $ptype
  python3.9 ${cdir}/plot/plotxdparam.py ${op} ${model} ${na} $ptype
  #rm obs*.npy
done #op
#rm ${model}*.txt 
#rm ${model}*.npy 
