#!/bin/sh
# This is a run script for Lorenz05 experiment
export OMP_NUM_THREADS=4
#alias python=python3.9
model="l05II"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear"
perturbations="envar"
na=240 # Number of assimilation cycle
nmem=240 # ensemble size
nobs=15 # observation volume
linf=True  # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
model_error=True
#L="-1.0 0.5 1.0 2.0"
ptype=infl
functype=gc5
a=-0.2
exp="envar_obs${nobs}mem${nmem}_${ptype}"
echo ${exp}
cdir=` pwd `
wdir=work/${model}_inperfect/${exp}
#rm -rf $wdir
mkdir -p $wdir
cd $wdir
cp ${cdir}/logging_config.ini .
if [ $model_error = True ]; then
if [ $model = l05II ]; then
ln -fs ${cdir}/data/l05III/truth.npy .
else
ln -fs ${cdir}/data/l05IIIm/truth.npy .
fi
else
ln -fs ${cdir}/data/${model}/truth.npy .
fi
rm -rf obs*.npy
rm -rf *.log
rm -rf timer
touch timer
nmemlist="40 80 120 160 200 240"
lsiglist="20 40 60 80 100 120 140 160"
nobslist="480 240 120 60 30 15"
infllist="1.0 1.05 1.1 1.15 1.2 1.25"
sigblist="0.2 0.4 0.6 0.8 1.0 1.2"
sigblist="1.6 2.0 2.4 2.8 3.2 3.6"
lblist="2.0 4.0 6.0 8.0 10.0 12.0"
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
  for infl in ${infllist}; do
    echo $infl >> params.txt
    ptmp=$infl
  #for sigb in ${sigblist}; do
  #  echo $sigb >> params.txt
  #  ptmp=$sigb
  #for lb in ${lblist}; do
  #  echo $lb >> params.txt
  #  ptmp=$lb
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
      if [ $ptype = infl ]; then
        gsed -i -e "6i \ \"infl_parm\":${infl}," config.py
      fi
      if [ $ptype = sigb ]; then
        gsed -i -e "6i \ \"sigb\":${sigb}," config.py
      fi
      if [ $ptype = lb ]; then
        gsed -i -e "6i \ \"lb\":${lb}," config.py
      fi
      if [ $pert = var ]; then
        gsed -i -e "6i \ \"functype\":\"${functype}\"," config.py
        gsed -i -e "6i \ \"a\":${a}," config.py
      fi
      gsed -i -e "6i \ \"model_error\":${model_error}," config.py
      cat config.py
      ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
      pt=${ptline#\"*}; pt=${pt%\"*}
      echo $pt
      for count in $(seq 1 10); do
        echo $count
        start_time=$(date +"%s")
        python ${cdir}/l05.py ${model} > ${model}_${op}_${pert}.log 2>&1
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
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} e ${pt}
      rm e_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} stda ${pt}
      rm stda_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} xdmean ${pt}
      rm xdmean_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} xsmean ${pt}
      rm xsmean_${op}_${pt}_*.txt
      mv ${model}_e_${op}_${pt}.txt ${model}_e_${op}_${pert}_${ptmp}.txt
      mv ${model}_stda_${op}_${pt}.txt ${model}_stda_${op}_${pert}_${ptmp}.txt
      mv ${model}_xdmean_${op}_${pt}.txt ${model}_xdmean_${op}_${pert}_${ptmp}.txt
      mv ${model}_xsmean_${op}_${pt}.txt ${model}_xsmean_${op}_${pert}_${ptmp}.txt
    done #pert
  done #nmem
  cat params.txt
  python ${cdir}/plot/ploteparam.py ${op} ${model} ${na} $ptype
  python ${cdir}/plot/plotxdparam.py ${op} ${model} ${na} $ptype
  rm ${model}*cycle*.npy ${model}*cycle*.txt
  #rm obs*.npy
done #op
#rm ${model}*.txt 
#rm ${model}*.npy 
