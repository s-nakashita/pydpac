#!/bin/sh
# This is a run script for Nesting Lorenz experiment
export OMP_NUM_THREADS=4
alias python=python3.9
model="l05nest"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="envar envar_nest"
na=300 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=15 # observation volume
linf=True # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
ptype=sigo
functype=gc5
#lgsig=110
#llsig=70
#exp="letkf_K15_${ptype}_mem${nmem}obs${nobs}"
exp="envar+envar_nest_${ptype}_mem${nmem}obs${nobs}"
#exp="var_${functype}nmc_${ptype}_obs${nobs}"
#exp="${datype}_loc_hint"
echo ${exp}
cdir=` pwd `
wdir=work/${model}_K15/${exp}
rm -rf ${wdir}
mkdir -p ${wdir}
cd ${wdir}
cp ${cdir}/logging_config.ini .
ln -fs ${cdir}/data/l05III/truth.npy .
rm -rf obs*.npy
rm -rf *.log
rm -rf timer
touch timer
nmemlist="40 80 120 160 200 240"
lsiglist="20 30 40 50 60 70 80 90 100"
nobslist="480 240 120 60 30 15"
sigolist="1.0 0.5 0.3 0.1 0.05 0.03"
infllist="1.0 1.01 1.02 1.03 1.04 1.05"
sigblist="0.1 0.2 0.4 0.6 0.8 1.0"
#sigblist="0.8 1.2 1.6 2.0 2.4 2.8"
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
  for sigo in ${sigolist}; do
    echo $sigo >> params.txt
    ptmp=$sigo
  #for infl in ${infllist}; do
  #  echo $infl >> params.txt
  #  ptmp=$infl
  #for gsigb in ${sigblist}; do
  #for lsigb in ${sigblist}; do
  #  echo $gsigb $lsigb >> params.txt
  #  ptmp=g${gsigb}l${lsigb}
  #for glb in ${lblist}; do
  #for llb in ${lblist}; do
  #  echo $glb $llb >> params.txt
  #  ptmp=g${glb}l${llb}
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
      #sed -i -e '/ss/s/False/True/' config.py
      #sed -i -e '/getkf/s/True/False/' config.py
      if [ $ptype = sigo ]; then
        gsed -i -e "6i \ \"sigo\":${sigo}," config.py
      fi
#      if [ $ptype = loc ]; then
#        gsed -i -e "6i \ \"lsig\":${lsig}," config.py
#      fi
      if [ $ptype = infl ]; then
        gsed -i -e "8i \ \"infl_parm\":${infl}," config.py
      fi
      ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
      pt=${ptline#\"*}; pt=${pt%\"*}
      echo $pt
      mv config.py config_gm.py
      cp config_gm.py config_lam.py
      if [ ! -z $lgsig ]; then
        gsed -i -e "8i \ \"lsig\":${lgsig}," config_gm.py
      elif [ $ptype = loc ]; then
        gsed -i -e "8i \ \"lsig\":${lsig}," config_gm.py
      fi
      if [ ! -z $llsig ]; then
        gsed -i -e "8i \ \"lsig\":${llsig}," config_lam.py
      elif [ $ptype = loc ]; then
        gsed -i -e "8i \ \"lsig\":${lsig}," config_lam.py
      fi
      if [ $ptype = sigb ]; then
        gsed -i -e "6i \ \"sigb\":${gsigb}," config_gm.py
        gsed -i -e "6i \ \"sigb\":${lsigb}," config_lam.py
        #gsed -i -e "6i \ \"sigv\":${gsigb}," config_lam.py
      fi
      if [ $ptype = lb ]; then
        gsed -i -e "6i \ \"lb\":${glb}," config_gm.py
        gsed -i -e "6i \ \"lb\":${llb}," config_lam.py
        gsed -i -e "6i \ \"lv\":${glb}," config_lam.py
      fi
      ###
      gsed -i -e "6i \ \"functype\":\"${functype}\"," config_gm.py
      gsed -i -e "6i \ \"functype\":\"${functype}\"," config_lam.py
      ### gmonly
      #gsed -i -e "3i \ \"lamstart\":2000," config_lam.py
      ###
      cat config_gm.py
      cat config_lam.py
      for count in $(seq 1 10); do
        echo $count
        rm -f ${model}_*_${op}_${pt}.txt
        start_time=$(date +"%s")
        python ${cdir}/${model}.py > ${model}_${op}_${pert}.log 2>&1
        wait
        end_time=$(date +"%s")
        cputime=`echo "scale=3; (${end_time}-${start_time})/1000" | bc`
        echo "${op} ${pert} ${count} ${cputime}" >> timer
        mv ${model}_e_gm_${op}_${pt}.txt e_gm${ptmp}_${op}_${pt}_${count}.txt
        mv ${model}_stda_gm_${op}_${pt}.txt stda_gm${ptmp}_${op}_${pt}_${count}.txt
        mv ${model}_xdmean_gm_${op}_${pt}.txt xdmean_gm${ptmp}_${op}_${pt}_${count}.txt
        mv ${model}_xsmean_gm_${op}_${pt}.txt xsmean_gm${ptmp}_${op}_${pt}_${count}.txt
        mv ${model}_e_lam_${op}_${pt}.txt e_lam${ptmp}_${op}_${pt}_${count}.txt
        mv ${model}_stda_lam_${op}_${pt}.txt stda_lam${ptmp}_${op}_${pt}_${count}.txt
        mv ${model}_xdmean_lam_${op}_${pt}.txt xdmean_lam${ptmp}_${op}_${pt}_${count}.txt
        mv ${model}_xsmean_lam_${op}_${pt}.txt xsmean_lam${ptmp}_${op}_${pt}_${count}.txt
        rm obs*.npy
      done
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} e_gm${ptmp} ${pt}
      #rm e_gm_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} stda_gm${ptmp} ${pt}
      #rm stda_gm_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} xdmean_gm${ptmp} ${pt}
      #rm xdmean_gm_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} xsmean_gm${ptmp} ${pt}
      #rm xsmean_gm_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} e_lam${ptmp} ${pt}
      #rm e_lam_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} stda_lam${ptmp} ${pt}
      #rm stda_lam_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} xdmean_lam${ptmp} ${pt}
      #rm xdmean_lam_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} xsmean_lam${ptmp} ${pt}
      #rm xsmean_lam_${op}_${pt}_*.txt
      mv ${model}_e_gm${ptmp}_${op}_${pt}.txt ${model}_e_gm_${op}_${pert}_${ptmp}.txt
      mv ${model}_stda_gm${ptmp}_${op}_${pt}.txt ${model}_stda_gm_${op}_${pert}_${ptmp}.txt
      mv ${model}_xdmean_gm${ptmp}_${op}_${pt}.txt ${model}_xdmean_gm_${op}_${pert}_${ptmp}.txt
      mv ${model}_xsmean_gm${ptmp}_${op}_${pt}.txt ${model}_xsmean_gm_${op}_${pert}_${ptmp}.txt
      mv ${model}_e_lam${ptmp}_${op}_${pt}.txt ${model}_e_lam_${op}_${pert}_${ptmp}.txt
      mv ${model}_stda_lam${ptmp}_${op}_${pt}.txt ${model}_stda_lam_${op}_${pert}_${ptmp}.txt
      mv ${model}_xdmean_lam${ptmp}_${op}_${pt}.txt ${model}_xdmean_lam_${op}_${pert}_${ptmp}.txt
      mv ${model}_xsmean_lam${ptmp}_${op}_${pt}.txt ${model}_xsmean_lam_${op}_${pert}_${ptmp}.txt
    done #pert
    #python ${cdir}/plot/plote.py ${op} ${model} ${na} #mlef
  done #ptmp
  #done #ptmp (sigb or lb)
  cat params.txt
  if [ $ptype = sigb ] || [ $ptype = lb ]; then
  python ${cdir}/plot/ploteparam2d_nest.py ${op} ${model} ${na} $ptype
  python ${cdir}/plot/plotxdparam2d_nest.py ${op} ${model} ${na} $ptype
  else
  python ${cdir}/plot/ploteparam_nest.py ${op} ${model} ${na} $ptype
  python ${cdir}/plot/plotxdparam_nest.py ${op} ${model} ${na} $ptype
  fi
  #rm obs*.npy
done #op
#rm ${model}*.txt 
#rm ${model}*.npy 
