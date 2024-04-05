#!/bin/sh
# This is a run script for Nesting Lorenz experiment
export OMP_NUM_THREADS=4
#alias python=python3.9
model="l05nestm"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="envar_nest"
na=240 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=30 # observation volume
linf=True # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
ntest=10
ptype=infl_lrg
functype=gc5
#lgsig=110
#llsig=70
#exp="letkf_K15_${ptype}_mem${nmem}obs${nobs}"
exp="envar_nest_${ptype}_mem${nmem}obs${nobs}"
#exp="var_nest_${functype}nmc_${ptype}_obs${nobs}"
#exp="${datype}_loc_hint"
echo ${exp}
cdir=` pwd `
wdir=work/${model}/${exp}
rm -rf ${wdir}
mkdir -p ${wdir}
cd ${wdir}
cp ${cdir}/logging_config.ini .
if [ $model = l05nest ]; then
ln -fs ${cdir}/data/l05III/truth.npy .
elif [ $model = l05nestm ]; then
ln -fs ${cdir}/data/l05IIIm/truth.npy .
fi
rm -rf obs*.npy
rm -rf *.log
rm -rf timer
touch timer
nmemlist="40 80 120 160 200 240"
lsiglist="20 30 40 50 60 70 80 90 100"
nobslist="480 240 120 60 30 15"
sigolist="1.0 0.5 0.3 0.1 0.05 0.03"
infllist="1.0 1.05 1.1 1.15 1.2 1.25"
sigblist="0.4 0.6 0.8 1.0 1.2 1.6"
sigvlist="0.4 0.6 0.8 1.0 1.2 1.6"
#sigblist="0.8 1.0 1.2 1.6"
lblist="2.0 4.0 6.0 8.0 10.0 12.0"
## create seeds
touch seeds.txt
rseed=`date +%s | cut -c5-10`
for i in $(seq 1 $ntest);do
rseed1=`expr $rseed + \( 2 \* $i \)`
rseed2=`expr $rseed + \( 2 \* $i + 1 \)`
echo $rseed1 $rseed2 >> seeds.txt
done
##
touch params.txt
set -e
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
  #for sigo in ${sigolist}; do
  #  echo $sigo >> params.txt
  #  ptmp=$sigo
  for infl in ${infllist}; do
    echo $infl >> params.txt
    ptmp=$infl
  #for sigb in ${sigblist}; do
  #  echo $sigb >> params.txt
  #  ptmp=$sigb
  #for sigv in ${sigvlist}; do
  #  echo $sigv >> params.txt
  #  ptmp=$sigv
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
#      if [ $ptype = infl ]; then
#        gsed -i -e "8i \ \"infl_parm\":${infl}," config.py
#      fi
#      if [ $ptype = sigb ]; then
#        gsed -i -e "6i \ \"sigb\":${sigb}," config.py
#      fi
      ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
      pt=${ptline#\"*}; pt=${pt%\"*}
      echo $pt
      mv config.py config_gm.py
      cp config_gm.py config_lam.py
      if [ $ptype = infl ]; then
        gsed -i -e "8i \ \"infl_parm\":1.1," config_gm.py
        gsed -i -e "8i \ \"infl_parm\":${infl}," config_lam.py
      fi
      if [ $ptype = infl_lrg ]; then
        gsed -i -e "8i \ \"infl_parm_lrg\":${infl}," config_lam.py
      fi
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
        #gsed -i -e "6i \ \"sigb\":${gsigb}," config_gm.py
        gsed -i -e "6i \ \"sigb\":1.0," config_gm.py
        gsed -i -e "6i \ \"sigb\":${sigb}," config_lam.py
        gsed -i -e "6i \ \"sigv\":1.0," config_lam.py
      fi
      if [ $ptype = lb ]; then
        gsed -i -e "6i \ \"lb\":${glb}," config_gm.py
        gsed -i -e "6i \ \"lb\":${llb}," config_lam.py
        gsed -i -e "6i \ \"lv\":${glb}," config_lam.py
      fi
      if [ $ptype = sigv ]; then
        gsed -i -e "6i \ \"sigv\":${sigv}," config_lam.py
      fi
      ###
      gsed -i -e "6i \ \"functype\":\"${functype}\"," config_gm.py
      gsed -i -e "6i \ \"functype\":\"${functype}\"," config_lam.py
      ### gmonly
      #gsed -i -e "3i \ \"lamstart\":2000," config_lam.py
      ###
      if [ $pert = var_nest ]; then
      gsed -i -e "/pt/s/\"${pert}\"/\"var\"/" config_gm.py
      fi
      if [ $pert = envar_nest ]; then
        gsed -i -e "/pt/s/\"${pert}\"/\"envar\"/" config_gm.py
      fi
      cat config_gm.py
      cat config_lam.py
      gsed -i -e "2i \ \"save_hist\":False," config_gm.py
      gsed -i -e "2i \ \"save_dh\":False," config_gm.py
      cp config_gm.py config_gm_orig.py
      for count in $(seq 1 $ntest); do
        echo $count
        rseed1=`awk '(NR=='$count'){print $1}' seeds.txt`
        rseed2=`awk '(NR=='$count'){print $2}' seeds.txt`
        cp config_gm_orig.py config_gm.py
        gsed -i -e "2i \ \"rseed\":${rseed1}," config_gm.py
        gsed -i -e "2i \ \"roseed\":${rseed2}," config_gm.py
        rm -f ${model}_*_${op}_${pt}.txt
        start_time=$(date +"%s")
        python ${cdir}/l05nest.py ${model} > ${model}_${op}_${pert}.log 2>&1
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
#  if [ $ptype = sigb ] || [ $ptype = lb ]; then
#  python ${cdir}/plot/ploteparam2d_nest.py ${op} ${model} ${na} $ptype
#  python ${cdir}/plot/plotxdparam2d_nest.py ${op} ${model} ${na} $ptype
#  else
  python ${cdir}/plot/ploteparam_nest.py ${op} ${model} ${na} $ptype
  python ${cdir}/plot/plotxdparam_nest.py ${op} ${model} ${na} $ptype
#  fi
  #rm obs*.npy
done #op
#rm ${model}*.txt 
#rm ${model}*.npy 
