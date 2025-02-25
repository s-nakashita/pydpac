#!/bin/sh
# This is a run script for Nesting Lorenz experiment
export OMP_NUM_THREADS=4
#alias python=python3.9
model="l05nestm"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear"
perturbations="envar"
na=1000 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=30 # observation volume
linf=True # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
extfcst=False # for NMC
#lgsig=110
#llsig=70
opt=0
functype=gc5
#a=-0.1
ntrunc=12
coef_a=None
hyper_mu=0.0
obsloc=${1}
exp="var_vs_envar_m${nmem}obs${nobs}"
echo ${exp}
cdir=` pwd `
ddir=${cdir}/work/${model}
preGM=False
preGMda="envar"
preGMdir="${ddir}/var_vs_envar_dscl_m${nmem}obs${nobs}"
wdir=${ddir}/${exp}
if [ ! -d $wdir ]; then
  echo "No such directory ${wdir}"
  exit
fi
cd $wdir
rseed=`date +%s | cut -c5-10`
rseed=`expr $rseed + 0`
touch timer_fcst
for op in ${operators}; do
  for pert in ${perturbations}; do
    echo $pert
    cp ${cdir}/analysis/config/config_${pert}_sample.py config.py
    gsed -i -e "2i \ \"op\":\"${op}\"," config.py
    gsed -i -e "2i \ \"na\":${na}," config.py
    gsed -i -e "2i \ \"nobs\":${nobs}," config.py
    gsed -i -e "2i \ \"rseed\":${rseed}," config.py
    gsed -i -e "/nmem/s/40/${nmem}/" config.py
    mv config.py config_gm.py
    cp config_gm.py config_lam.py
    ###
    gsed -i -e "3i \ \"extfcst\":${extfcst}," config_gm.py
    if [ $pert = var_nest ]; then
      gsed -i -e "/pt/s/\"${pert}\"/\"var\"/" config_gm.py
    fi
    if [ $pert = envar_nest ] || [ $pert = envar_nestc ]; then
      gsed -i -e "/pt/s/\"${pert}\"/\"envar\"/" config_gm.py
    fi
    if [ $pert = mlef_nest ] || [ $pert = mlef_nestc ]; then
      gsed -i -e "/pt/s/\"${pert}\"/\"mlef\"/" config_gm.py
    fi
    gsed -i -e "3i \ \"ntmax\":8," config_gm.py
    #gsed -i -e "4i \ \"save1h\":True," config_gm.py
    ### precomputed GM
    if [ $preGM = True ]; then
      gsed -i -e "6i \ \"lamstart\":40," config_lam.py # spinup
      gsed -i -e "6i \ \"preGM\":${preGM}," config_lam.py
      gsed -i -e "6i \ \"preGMdir\":\"${preGMdir}\"," config_lam.py
      gsed -i -e "6i \ \"preGMda\":\"${preGMda}\"," config_lam.py
    fi
    ### gmonly
    #gsed -i -e "3i \ \"lamstart\":2000," config_lam.py
    ###
    cat config_gm.py
    cat config_lam.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config_lam.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    start_time=$(date +"%s")
    python ${cdir}/l05nest_fcst.py ${model} ${opt} > ${model}_fcst_${op}_${pert}.log 2>&1 || exit 2
    wait
    end_time=$(date +"%s")
    echo "${op} ${pert}" >> timer_fcst
    echo "scale=3; (${end_time}-${start_time})/1000" | bc >> timer_fcst
  done
  #rm obs*.npy
done
cat timer_fcst
#rm ${model}*.txt 
#rm ${model}_*_cycle*.npy 
