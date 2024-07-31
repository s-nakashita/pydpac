#!/bin/sh
# This is a run script for Nesting Lorenz experiment
export OMP_NUM_THREADS=4
#alias python=python3.9
model="l05nestm"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="var_nest var envar_nest envar"
#perturbations="envar"
#perturbations="envar_nestc"
#datype="4dmlef"
#perturbations="4dvar 4dletkf ${datype}be ${datype}bm ${datype}cw ${datype}y"
#perturbations="lmlefcw lmlefy mlef"
#perturbations="mlef 4dmlef mlefbe"
#perturbations="etkfbm"
na=1000 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=30 # observation volume
linf=True # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
extfcst=False # for NMC
#lgsig=110
#llsig=70
#L="-1.0 0.5 1.0 2.0"
opt=0
functype=gc5
#a=-0.1
ntrunc=12
coef_a=None
hyper_mu=0.0
#exp="var+var_nest_${functype}nmc_obs${nobs}"
#exp="var_vs_envar_dscl_m${nmem}obs${nobs}"
#exp="var_vs_envar_preGM_m${nmem}obs${nobs}"
exp="var_vs_envar_shrink_dct_preGM_partialm_m${nmem}obs${nobs}"
#exp="mlef_dscl_m${nmem}obs${nobs}"
#exp="envar_nestc_reg${hyper_mu}_shrink_preGM_m${nmem}obs${nobs}"
#exp="envar_nestc_a_shrink_preGM_m${nmem}obs${nobs}"
#exp="var_vs_envar_ntrunc${ntrunc}_m${nmem}obs${nobs}" #lg${lgsig}l${llsig}"
#exp="var_nmc6_obs${nobs}"
echo ${exp}
cdir=` pwd `
ddir=${cdir}/work/${model}
#ddir=/Volumes/FF520/nested_envar/data/${model}
preGM=True
preGMda="envar"
preGMdir="${ddir}/var_vs_envar_dscl_m${nmem}obs${nobs}"
#preGMdir="${ddir}/${preGMda}_dscl_m${nmem}obs${nobs}"
#preGMdir="${ddir}/var_vs_envar_nest_ntrunc${ntrunc}_m${nmem}obs${nobs}"
wdir=${ddir}/${exp}
mkdir -p $wdir
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
    ### gmonly
    #gsed -i -e "3i \ \"lamstart\":2000," config_lam.py
    ### precomputed GM
    if [ $preGM = True ]; then
      gsed -i -e "6i \ \"lamstart\":40," config_lam.py # spinup
      gsed -i -e "6i \ \"preGM\":${preGM}," config_lam.py
      gsed -i -e "6i \ \"preGMdir\":\"${preGMdir}\"," config_lam.py
      gsed -i -e "6i \ \"preGMda\":\"${preGMda}\"," config_lam.py
    fi
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
#rm ${model}*.txt 
#rm ${model}_*_cycle*.npy 
