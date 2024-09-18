#!/bin/sh
# This is a run script for Nesting Lorenz experiment
export OMP_NUM_THREADS=4
#alias python=python3.9
model="l05nestm"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear"
perturbations="envar_nest envar var_nest var"
na=100 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=30 # observation volume
linf=True # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
extfcst=False # for NMC
blending=True # LSB
blsb=False
alsb=True
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
#if [ ! -d $wdir ]; then
#  echo "No such directory ${wdir}"
#  exit
#fi
#rm -rf $wdir
mkdir -p $wdir
cd $wdir
cp ${cdir}/logging_config.ini .
if [ ${model} = l05nest ]; then
ln -fs ${cdir}/data/l05III/truth.npy .
elif [ ${model} = l05nestm ]; then
ln -fs ${cdir}/data/l05IIIm/truth.npy .
#ln -fs ${ddir}/truth.npy .
fi
rm -rf obs*.npy
rm -rf *.log
rm -rf timer
touch timer
if [ $preGM = True ]; then
  cp ${preGMdir}/obs*.npy .
fi
rseed=`date +%s | cut -c5-10`
rseed=`expr $rseed + 0`
roseed=514
mkdir -p data
for op in ${operators}; do
  for pert in ${perturbations}; do
    echo $pert
    cp ${cdir}/analysis/config/config_${pert}_sample.py config.py
    gsed -i -e "2i \ \"op\":\"${op}\"," config.py
    gsed -i -e "2i \ \"na\":${na}," config.py
    gsed -i -e "2i \ \"nobs\":${nobs}," config.py
    gsed -i -e "2i \ \"rseed\":${rseed}," config.py
    gsed -i -e "2i \ \"roseed\":${roseed}," config.py
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
    if [ ! -z $lsig ]; then
    gsed -i -e "8i \ \"lsig\":${lsig}," config.py
    fi
    #gsed -i -e "5i \ \"obsloctype\":\"partial\"," config.py
    mv config.py config_gm.py
    cp config_gm.py config_lam.py
    if [ ! -z $lgsig ]; then
    gsed -i -e "8i \ \"lsig\":${lgsig}," config_gm.py
    fi
    if [ ! -z $llsig ]; then
    gsed -i -e "8i \ \"lsig\":${llsig}," config_lam.py
    fi
    ### climatological B settings
    #gsed -i -e "6i \ \"lb\":17.07," config_gm.py
    #gsed -i -e "6i \ \"lb\":33.48," config_lam.py
    #gsed -i -e "6i \ \"a\":0.21," config_gm.py
    #gsed -i -e "6i \ \"a\":-0.14," config_lam.py
    ###
    #gsed -i -e "6i \ \"sigb\":0.8," config_gm.py
    #gsed -i -e "6i \ \"sigb\":0.8," config_lam.py
    ##
    gsed -i -e "6i \ \"functype\":\"${functype}\"," config_gm.py
    gsed -i -e "6i \ \"functype\":\"${functype}\"," config_lam.py
    ### climatological V settings
    #if [ $pert = var_nest ]; then
    #gsed -i -e "6i \ \"a_v\":${a}," config_lam.py
    ##gsed -i -e "6i \ \"lv\":4," config_lam.py
    #fi
    gsed -i -e "6i \ \"ntrunc\":${ntrunc}," config_lam.py
    #if [ $pert = var_nestc ] || [ $pert = envar_nestc ]; then
    #  gsed -i -e "6i \ \"ortho\":True," config_lam.py
    #  gsed -i -e "7i \ \"coef_a\":${coef_a}," config_lam.py
    #fi
    #if [ $pert = envar_nestc ]; then
    #  gsed -i -e "6i \ \"ridge\":True," config_lam.py
    #  #gsed -i -e "6i \ \"ridge\":False," config_lam.py
    #  #gsed -i -e "6i \ \"reg\":True," config_lam.py
    #  gsed -i -e "6i \ \"hyper_mu\":${hyper_mu}," config_lam.py
    #fi
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
    ### LSB
    if [ $blending = True ]; then
      gsed -i -e "6i \ \"blending\":${blending}," config_lam.py
      gsed -i -e "7i \ \"blsb\":${blsb}," config_lam.py
      gsed -i -e "8i \ \"alsb\":${alsb}," config_lam.py
    fi
    ### precomputed GM
    if [ $preGM = True ]; then
      gsed -i -e "6i \ \"lamstart\":40," config_lam.py # spinup
      gsed -i -e "6i \ \"preGM\":${preGM}," config_lam.py
      gsed -i -e "6i \ \"preGMdir\":\"${preGMdir}/data/${preGMda}\"," config_lam.py
      gsed -i -e "6i \ \"preGMda\":\"${preGMda}\"," config_lam.py
    fi
    ###
    cat config_gm.py
    cat config_lam.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config_lam.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    ##for nmc
    #gsed -i -e "6i \ \"extfcst\":True," config_gm.py
    start_time=$(date +"%s")
    python ${cdir}/l05nest.py ${model} ${opt} > ${model}_${op}_${pert}.log 2>&1 || exit 2
    wait
    end_time=$(date +"%s")
    echo "${op} ${pert}" >> timer
    echo "scale=3; (${end_time}-${start_time})/1000" | bc >> timer
    mv ${model}_e_gm_${op}_${pt}.txt e_gm_${op}_${pert}.txt
    mv ${model}_stda_gm_${op}_${pt}.txt stda_gm_${op}_${pert}.txt
    mv ${model}_xdmean_gm_${op}_${pt}.txt xdmean_gm_${op}_${pert}.txt
    mv ${model}_xsmean_gm_${op}_${pt}.txt xsmean_gm_${op}_${pert}.txt
    mv ${model}_ef_gm_${op}_${pt}.txt ef_gm_${op}_${pert}.txt
    mv ${model}_stdf_gm_${op}_${pt}.txt stdf_gm_${op}_${pert}.txt
    mv ${model}_xdfmean_gm_${op}_${pt}.txt xdfmean_gm_${op}_${pert}.txt
    mv ${model}_xsfmean_gm_${op}_${pt}.txt xsfmean_gm_${op}_${pert}.txt
    mv ${model}_xagm_${op}_${pt}.npy xagm_${op}_${pert}.npy
    mv ${model}_xsagm_${op}_${pt}.npy xsagm_${op}_${pert}.npy
    mv ${model}_xfgm_${op}_${pt}.npy xfgm_${op}_${pert}.npy
    mv ${model}_e_lam_${op}_${pt}.txt e_lam_${op}_${pert}.txt
    mv ${model}_stda_lam_${op}_${pt}.txt stda_lam_${op}_${pert}.txt
    mv ${model}_xdmean_lam_${op}_${pt}.txt xdmean_lam_${op}_${pert}.txt
    mv ${model}_xsmean_lam_${op}_${pt}.txt xsmean_lam_${op}_${pert}.txt
    mv ${model}_ef_lam_${op}_${pt}.txt ef_lam_${op}_${pert}.txt
    mv ${model}_stdf_lam_${op}_${pt}.txt stdf_lam_${op}_${pert}.txt
    mv ${model}_xdfmean_lam_${op}_${pt}.txt xdfmean_lam_${op}_${pert}.txt
    mv ${model}_xsfmean_lam_${op}_${pt}.txt xsfmean_lam_${op}_${pert}.txt
    mv ${model}_xalam_${op}_${pt}.npy xalam_${op}_${pert}.npy
    mv ${model}_xsalam_${op}_${pt}.npy xsalam_${op}_${pert}.npy
    mv ${model}_xflam_${op}_${pt}.npy xflam_${op}_${pert}.npy
    python ${cdir}/plot/plotpf_nest.py ${op} ${model} ${na} ${pert} 1 100
    mkdir -p data/${pert}
    for vname in d dh dx pa pf spf ua uf; do
      mv ${model}_*_${vname}_${op}_${pert}_cycle*.npy data/${pert}
    done
    if [ $pt = var_nest ] || \
    [ $pt = envar_nest ] || [ $pt = envar_nestc ] || \
    [ $pt = mlef_nest ] || [ $pt = mlef_nestc ]; then
      for vname in dk svmat qmat dk2 hess tmat heinv zbmat zvmat dxc1 dxc2 schur;do
        mv ${model}_*_${vname}_${op}_${pert}_cycle*.npy data/${pert}
      done
    fi
    #if [ $pt = envar_nestc ]; then
    #  python ${cdir}/plot/plotcoef_a_nest.py ${op} ${model} ${na} 
    #  mv ${model}_lam_coef_a_${op}_${pert}.txt data/$pert
    #  rm ${model}_lam_coef_a_${op}_${pert}_cycle*.txt
    #fi
  done
  python ${cdir}/plot/plote_nest.py ${op} ${model} ${na}
  python ${cdir}/plot/plote_nest.py ${op} ${model} ${na} F
  python ${cdir}/plot/plotxd_nest.py ${op} ${model} ${na}
  #python ${cdir}/plot/plotchi.py ${op} ${model} ${na}
  #python ${cdir}/plot/plotinnv.py ${op} ${model} ${na} > innv_${op}.log
  python ${cdir}/plot/plotxa_nest.py ${op} ${model} ${na}
  #python ${cdir}/plot/plotdof.py ${op} ${model} ${na}
  python ${cdir}/plot/ploterrspectra_nest.py ${op} ${model} ${na}
  if [ ${extfcst} = True ]; then 
  python ${cdir}/plot/nmc_nest.py ${op} ${model} ${na}
  fi
  python ${cdir}/plot/plotjh+gh_nest.py ${op} ${model} ${na} && \
  rm ${model}_*_jh_${op}_*_cycle*.txt && \
  rm ${model}_*_gh_${op}_*_cycle*.txt
  rm ${model}_*_alpha_${op}_*_cycle*.txt
  #rm obs*.npy
done
#rm ${model}*.txt 
#rm ${model}_*_cycle*.npy 
mkdir -p data
mv ${model}_*_cycle*.npy data/
