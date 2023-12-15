#!/bin/sh
# This is a run script for Nesting Lorenz experiment
model="l05nest"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="mlef"
#datype="4dmlef"
#perturbations="4dvar 4dletkf ${datype}be ${datype}bm ${datype}cw ${datype}y"
#perturbations="lmlefcw lmlefy mlef"
#perturbations="mlef 4dmlef mlefbe"
#perturbations="etkfbm"
na=30 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=240 # observation volume
linf=True # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
#lgsig=110
#llsig=70
#L="-1.0 0.5 1.0 2.0"
exp="mlef_m${nmem}obs${nobs}" #lg${lgsig}l${llsig}"
#exp="${datype}_loc_hint"
echo ${exp}
cdir=` pwd `
rm -rf work/${model}/${exp}
mkdir -p work/${model}/${exp}
cd work/${model}/${exp}
cp ${cdir}/logging_config.ini .
ln -fs ${cdir}/data/l05III/truth.npy .
rm -rf obs*.npy
rm -rf *.log
rm -rf timer
touch timer
for op in ${operators}; do
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
    if [ ! -z $lsig ]; then
    gsed -i -e "8i \ \"lsig\":${lsig}," config.py
    fi
    mv config.py config_gm.py
    cp config_gm.py config_lam.py
    if [ ! -z $lgsig ]; then
    gsed -i -e "8i \ \"lsig\":${lgsig}," config_gm.py
    fi
    if [ ! -z $llsig ]; then
    gsed -i -e "8i \ \"lsig\":${llsig}," config_lam.py
    fi
    cat config_gm.py
    cat config_lam.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config_gm.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    start_time=$(date +"%s")
    python ${cdir}/${model}.py > ${model}_${op}_${pert}.log 2>&1
    wait
    end_time=$(date +"%s")
    echo "${op} ${pert}" >> timer
    echo "scale=3; (${end_time}-${start_time})/1000" | bc >> timer
    mv ${model}_e_gm_${op}_${pt}.txt e_gm_${op}_${pert}.txt
    mv ${model}_stda_gm_${op}_${pt}.txt stda_gm_${op}_${pert}.txt
    mv ${model}_xdmean_gm_${op}_${pt}.txt xdmean_gm_${op}_${pert}.txt
    mv ${model}_xsmean_gm_${op}_${pt}.txt xsmean_gm_${op}_${pert}.txt
    mv ${model}_xagm_${op}_${pt}.npy ${model}_xagm_${op}_${pert}.npy
    mv ${model}_xsagm_${op}_${pt}.npy ${model}_xsagm_${op}_${pert}.npy
    mv ${model}_e_lam_${op}_${pt}.txt e_lam_${op}_${pert}.txt
    mv ${model}_stda_lam_${op}_${pt}.txt stda_lam_${op}_${pert}.txt
    mv ${model}_xdmean_lam_${op}_${pt}.txt xdmean_lam_${op}_${pert}.txt
    mv ${model}_xsmean_lam_${op}_${pt}.txt xsmean_lam_${op}_${pert}.txt
    mv ${model}_xalam_${op}_${pt}.npy ${model}_xalam_${op}_${pert}.npy
    mv ${model}_xsalam_${op}_${pt}.npy ${model}_xsalam_${op}_${pert}.npy
    loctype=`echo $pert | cut -c5-5`
    if [ "${loctype}" = "b" ]; then
      mv ${model}_rhogm_${op}_${pt}.npy ${model}_rhogm_${op}_${pert}.npy
      mv ${model}_rholam_${op}_${pt}.npy ${model}_rholam_${op}_${pert}.npy
      icycle=$((${na} - 1))
      #for icycle in $(seq 0 $((${na} - 1))); do
      #if test -e wa_${op}_${pt}_cycle${icycle}.npy; then
      #  mv wa_${op}_${pt}_cycle${icycle}.npy ${pert}/wa_${op}_cycle${icycle}.npy
      #fi
      if test -e ${model}_pagm_${op}_${pt}_cycle${icycle}.npy; then
        mv ${model}_pagm_${op}_${pt}_cycle${icycle}.npy ${model}_pagm_${op}_${pert}_cycle${icycle}.npy
      fi
      if test -e ${model}_palam_${op}_${pt}_cycle${icycle}.npy; then
        mv ${model}_palam_${op}_${pt}_cycle${icycle}.npy ${model}_palam_${op}_${pert}_cycle${icycle}.npy
      fi
      #done
      python ${cdir}/plot/plotpa_nest.py ${op} ${model} ${na} ${pert}
    fi
    #python ${cdir}/plot/plotk.py ${op} ${model} ${na} ${pert}
    #python ${cdir}/plot/plotdxa.py ${op} ${model} ${na} ${pert}
    #python ${cdir}/plot/plotpf.py ${op} ${model} ${na} ${pert}
    #python ${cdir}/plot/plotlpf.py ${op} ${model} ${na} ${pert} 
    #done
  done
  python ${cdir}/plot/plote_nest.py ${op} ${model} ${na} mlef
  python ${cdir}/plot/plotxd_nest.py ${op} ${model} ${na} mlef
  #python ${cdir}/plot/plotchi.py ${op} ${model} ${na}
  #python ${cdir}/plot/plotinnv.py ${op} ${model} ${na} > innv_${op}.log
  python ${cdir}/plot/plotxa_nest.py ${op} ${model} ${na}
  #python ${cdir}/plot/plotdof.py ${op} ${model} ${na}
  python ${cdir}/plot/ploterrspectra_nest.py ${op} ${model} ${na}
  
  #rm obs*.npy
done
#rm ${model}*.txt 
#rm ${model}*.npy 
