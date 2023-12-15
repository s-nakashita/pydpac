#!/bin/sh
# This is a run script for Lorenz05 experiment
model="l05III"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="var mlef"
na=30 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=30 # observation volume
linf=True  # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
#L="-1.0 0.5 1.0 2.0"
#lsig=120
#exp="mlef_mem${nmem}obs${nobs}"
exp="var+mlef_obs${nobs}"
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
ln -fs ${cdir}/data/l05III/truth.npy .
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
      gsed -i -e "6i \ \"lsig\":${lsig}," config.py
    fi
    cat config.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    start_time=$(date +"%s")
    python3.9 ${cdir}/l05.py ${model} > ${model}_${op}_${pert}.log 2>&1
    wait
    end_time=$(date +"%s")
    echo "${op} ${pert}" >> timer
    echo "scale=3; ${end_time}-${start_time}" | bc >> timer
    mv ${model}_e_${op}_${pt}.txt e_${op}_${pert}.txt
    mv ${model}_stda_${op}_${pt}.txt stda_${op}_${pert}.txt
    mv ${model}_xdmean_${op}_${pt}.txt xdmean_${op}_${pert}.txt
    mv ${model}_xsmean_${op}_${pt}.txt xsmean_${op}_${pert}.txt
    loctype=`echo $pert | cut -c5-5`
    if [ "${loctype}" = "b" ]; then
    mv ${model}_rho_${op}_${pt}.npy ${model}_rho_${op}_${pert}.npy
    fi
    for icycle in $(seq 0 $((${na} - 1))); do
      if test -e wa_${op}_${pt}_cycle${icycle}.npy; then
        mv wa_${op}_${pt}_cycle${icycle}.npy ${pert}/wa_${op}_cycle${icycle}.npy
      fi
      if test -e ${model}_ua_${op}_${pt}_cycle${icycle}.npy; then
        mv ${model}_ua_${op}_${pt}_cycle${icycle}.npy ${pert}/ua_${op}_${pert}_cycle${icycle}.npy
      fi
    #  mv Wmat_${op}_${pt}_cycle${icycle}.npy ${pert}/Wmat_${op}_cycle${icycle}.npy
    #  mv ${model}_K_${op}_${pt}_cycle$icycle.npy ${model}_K_${op}_${pert}_cycle$icycle.npy
    #  mv ${model}_dxaorig_${op}_${pt}_cycle$icycle.npy ${model}_dxaorig_${op}_${pert}_cycle$icycle.npy
    #  mv ${model}_dxa_${op}_${pt}_cycle$icycle.npy ${model}_dxa_${op}_${pert}_cycle$icycle.npy
    #  mv ${model}_pa_${op}_${pt}_cycle$icycle.npy ${model}_pa_${op}_${pert}_cycle$icycle.npy
    #  mv ${model}_pf_${op}_${pt}_cycle$icycle.npy ${model}_pf_${op}_${pert}_cycle$icycle.npy
    #  mv ${model}_spf_${op}_${pt}_cycle$icycle.npy ${model}_spf_${op}_${pert}_cycle$icycle.npy
    #  if [ "${pert:4:1}" = "b" ]; then
    #  mv ${model}_lpf_${op}_${pt}_cycle$icycle.npy ${model}_lpf_${op}_${pert}_cycle$icycle.npy
    #  mv ${model}_lspf_${op}_${pt}_cycle$icycle.npy ${model}_lspf_${op}_${pert}_cycle$icycle.npy
    #  fi
    done
    #python3.9 ${cdir}/plot/plotk.py ${op} ${model} ${na} ${pert}
    #python3.9 ${cdir}/plot/plotdxa.py ${op} ${model} ${na} ${pert}
    #python3.9 ${cdir}/plot/plotpf.py ${op} ${model} ${na} ${pert}
    #python3.9 ${cdir}/plot/plotlpf.py ${op} ${model} ${na} ${pert} 
    #done
  done
  python3.9 ${cdir}/plot/plote.py ${op} ${model} ${na} #mlef
  python3.9 ${cdir}/plot/plotxd.py ${op} ${model} ${na} #mlef
  #python3.9 ${cdir}/plot/plotchi.py ${op} ${model} ${na}
  #python3.9 ${cdir}/plot/plotinnv.py ${op} ${model} ${na} > innv_${op}.log
  python3.9 ${cdir}/plot/plotxa.py ${op} ${model} ${na}
  #python3.9 ${cdir}/plot/plotdof.py ${op} ${model} ${na}
  python3.9 ${cdir}/plot/ploterrspectra.py ${op} ${model} ${na}
  
  #rm obs*.npy
done
#rm ${model}*.txt 
#rm ${model}*.npy 
