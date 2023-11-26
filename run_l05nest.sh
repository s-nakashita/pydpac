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
na=50 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=120 # observation volume
linf=True  # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
#L="-1.0 0.5 1.0 2.0"
exp="test_m${nmem}obs${nobs}"
#exp="${datype}_loc_hint"
echo ${exp}
cdir=` pwd `
rm -rf work/${model}/${exp}
mkdir -p work/${model}/${exp}
cd work/${model}/${exp}
cp ${cdir}/logging_config.ini .
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
    mv config.py config_gm.py
    cp config_gm.py config_lam.py
    cat config_gm.py
    cat config_lam.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config_gm.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    start_time=$(gdate +"%s.%5N")
    python ${cdir}/${model}.py > ${model}_${op}_${pert}.log 2>&1
    wait
    end_time=$(gdate +"%s.%5N")
    echo "${op} ${pert}" >> timer
    echo "scale=1; ${end_time}-${start_time}" | bc >> timer
    mv ${model}_e_gm_${op}_${pt}.txt e_gm_${op}_${pert}.txt
    mv ${model}_stda_gm_${op}_${pt}.txt stda_gm_${op}_${pert}.txt
    mv ${model}_xdmean_gm_${op}_${pt}.txt xdmean_gm_${op}_${pert}.txt
    mv ${model}_xsmean_gm_${op}_${pt}.txt xsmean_gm_${op}_${pert}.txt
    mv ${model}_e_lam_${op}_${pt}.txt e_lam_${op}_${pert}.txt
    mv ${model}_stda_lam_${op}_${pt}.txt stda_lam_${op}_${pert}.txt
    mv ${model}_xdmean_lam_${op}_${pt}.txt xdmean_lam_${op}_${pert}.txt
    mv ${model}_xsmean_lam_${op}_${pt}.txt xsmean_lam_${op}_${pert}.txt
    #if [ "${pert:4:1}" = "b" ]; then
    #mv ${model}_rho_${op}_${pt}.npy ${model}_rho_${op}_${pert}.npy
    #fi
    for icycle in $(seq 0 $((${na} - 1))); do
      #if test -e wa_${op}_${pt}_cycle${icycle}.npy; then
      #  mv wa_${op}_${pt}_cycle${icycle}.npy ${pert}/wa_${op}_cycle${icycle}.npy
      #fi
      if test -e ${model}_ua_gm_${op}_${pt}_cycle${icycle}.npy; then
        mv ${model}_ua_gm_${op}_${pt}_cycle${icycle}.npy ${pert}/ua_gm_${op}_${pert}_cycle${icycle}.npy
      fi
      if test -e ${model}_ua_lam_${op}_${pt}_cycle${icycle}.npy; then
        mv ${model}_ua_lam_${op}_${pt}_cycle${icycle}.npy ${pert}/ua_lam_${op}_${pert}_cycle${icycle}.npy
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
  
  #rm obs*.npy
done
#rm ${model}*.txt 
#rm ${model}*.npy 
