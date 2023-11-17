#!/bin/sh
# This is a run script for Lorenz96 experiment
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="var 4dvar letkf 4dletkf mlefy 4dmlefy"
#datype="4dmlef"
#perturbations="4dvar 4dletkf ${datype}be ${datype}bm ${datype}cw ${datype}y"
#perturbations="lmlefcw lmlefy mlef"
#perturbations="mlef 4dmlef mlefbe"
perturbations="etkfbm"
na=100 # Number of assimilation cycle
nmem=40 # ensemble size
nobs=40 # observation volume
linf=True  # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
#L="-1.0 0.5 1.0 2.0"
exp="test_ss2"
#exp="${datype}_loc_hint"
echo ${exp}
rm -rf work/${exp}
mkdir -p work/${exp}
cd work/${exp}
cp ../../logging_config.ini .
rm -rf *.npy
rm -rf *.log
rm -rf timer
touch timer
for op in ${operators}; do
  for pert in ${perturbations}; do
    echo $pert
    cp ../../analysis/config/config_${pert}_sample.py config.py
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
    cat config.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    start_time=$(gdate +"%s.%5N")
    python ../../l96.py > l96_${op}_${pert}.log 2>&1
    #python ../../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window} ${iloc} > l96_${op}_${pert}.log 2>&1
    #python ../../l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${lb} > l96_${op}_${pt}_${lb}.log 2>&1
    wait
    end_time=$(gdate +"%s.%5N")
    echo "${op} ${pert}" >> timer
    echo "scale=1; ${end_time}-${start_time}" | bc >> timer
    mv l96_e_${op}_${pt}.txt e_${op}_${pert}.txt
    mv l96_stda_${op}_${pt}.txt stda_${op}_${pert}.txt
    #if [ "${pert:4:1}" = "b" ]; then
    #mv l96_rho_${op}_${pt}.npy l96_rho_${op}_${pert}.npy
    #fi
    for icycle in $(seq 0 $((${na} - 1))); do
      if test -e wa_${op}_${pt}_cycle${icycle}.npy; then
        mv wa_${op}_${pt}_cycle${icycle}.npy ${pert}/wa_${op}_cycle${icycle}.npy
      fi
      if test -e l96_ua_${op}_${pt}_cycle${icycle}.npy; then
        mv l96_ua_${op}_${pt}_cycle${icycle}.npy ${pert}/ua_${op}_${pert}_cycle${icycle}.npy
      fi
    #  mv Wmat_${op}_${pt}_cycle${icycle}.npy ${pert}/Wmat_${op}_cycle${icycle}.npy
    #  mv l96_K_${op}_${pt}_cycle$icycle.npy l96_K_${op}_${pert}_cycle$icycle.npy
    #  mv l96_dxaorig_${op}_${pt}_cycle$icycle.npy l96_dxaorig_${op}_${pert}_cycle$icycle.npy
    #  mv l96_dxa_${op}_${pt}_cycle$icycle.npy l96_dxa_${op}_${pert}_cycle$icycle.npy
    #  mv l96_pa_${op}_${pt}_cycle$icycle.npy l96_pa_${op}_${pert}_cycle$icycle.npy
    #  mv l96_pf_${op}_${pt}_cycle$icycle.npy l96_pf_${op}_${pert}_cycle$icycle.npy
    #  mv l96_spf_${op}_${pt}_cycle$icycle.npy l96_spf_${op}_${pert}_cycle$icycle.npy
    #  if [ "${pert:4:1}" = "b" ]; then
    #  mv l96_lpf_${op}_${pt}_cycle$icycle.npy l96_lpf_${op}_${pert}_cycle$icycle.npy
    #  mv l96_lspf_${op}_${pt}_cycle$icycle.npy l96_lspf_${op}_${pert}_cycle$icycle.npy
    #  fi
    done
    #python ../../plot/plotk.py ${op} l96 ${na} ${pert}
    #python ../../plot/plotdxa.py ${op} l96 ${na} ${pert}
    #python ../../plot/plotpf.py ${op} l96 ${na} ${pert}
    #python ../../plot/plotlpf.py ${op} l96 ${na} ${pert} 
    #done
  done
  python ../../plot/plote.py ${op} l96 ${na} etkf #mlef
  #python ../../plot/plotchi.py ${op} l96 ${na}
  #python ../../plot/plotinnv.py ${op} l96 ${na} > innv_${op}.log
  #python ../../plot/plotxa.py ${op} l96 ${na}
  #python ../../plot/plotdof.py ${op} l96 ${na}
  
  #rm obs*.npy
done
#rm l96*.txt 
#rm l96*.npy 
