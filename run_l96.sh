#!/bin/sh
# This is a run script for Lorenz96 experiment
alias python=python3
model="l96"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
#perturbations="var 4dvar letkf 4dletkf mlefy 4dmlefy"
datype="envar"
#perturbations="4dvar 4dletkf ${datype}be ${datype}bm ${datype}cw ${datype}y"
#perturbations="lmlefcw lmlefy mlef"
#perturbations="mlef 4dmlef mlefbe"
perturbations="envar"
na=200 # Number of assimilation cycle
nmem=20 # ensemble size
nobs=40 # observation volume
linf=True  # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
#L="-1.0 0.5 1.0 2.0"
iinf=-3
exp="${datype}_infl"
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
for op in ${operators}; do
  for pert in ${perturbations}; do
    echo $pert
    for iinf in $(seq -3 5); do
    #for iinf in $(seq 4 5); do
      cp ${cdir}/analysis/config/config_${pert}_sample.py config.py
      gsed -i -e "2i \ \"op\":\"${op}\"," config.py
      gsed -i -e "2i \ \"na\":${na}," config.py
      gsed -i -e "2i \ \"nobs\":${nobs}," config.py
      gsed -i -e "/nmem/s/40/${nmem}/" config.py
      if [ $linf = True ];then
      gsed -i -e '/linf/s/False/True/' config.py
      gsed -i -e "4i \ \"iinf\":${iinf}," config.py
      if [ $iinf -lt 1 ]; then
        gsed -i -e "5i \ \"infl_parm\":1.05," config.py
      elif [ $iinf -eq 1 ]; then
        gsed -i -e "5i \ \"infl_parm\":0.2," config.py
      elif [ $iinf -lt 4 ]; then
        gsed -i -e "5i \ \"infl_parm\":0.8," config.py
      else
        gsed -i -e "5i \ \"infl_parm\":0.6," config.py
      fi
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
      python ${cdir}/l96.py > l96_${op}_${pert}_${iinf}.log 2>&1
      #python ${cdir}/l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${a_window} ${iloc} > l96_${op}_${pert}.log 2>&1
      #python ${cdir}/l96.py ${op} ${pt} ${na} ${linf} ${lloc} ${ltlm} ${lb} > l96_${op}_${pt}_${lb}.log 2>&1
      wait
      end_time=$(gdate +"%s.%5N")
      echo "${pert} ${iinf}" >> timer
      echo "scale=1; ${end_time}-${start_time}" | bc >> timer
      mv l96_e_${op}_${pt}.txt e_${op}_${pert}_${iinf}.txt
      mv l96_stda_${op}_${pt}.txt stda_${op}_${pert}_${iinf}.txt
      mv l96_ef_${op}_${pt}.txt ef_${op}_${pert}_${iinf}.txt
      mv l96_stdf_${op}_${pt}.txt stdf_${op}_${pert}_${iinf}.txt
      mv l96_xdmean_${op}_${pt}.txt xdmean_${op}_${pert}_${iinf}.txt
      mv l96_xsmean_${op}_${pt}.txt xsmean_${op}_${pert}_${iinf}.txt
      mv l96_xdfmean_${op}_${pt}.txt xdfmean_${op}_${pert}_${iinf}.txt
      mv l96_xsfmean_${op}_${pt}.txt xsfmean_${op}_${pert}_${iinf}.txt
      mv l96_xf_${op}_${pt}.npy xf_${op}_${pert}_${iinf}.npy
      mv l96_xa_${op}_${pt}.npy xa_${op}_${pert}_${iinf}.npy
      if [ $iinf -le -2 ]; then
        mv l96_infl_${op}_${pt}.txt infl_${op}_${pert}.txt
      fi
      mv l96_pdr_${op}_${pt}.txt pdr_${op}_${pert}_${iinf}.txt
      #if [ "${pert:4:1}" = "b" ]; then
      #mv l96_rho_${op}_${pt}.npy l96_rho_${op}_${pert}.npy
      #fi
      #for icycle in $(seq 0 $((${na} - 1))); do
      #  if test -e wa_${op}_${pt}_cycle${icycle}.npy; then
      #    mv wa_${op}_${pt}_cycle${icycle}.npy ${pert}/wa_${op}_cycle${icycle}.npy
      #  fi
      #  if test -e l96_ua_${op}_${pt}_cycle${icycle}.npy; then
      #    mv l96_ua_${op}_${pt}_cycle${icycle}.npy ${pert}/ua_${op}_${pert}_cycle${icycle}.npy
      #  fi
      ##  mv Wmat_${op}_${pt}_cycle${icycle}.npy ${pert}/Wmat_${op}_cycle${icycle}.npy
      ##  mv l96_K_${op}_${pt}_cycle$icycle.npy l96_K_${op}_${pert}_cycle$icycle.npy
      ##  mv l96_dxaorig_${op}_${pt}_cycle$icycle.npy l96_dxaorig_${op}_${pert}_cycle$icycle.npy
      ##  mv l96_dxa_${op}_${pt}_cycle$icycle.npy l96_dxa_${op}_${pert}_cycle$icycle.npy
      ##  mv l96_pa_${op}_${pt}_cycle$icycle.npy l96_pa_${op}_${pert}_cycle$icycle.npy
      ##  mv l96_pf_${op}_${pt}_cycle$icycle.npy l96_pf_${op}_${pert}_cycle$icycle.npy
      ##  mv l96_spf_${op}_${pt}_cycle$icycle.npy l96_spf_${op}_${pert}_cycle$icycle.npy
      ##  if [ "${pert:4:1}" = "b" ]; then
      ##  mv l96_lpf_${op}_${pt}_cycle$icycle.npy l96_lpf_${op}_${pert}_cycle$icycle.npy
      ##  mv l96_lspf_${op}_${pt}_cycle$icycle.npy l96_lspf_${op}_${pert}_cycle$icycle.npy
      ##  fi
      #done
      #python ${cdir}/plot/plotk.py ${op} l96 ${na} ${pert}
      #python ${cdir}/plot/plotdxa.py ${op} l96 ${na} ${pert}
      #python ${cdir}/plot/plotpf.py ${op} l96 ${na} ${pert}
      #python ${cdir}/plot/plotlpf.py ${op} l96 ${na} ${pert} 
      mkdir -p data/${pert}_${iinf}
      mv l96_*_${op}_${pt}_cycle*.npy data/${pert}_${iinf}
    done
    python ${cdir}/plot/plote.py ${op} l96 ${na} ${pert} infl
    python ${cdir}/plot/plotxd.py ${op} l96 ${na} ${pert} infl
    python ${cdir}/plot/plotpdr.py ${op} l96 ${na} ${pert} infl
  done
  #python ${cdir}/plot/plote.py ${op} l96 ${na}
  #python ${cdir}/plot/plotxd.py ${op} l96 ${na}
  #python ${cdir}/plot/plotchi.py ${op} l96 ${na}
  #python ${cdir}/plot/plotinnv.py ${op} l96 ${na} > innv_${op}.log
  #python ${cdir}/plot/plotxa.py ${op} l96 ${na}
  #python ${cdir}/plot/nmc.py ${op} l96 ${na}
  #python ${cdir}/plot/plotdof.py ${op} l96 ${na}
  python ${cdir}/plot/plotinfl.py ${op} l96 ${na}
  
  #rm obs*.npy
done
#rm l96*.txt 
#rm l96*.npy 
