#!/bin/sh
# This is a run script for JAX Lorenz experiment
export OMP_NUM_THREADS=4
ltype=${1:-1}
case $ltype in
  1) model="l05I";;
  2) model="l05II";;
  3) model="l05III";;
esac
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="var 4dvar letkf 4dletkf mlefy 4dmlefy"
#datype="4dmlef"
#perturbations="4dvar 4dletkf ${datype}be ${datype}bm ${datype}cw ${datype}y"
#perturbations="lmlefcw lmlefy mlef"
#perturbations="mlef 4dmlef mlefbe"
perturbations="letkf"
na=1460 # Number of assimilation cycle
nmem=8  # ensemble size
nobs=40 # observation volume
linf=True  # True:Apply inflation False:Not apply
lloc=True # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
#L="-1.0 0.5 1.0 2.0"
exp="extfcst_m${nmem}"
#exp="${datype}_loc_hint"
echo ${exp}
cdir=` pwd `
#wdir=work/${model}/${exp}
#ddir=/Volumes/FF520/pyesa/data/${model}/extfcst_m8
wdir=/Volumes/FF520/pyesa/data/${model}/${exp}
rm -rf $wdir
mkdir -p $wdir
cd $wdir
cp ${cdir}/logging_config.ini .
rm -rf *.npy
rm -rf *.log
rm -rf timer
touch timer
#ln -s $ddir/truth.npy .
#ln -s $ddir/obs*.npy .
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
    gsed -i -e "6i \ \"extfcst\":True," config.py
    cat config.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    start_time=$(gdate +"%s.%5N")
    python ${cdir}/l05j.py ${ltype} > ${model}_${op}_${pert}.log 2>&1
    python ${cdir}/l05j_fcst.py > ${model}_fcst_${op}_${pert}.log 2>&1
    wait
    end_time=$(gdate +"%s.%5N")
    echo "${op} ${pert}" >> timer
    echo "scale=1; ${end_time}-${start_time}" | bc >> timer
    mv ${model}_e_${op}_${pt}.txt e_${op}_${pert}.txt
    mv ${model}_stda_${op}_${pt}.txt stda_${op}_${pert}.txt
    mv ${model}_xa_${op}_${pt}.npy ${model}_xa_${op}_${pert}.npy
    mv ${model}_xf_${op}_${pt}.npy ${model}_xf_${op}_${pert}.npy
    for ft in 00 24 48 72 96 120; do
      mv ${model}_xf${ft}_${op}_${pt}.npy ${model}_xf${ft}_${op}_${pert}.npy
    done
    #if [ "${pert:4:1}" = "b" ]; then
    #mv ${model}_rho_${op}_${pt}.npy ${model}_rho_${op}_${pert}.npy
    #fi
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
    #python ${cdir}/plot/plotk.py ${op} l96 ${na} ${pert}
    #python ${cdir}/plot/plotdxa.py ${op} l96 ${na} ${pert}
    #python ${cdir}/plot/plotpf.py ${op} l96 ${na} ${pert}
    #python ${cdir}/plot/plotlpf.py ${op} l96 ${na} ${pert} 
    #done
  done
  python ${cdir}/plot/plote.py ${op} ${model} ${na} #mlef
  #python ${cdir}/plot/plotchi.py ${op} l96 ${na}
  #python ${cdir}/plot/plotinnv.py ${op} l96 ${na} > innv_${op}.log
  python ${cdir}/plot/plotxa.py ${op} ${model} ${na}
  #python ${cdir}/plot/nmc.py ${op} l96 ${na}
  #python ${cdir}/plot/plotdof.py ${op} l96 ${na}
  
  #rm obs*.npy
done
#rm l96*.txt 
#rm l96*.npy 
