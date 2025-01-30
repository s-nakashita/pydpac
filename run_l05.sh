#!/bin/sh
# This is a run script for Lorenz05 experiment
export OMP_NUM_THREADS=4
#alias python=python3.9
model="l05IIm"
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="envar"
na=240 # Number of assimilation cycle
nmem=80 # ensemble size
nobs=30 # observation volume
linf=True # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
iinf=-2
model_error=True
#L="-1.0 0.5 1.0 2.0"
#lsig=120
functype=gc5
a=-0.2
#exp="var_${functype}a${a}_obs${nobs}"
exp="envar_infl_mem${nmem}obs${nobs}"
#exp="${datype}_loc_hint"
echo ${exp}
cdir=` pwd `
if [ $model_error = True ]; then
wdir=work/${model}_inperfect/${exp}
else
wdir=work/${model}/${exp}
fi
rm -rf $wdir
mkdir -p $wdir
cd $wdir
cp ${cdir}/logging_config.ini .
rm -rf *.npy
rm -rf *.log
rm -rf timer
touch timer
if [ $model_error = True ]; then
if [ $model = l05II ]; then
if [ ! -f ${cdir}/data/l05III/truth.npy ]; then
echo "Please create nature run first by executing model/lorenz3.py"
exit
fi
ln -fs ${cdir}/data/l05III/truth.npy .
elif [ $model = l05IIm ]; then
if [ ! -f ${cdir}/data/l05IIIm/truth.npy ]; then
echo "Please create nature run first by executing model/lorenz3m.py"
exit
fi
ln -fs ${cdir}/data/l05IIIm/truth.npy .
fi
else
if [ ! -f ${cdir}/data/${model}/truth.npy ]; then
echo "Please create nature run first"
exit
fi
ln -fs ${cdir}/data/${model}/truth.npy .
fi
rseed=514
roseed=515
for op in ${operators}; do
  for pert0 in ${perturbations}; do
    echo $pert0
    pert=${pert0}
    for iinf in $(seq -3 5);do
    pert=${pert0}_${iinf}
    cp ${cdir}/analysis/config/config_${pert0}_sample.py config.py
    gsed -i -e "2i \ \"op\":\"${op}\"," config.py
    gsed -i -e "2i \ \"na\":${na}," config.py
    gsed -i -e "2i \ \"nobs\":${nobs}," config.py
    gsed -i -e "/nmem/s/40/${nmem}/" config.py
    gsed -i -e "2i \ \"rseed\":${rseed}," config.py
    gsed -i -e "2i \ \"roseed\":${roseed}," config.py
    if [ $linf = True ];then
      gsed -i -e '/linf/s/False/True/' config.py
      gsed -i -e "4i \ \"iinf\":${iinf}," config.py
      if [ $iinf -lt 1 ]; then
        gsed -i -e "5i \ \"infl_parm\":1.1," config.py
      elif [ $iinf -eq 1 ]; then
        gsed -i -e "5i \ \"infl_parm\":0.2," config.py
      elif [ $iinf -lt 4 ]; then
        gsed -i -e "5i \ \"infl_parm\":0.3," config.py
      else
        gsed -i -e "5i \ \"infl_parm\":0.3," config.py
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
    if [ ! -z $lsig ]; then
      gsed -i -e "6i \ \"lsig\":${lsig}," config.py
    fi
    #gsed -i -e "6i \ \"functype\":\"${functype}\"," config.py
    #gsed -i -e "6i \ \"a\":${a}," config.py
    gsed -i -e "6i \ \"model_error\":${model_error}," config.py
    cat config.py
    ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
    pt=${ptline#\"*}; pt=${pt%\"*}
    echo $pt
    start_time=$(date +"%s")
    python ${cdir}/l05.py ${model} > ${model}_${op}_${pert}.log 2>&1
    wait
    end_time=$(date +"%s")
    echo "${op} ${pert}" >> timer
    echo "scale=3; ${end_time}-${start_time}" | bc >> timer
    mv ${model}_e_${op}_${pt}.txt e_${op}_${pert}.txt
    mv ${model}_stda_${op}_${pt}.txt stda_${op}_${pert}.txt
    mv ${model}_xdmean_${op}_${pt}.txt xdmean_${op}_${pert}.txt
    mv ${model}_xsmean_${op}_${pt}.txt xsmean_${op}_${pert}.txt
    mv ${model}_ef_${op}_${pt}.txt ef_${op}_${pert}.txt
    mv ${model}_stdf_${op}_${pt}.txt stdf_${op}_${pert}.txt
    mv ${model}_xdfmean_${op}_${pt}.txt xdfmean_${op}_${pert}.txt
    mv ${model}_xsfmean_${op}_${pt}.txt xsfmean_${op}_${pert}.txt
    if [ $iinf -le -2 ]; then
      mv ${model}_infl_${op}_${pt}.txt infl_${op}_${pert}.txt
    fi
    mv ${model}_pdr_${op}_${pt}.txt pdr_${op}_${pert}.txt

    loctype=`echo $pert | cut -c5-5`
    if [ "${loctype}" = "b" ]; then
    mv ${model}_rho_${op}_${pt}.npy ${model}_rho_${op}_${pert}.npy
    fi
    #for icycle in $(seq 0 $((${na} - 1))); do
    #  if test -e wa_${op}_${pt}_cycle${icycle}.npy; then
    #    mv wa_${op}_${pt}_cycle${icycle}.npy ${pert}/wa_${op}_cycle${icycle}.npy
    #  fi
    #  if test -e ${model}_ua_${op}_${pt}_cycle${icycle}.npy; then
    #    mv ${model}_ua_${op}_${pt}_cycle${icycle}.npy ${pert}/ua_${op}_${pert}_cycle${icycle}.npy
    #  fi
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
    #done
    #python ${cdir}/plot/plotk.py ${op} ${model} ${na} ${pert}
    #python ${cdir}/plot/plotdxa.py ${op} ${model} ${na} ${pert}
    #python ${cdir}/plot/plotpf.py ${op} ${model} ${na} ${pert}
    #python ${cdir}/plot/plotlpf.py ${op} ${model} ${na} ${pert} 
    mkdir -p data/${pert}
    mv ${model}_*_cycle*.npy data/${pert}
    done #iinf
    python ${cdir}/plot/plote.py ${op} ${model} ${na} ${pert0} infl
    python ${cdir}/plot/plotxd.py ${op} ${model} ${na} ${pert0} infl
    python ${cdir}/plot/plotpdr.py ${op} ${model} ${na} ${pert0} infl
    python ${cdir}/plot/plotinfl.py ${op} ${model} ${na} ${pert0} infl
  done #pert
#  python ${cdir}/plot/plote.py ${op} ${model} ${na} #mlef
#  python ${cdir}/plot/plotxd.py ${op} ${model} ${na} #mlef
  #python ${cdir}/plot/plotchi.py ${op} ${model} ${na}
  #python ${cdir}/plot/plotinnv.py ${op} ${model} ${na} > innv_${op}.log
#  python ${cdir}/plot/plotxa.py ${op} ${model} ${na} ${model_error}
  #python ${cdir}/plot/plotdof.py ${op} ${model} ${na}
#  python ${cdir}/plot/ploterrspectra.py ${op} ${model} ${na} ${model_error}
#  if [ ${na} -gt 1000 ]; then python ${cdir}/plot/nmc.py ${op} ${model} ${na}; fi
  python ${cdir}/plot/plotjh+gh.py ${op} ${model} ${na}
  rm ${model}_jh_${op}_*_cycle*.txt ${model}_gh_${op}_*_cycle*.txt ${model}_alpha_${op}_*_cycle*.txt
#  python ${cdir}/plot/plotinfl.py ${op} ${model} ${na}
  
  #rm obs*.npy
done
#rm ${model}*.txt 
#rm ${model}_*_cycle*.npy 
