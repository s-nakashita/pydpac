#!/bin/sh
# This is a run script for KdVB experiment
operators="quadratic"
perturbations="mlef" # etkf po srf letkf" # kf var"
na=100 # Number of assimilation cycle
nobs=101 # Number of observations
obsnet="all" # Observation network
ltlm="F" # "T":Use tangent linear approximation "F":Not use
nmem= # Ensemble size
ref=z05
exp="z05_mlef_iter1_test" #_${obsnet}"
echo ${exp}
rm -rf work/${exp}
mkdir -p work/${exp}
cd work/${exp}
cp ../../logging_config.ini .
#cp ../z05_mlef/truth.npy .
#cp ../z05_mlef/obsall_*_500.npy .
#rm -rf *.npy
#rm -rf *.log
#rm -rf timer
touch timer
for itest in $(seq 1 100); do
ln -s ../z05_mlef_test/truth_${itest}.npy ./truth.npy
for op in ${operators}; do
  ln -s ../z05_mlef_test/obs${obsnet}_${op}_500_${itest}.npy ./obs${obsnet}_${op}_500.npy
  for pt in ${perturbations}; do
    echo $pt
    args="${op} ${pt} ${na} ${nobs} ${obsnet} ${ltlm} ${nmem}"
    echo $args
    start_time=$(gdate +"%s.%5N")
    python ../../${ref}.py ${args} > ${ref}_${op}_${pt}.log 2>&1
    wait
    end_time=$(gdate +"%s.%5N")
    echo ${op} >> timer
    echo "scale=1; ${end_time}-${start_time}" | bc >> timer
    mv ${ref}_e_${op}_${pt}.txt ${ref}_e_${op}_${pt}_${itest}.txt
    #mv ${ref}_pa_${op}_${pt}.npy ${ref}_pa_${op}_${pert}.npy
    #if [ "${pert:4:1}" = "b" ]; then
    #mv ${ref}_rho_${op}_${pt}.npy ${ref}_rho_${op}_${pert}.npy
    #fi
    #for icycle in $(seq 0 4); do
    #  mv ${ref}_K_${op}_${pt}_cycle$icycle.npy ${ref}_K_${op}_${pert}_cycle$icycle.npy
    #  mv ${ref}_dxaorig_${op}_${pt}_cycle$icycle.npy ${ref}_dxaorig_${op}_${pert}_cycle$icycle.npy
    #  mv ${ref}_dxa_${op}_${pt}_cycle$icycle.npy ${ref}_dxa_${op}_${pert}_cycle$icycle.npy
    #  mv ${ref}_pa_${op}_${pt}_cycle$icycle.npy ${ref}_pa_${op}_${pert}_cycle$icycle.npy
    #  mv ${ref}_pf_${op}_${pt}_cycle$icycle.npy ${ref}_pf_${op}_${pert}_cycle$icycle.npy
    #  mv ${ref}_spf_${op}_${pt}_cycle$icycle.npy ${ref}_spf_${op}_${pert}_cycle$icycle.npy
    #  if [ "${pert:4:1}" = "b" ]; then
    #  mv ${ref}_lpf_${op}_${pt}_cycle$icycle.npy ${ref}_lpf_${op}_${pert}_cycle$icycle.npy
    #  mv ${ref}_lspf_${op}_${pt}_cycle$icycle.npy ${ref}_lspf_${op}_${pert}_cycle$icycle.npy
    #  fi
    #done
    #python ../../plot/plotk.py ${op} ${ref} ${na} ${pert}
    #python ../../plot/plotdxa.py ${op} ${ref} ${na} ${pert}
    #python ../../plot/plotua.py ${op} ${ref} ${na} ${obsnet} 
    #python ../../plot/plotpa.py ${op} ${ref} ${na} ${pt}
    #python ../../plot/plotpf.py ${op} ${ref} ${na} ${pert}
    #python ../../plot/plotlpf.py ${op} ${ref} ${na} ${pert} 
    #done
  done
  #python ../../plot/plote.py ${op} ${ref} ${na} ${datype}
  #python ../../plot/plotchi.py ${op} ${ref} ${na}
  #python ../../plot/plotinnv.py ${op} ${ref} ${na} > innv_${op}.log
  #python ../../plot/plotxa.py ${op} ${ref} ${na}
  #python ../../plot/plotdof.py ${op} ${ref} ${na}
  
  #rm obs*.npy
  #mv obs${obsnet}_${op}_500.npy obs${obsnet}_${op}_500_${itest}.npy
  rm obs${obsnet}_${op}_500.npy
done
#mv truth.npy truth_${itest}.npy
rm truth.npy
done
#rm ${ref}*.txt 
#rm ${ref}*.npy 