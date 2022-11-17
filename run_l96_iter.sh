#!/bin/sh
# This is a run script for Lorenz96 experiment
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic" # cubic"
perturbations="var 4dvar etkf 4detkf letkf 4dletkf \
mlef 4dmlef mlefbe 4dmlefbe mlefcw 4dmlefcw mlefy 4dmlefy \
mlef_incr 4dmlef_incr mlefbe_incr 4dmlefbe_incr \
mlefcw_incr 4dmlefcw_incr mlefy_incr 4dmlefy_incr"
#perturbations="mlef 4dmlef"
#perturbations="var 4dvar letkf 4dletkf mlefbe 4dmlefbe mlefcw 4dmlefcw mlefy 4dmlefy"
#perturbations="mlefbe 4dmlefbe mlefcw 4dmlefcw mlefy 4dmlefy"
#datype="mlef"
#perturbations="var 4dvar letkf 4dletkf ${datype}be ${datype}bm ${datype}cw ${datype}y"
#perturbations="lmlefcw lmlefy mlef"
#perturbations="mlef 4dmlef"
na=100 # Number of assimilation cycle
linf=True  # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
#L="-1.0 0.5 1.0 2.0"
exp="comp"
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
  for count in $(seq 1 2); do
    for pert in ${perturbations}; do
      echo $pert
      if [[ $pert =~ ^[0-9]*[a-z]+_incr ]];then
      cp ../../analysis/config/config_${pert%\_*}_sample.py config.py
      sed -i -e '/incremental/s/False/True/' config.py
      else
      cp ../../analysis/config/config_${pert}_sample.py config.py
      fi
      if [ $linf = True ];then
      sed -i -e '/linf/s/False/True/' config.py
      else
      sed -i -e '/linf/s/True/False/' config.py
      fi
      if [ $ltlm = True ];then
      sed -i -e '/ltlm/s/False/True/' config.py
      else
      sed -i -e '/ltlm/s/True/False/' config.py
      fi
      cat config.py
      ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
      pt=${ptline#\"*}; pt=${pt%\"*}
      echo $pt
      start_time=$(gdate +"%s.%5N")
      python ../../l96.py > l96_${op}_${pert}.log 2>&1
      wait
      end_time=$(gdate +"%s.%5N")
      ctime=`echo "scale=1; ${end_time}-${start_time}" | bc`
      echo "${op} ${pert} ${count} ${ctime}" >> timer
      echo $count
      mv l96_e_${op}_${pt}.txt e_${op}_${pert}_${count}.txt
      rm obs*.npy
    done
  done
  for pert in ${perturbations}; do
    python ../../plot/calc_mean.py $op l96 $na $count e $pert
  done
  python ../../plot/plotemean.py ${op} l96 ${na} mlef
  #python ../../plot/plotchi.py ${op} l96 ${na}
  #python ../../plot/plotinnv.py ${op} l96 ${na} > innv_${op}.log
  #python ../../plot/plotxa.py ${op} l96 ${na}
  #python ../../plot/plotdof.py ${op} l96 ${na}
  
  #rm obs*.npy
done
#rm l96*.txt 
#rm l96*.npy 