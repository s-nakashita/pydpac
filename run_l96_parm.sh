#!/bin/sh
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear quadratic cubic"
#perturbations="mlef grad etkf po srf letkf" # kf var var4d"
perturbations="mlef" # grad etkf"
na=300 # Number of assimilation cycle
linf="T" # "T":Apply inflation "F":Not apply
lloc="T" # "T":Apply localization "F":Not apply
ltlm="F" # "T":Use tangent linear approximation "F":Not use
a_window=5
exp="mlef-infl"
echo ${exp}
rm -rf ${exp}
mkdir ${exp}
cd ${exp}
#sig="1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0"
inf="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9"
cp ../logging_config.ini .
for op in ${operators}; do
  #if [ ${linf} = "T" ]; then
  #  if [ ${op} = "linear" ]; then
  #    infl_parm=1.1
  #  elif [ ${op} = "quadratic" ]; then
  #    infl_parm=1.3
  #  elif [ ${op} = "cubic" ]; then
  #    infl_parm=1.6
  #  fi
  #else
  #  infl_parm=-1.0
  #fi
  for pt in ${perturbations}; do
    #for lsig in ${sig}; do
    for infl_parm in ${inf}; do
      #echo ${op} ${pt} ${na} ${infl_parm} ${lsig} ${ltlm} ${a_window}
      echo ${op} ${pt} ${na} ${infl_parm} ${lloc} ${ltlm} ${a_window}
      for count in $(seq 1 10); do
        python ../l96.py ${op} ${pt} ${na} ${infl_parm} ${lloc} ${ltlm} ${a_window} > l96_${op}_${pt}.log 2>&1
        wait
        echo ${count}
        cp l96_e_${op}_${pt}.txt e_${op}_${pt}_${count}.txt
        #cp l96_chi_${op}_${pt}.txt chi_${op}_${pt}_${count}.txt
        #cp l96_dof_${op}_${pt}.txt dof_${op}_${pt}_${count}.txt
        #cp l96_ua_${op}_${pt}.npy ua_${op}_${pt}_${count}.npy
        #cp l96_innv_${op}_${pt}.npy innv_${op}_${pt}_${count}.npy
        rm obs*.npy
      done
      python ../calc_mean.py ${op} l96 ${na} ${count} e ${pt}
      rm e_${op}_${pt}_*.txt
      #cp l96_e_${op}_${pt}.txt l96_e_${op}_${pt}_${lsig}.txt
      cp l96_e_${op}_${pt}.txt l96_e_${op}_${pt}_${infl_parm}.txt
      #for vname in e chi dof; do
      #  python ../calc_mean.py ${op} l96 ${na} ${count} ${vname} ${pt}
      #  rm ${vname}_${op}_${pt}_*.txt
      #done
      #for vname in ua innv; do
      #  python ../calc_mean.py ${op} l96 ${na} ${count} ${vname} ${pt}
      #  rm ${vname}_${op}_${pt}_*.npy
      #done
    done
    #python ../ploteparam.py ${op} l96 ${na} loc
    #python ../ploteparam.py ${op} l96 ${na} infl
  done
  #python ../plote.py ${op} l96 ${na}
  #python ../plotchi.py ${op} l96 ${na}
  #python ../plotinnv.py ${op} l96 ${na}
  #python ../plotxa.py ${op} l96 ${na}
  #python ../plotdof.py ${op} l96 ${na}
  #python ../ploteparam.py ${op} l96 ${na} loc
  python ../ploteparam.py ${op} l96 ${na} infl
  rm obs*.npy
done
#mv l96_e_${operators}.png l96_e_${operators}_${exp}.png
rm l96*.txt 
rm l96*.npy 