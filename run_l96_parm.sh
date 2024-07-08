#!/bin/sh
# This is a run script for parameter sensitivity experiment
export OMP_NUM_THREADS=4
alias python=python3
model=l96
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
operators="linear" # quadratic cubic"
perturbations="etkf"
datype="etkf"
na=1000 # Number of assimilation cycle
nmem=20
nobs=40
linf=True  # True:Apply inflation False:Not apply
lloc=False # True:Apply localization False:Not apply
ltlm=False # True:Use tangent linear approximation False:Not use
a_window=1
ptype=infl
iinf=3
exp="${datype}_infl${iinf}"
echo ${exp}
cdir=` pwd `
wdir=work/${model}/${exp}
rm -rf $wdir
mkdir -p $wdir
cd $wdir
cp ${cdir}/logging_config.ini .
rm -rf obs*.npy
rm -rf *.log
rm -rf timer
touch timer
#sig="1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0"
#inf="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9"
#Nobs="40 35 30 25 20 15 10"
#Nmem="40 35 30 25 20 15 10 5"
Nt="1 2 3 4 5 6 7 8"
sigb_list="0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8"
lb_list="-1.0 1.0 2.0 3.0 4.0 5.0"
if [ $iinf -lt 1 ]; then
infllist="1.0 1.05 1.1 1.15 1.2 1.25"
elif [ $iinf -eq 1 ]; then
infllist="0.0 0.05 0.1 0.15 0.2 0.25"
elif [ $iinf -lt 4 ]; then
infllist="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
else
infllist="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5"
fi
touch params.txt
for op in ${operators}; do
  echo $ptype > params.txt
  #for lsig in ${sig}; do
  for infl_parm in ${infllist}; do
    echo $infl_parm >> params.txt
    ptmp=$infl_parm
  #for nobs in ${Nobs}; do
  #for nmem in ${Nmem}; do
  #for a_window in ${Nt}; do
  #for sigb in ${sigb_list}; do
  #for lb in ${lb_list}; do
    for pert in ${perturbations}; do
      echo $pert
      cp ${cdir}/analysis/config/config_${pert}_sample.py config.py
      gsed -i -e "2i \ \"op\":\"${op}\"," config.py
      gsed -i -e "2i \ \"na\":${na}," config.py
      gsed -i -e "2i \ \"nobs\":${nobs}," config.py
      gsed -i -e "/nmem/s/40/${nmem}/" config.py
      if [ $linf = True ];then
        gsed -i -e '/linf/s/False/True/' config.py
        gsed -i -e "4i \ \"iinf\":${iinf}," config.py
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
      if [ $ptype = loc ]; then
        gsed -i -e "6i \ \"lsig\":${lsig}," config.py
      fi
      if [ $ptype = infl ]; then
        gsed -i -e "6i \ \"infl_parm\":${infl_parm}," config.py
      fi
      if [ $ptype = sigb ]; then
        gsed -i -e "6i \ \"sigb\":${sigb}," config.py
      fi
      if [ $ptype = lb ]; then
        gsed -i -e "6i \ \"lb\":${lb}," config.py
      fi
      cat config.py
      ptline=$(awk -F: '(NR>1 && $1~/pt/){print $2}' config.py)
      pt=${ptline#\"*}; pt=${pt%\"*}
      echo $pt
      for count in $(seq 1 10); do
        echo $count
        start_time=$(date +"%s")
        python ${cdir}/${model}.py > ${model}_${op}_${pert}.log 2>&1
        wait
        end_time=$(date +"%s")
        cputime=`echo "scale=3; ${end_time}-${start_time}" | bc`
        echo "${op} ${pert} ${count} ${cputime}" >> timer
        mv ${model}_e_${op}_${pt}.txt e_${op}_${pt}_${count}.txt
        mv ${model}_stda_${op}_${pt}.txt stda_${op}_${pt}_${count}.txt
        mv ${model}_xdmean_${op}_${pt}.txt xdmean_${op}_${pt}_${count}.txt
        mv ${model}_xsmean_${op}_${pt}.txt xsmean_${op}_${pt}_${count}.txt
        mv ${model}_pdr_${op}_${pt}.txt pdr_${op}_${pt}_${count}.txt
        rm obs*.npy
      done
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} e ${pt}
      rm e_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} stda ${pt}
      rm stda_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} xdmean ${pt}
      rm xdmean_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} xsmean ${pt}
      rm xsmean_${op}_${pt}_*.txt
      python ${cdir}/plot/calc_mean.py ${op} ${model} ${na} ${count} pdr ${pt}
      rm pdr_${op}_${pt}_*.txt
      mv ${model}_e_${op}_${pt}.txt ${model}_e_${op}_${pert}_${ptmp}.txt
      mv ${model}_stda_${op}_${pt}.txt ${model}_stda_${op}_${pert}_${ptmp}.txt
      mv ${model}_xdmean_${op}_${pt}.txt ${model}_xdmean_${op}_${pert}_${ptmp}.txt
      mv ${model}_xsmean_${op}_${pt}.txt ${model}_xsmean_${op}_${pert}_${ptmp}.txt
      mv ${model}_pdr_${op}_${pt}.txt ${model}_pdr_${op}_${pert}_${ptmp}.txt
    done #pert
  done #params
  cat params.txt
  python ${cdir}/plot/ploteparam.py ${op} ${model} ${na} ${ptype}
  python ${cdir}/plot/plotxdparam.py ${op} ${model} ${na} ${ptype}
  rm ${model}*cycle*.npy ${model}*cycle*.txt
  #rm obs*.npy
done
#rm ${model}*.txt 
#rm ${model}*.npy 