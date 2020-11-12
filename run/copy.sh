#!/bin/sh
if [ $# -lt 3 ]; then
    echo "Usage :: $0 model exp operator"
    exit
fi
model=${1} #z08,l96
#operators="linear quadratic cubic quadratic-nodiff cubic-nodiff"
#if [ $model=z08 ]; then
perturbations="mlef grad etkf-jh etkf-fh"
#fi
#if [ $model="l96" ]; then
#    perturbations="mlef grad etkf"
#fi
exp=${2}
op=${3}
# output file name example
# z08_uf_linear_mlef.npy z08_ua_linear_mlef.npy z08_pa_linear_mlef.npy
# z08_dh_linear_mlef_cycle0.npy z08_cJ_linear_mlef_cycle0.npy
# z08_e_linear_mlef.txt(z08_e_linear_mlef_lag8.txt) z08_chi_linear_mlef.txt 
# z08_jh_linear_mlef_cycle0.txt z08_gh_linear_mlef_cycle0.txt
npy_header="MOD_uf MOD_ua"
npy_headerc="MOD_dh MOD_d MOD_cJ MOD_pa MOD_ua MOD_K MOD_Kloc MOD_dx"
txt_header="MOD_e MOD_chi MOD_dpa MOD_ndpa"
txt_headerc="MOD_jh MOD_gh"
for pt in ${perturbations};do
for f in ${npy_header};do
    f=${f/MOD/$model}
    inf=${f}_${op}_${pt}.npy
    cp ${inf%.npy}_${exp}.npy $inf
done
for f in ${npy_headerc};do
    for i in $(seq 0 9);do
        f=${f/MOD/$model}
        inf=${f}_${op}_${pt}_cycle${i}.npy
        cp ${inf%.npy}_${exp}.npy $inf
    done
done
for f in ${txt_header};do
    f=${f/MOD/$model}
    inf=${f}_${op}_${pt}.txt
    cp ${inf%.txt}_${exp}.txt $inf
done
for f in ${txt_headerc};do
    for i in $(seq 0 9);do
        f=${f/MOD/$model}
        inf=${f}_${op}_${pt}_cycle${i}.txt
        cp ${inf%.txt}_${exp}.txt $inf
    done
done
done

#ls -ltr | tail -n 9