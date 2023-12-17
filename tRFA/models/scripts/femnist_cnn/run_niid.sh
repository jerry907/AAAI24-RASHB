#! /bin/bash

export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

source scripts/parallelize.sh
set -u  # x: stack trace
#export CUDA_VISIBLE_DEVICES=""
cmds=""
echo "Starting at $(date)"

njobs=7
num_rounds=1500  # 200 = 1 epochs
num_epochs=1
clients_per_round=35
dataset="femnist"
model="erm_cnn_log_reg"
lr=0.1
reg=0.0
eval_every=50

m0=0
momcof=0.99 #momentum coefficient
minibatch=0.005 # 750*0.01=~7
norm_bound="0.215771"  # Norm 90th percentile; pre-computed
niter=1 # GM
dev_type="sign" # pga attack
alpha=0.1 # niid sample setting

declare -A aggArray=( [FedAvg]=mean [MEBRA]=minimum_enclosing_ball_with_outliers [MEBRA1]=minimum_enclosing_ball_with_outliers1 [MEBRA_C]=minimum_enclosing_ball_with_outliers [MEBRA1_C]=minimum_enclosing_ball_with_outliers1 [GM]=geom_median [CWM]=coord_median [MultiKrum]=krum [Norm]=norm_bounded_mean [CWTM]=trimmed_mean)

outf="outputs/femnist_cnn_niid_${alpha}/outputs"
logf="outputs/femnist_cnn_niid_${alpha}/log"

for frac in 0.1 0.2 0.4
do

for seed in 1 2 3 4 5
do

for corruption in "labelflip" "gauss" "gauss2" "omniscient" "empire" "signflip" "little" "pga"
do

for aggregation_out in "MultiKrum" "FedAvg" "GM" "Norm" "CWTM" "CWM" "MEBRA" "MEBRA1" "MEBRA_C" "MEBRA1_C" 
do

aggregation=${aggArray[${aggregation_out}]}
main_args=" -dataset ${dataset} -model ${model}  -lr $lr -momcof ${momcof} --minibatch ${minibatch} --train_data train_niid_${alpha} --test_data test_niid_${alpha}"
options=" --num-rounds ${num_rounds} --eval-every ${eval_every} --clients-per-round ${clients_per_round} --num_epochs ${num_epochs} -reg_param $reg --m0 ${m0} --gpu 0"
options="${options} --aggregation $aggregation --norm_bound ${norm_bound} --weiszfeld-maxiter ${niter} --dev_type ${dev_type}"
options="${options} --seed ${seed} --corruption ${corruption} --fraction-corrupt ${frac} --fraction_to_discard $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${seed} "
if [ "${aggregation_out}" = "MEBRA_C" ] || [ "${aggregation_out}" = "MEBRA1_C" ]; then
options="${options} --mebwo_alg cluster"
fi

cmds="$cmds ; time python -u main.py ${main_args} ${options} > ${logf}_${aggregation_out}_${corruption}_${frac}_${seed} 2>&1 "
done  # aggregation_out
done  # corruption
done  # frac
done  # seed

echo "executing..."
set +u # for parallel exec to work (unbound variables)
f_ParallelExec $njobs "$cmds"

echo "Done at $(date)"
