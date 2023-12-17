#! /bin/bash

export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

source scripts/parallelize.sh
set -u  # x: stack trace
#export CUDA_VISIBLE_DEVICES=""

cmds=""

echo "Starting at $(date)"

njobs=6
num_rounds=1500  # 200 = 1 epochs
num_epochs=1
clients_per_round=35
lr=0.1
reg=0.0
eval_every=50


m0=0
momcof=0.99 # momentum coefficient
minibatch=0.005 # 750*0.005=3~4

norm_bound="0.215771"  # Norm 90th percentile; pre-computed
niter=1 # GM

dataset="femnist"
model="erm_cnn_log_reg"


declare -A aggArray=( [FedAvg]=mean [MEBRA1]=minimum_enclosing_ball_with_outliers1 [MEBRA1_C]=minimum_enclosing_ball_with_outliers1 [MEBRA]=minimum_enclosing_ball_with_outliers [MEBRA_C]=minimum_enclosing_ball_with_outliers [GM]=geom_median [CWM]=coord_median [MultiKrum]=krum [Norm]=norm_bounded_mean [CWTM]=trimmed_mean)

outf="outputs/femnist_cnn/outputs"
logf="outputs/femnist_cnn/log"


for seed in 1 2 3 4 5
do

for frac in 0.1 0.2 0.4
do

for corruption in "labelflip" "gauss" "gauss2" "omniscient" "empire" "signflip" "little"
do

for aggregation_out in "MultiKrum" "FedAvg" "GM" "Norm" "CWTM" "CWM" "MEBRA" "MEBRA1" "MEBRA_C" "MEBRA1_C"
do

aggregation=${aggArray[${aggregation_out}]}
main_args=" -dataset ${dataset} -model ${model}  -lr $lr -momcof ${momcof} --minibatch ${minibatch}"
options=" --num-rounds ${num_rounds} --eval-every ${eval_every} --clients-per-round ${clients_per_round} --num_epochs ${num_epochs} -reg_param $reg --m0 ${m0} --gpu 0"
options="${options} --aggregation $aggregation --norm_bound ${norm_bound} --weiszfeld-maxiter ${niter}"
options="${options} --seed ${seed} --corruption ${corruption} --fraction-corrupt ${frac} --fraction_to_discard $frac --output_summary_file ${outf}_${aggregation_out}_${corruption}_${frac}_${seed}"
if [ "${aggregation_out}" = "MEBRA_C" ] || [ "${aggregation_out}" = "MEBRA1_C" ]; then
options="${options} --mebwo_alg cluster"
fi

cmds="$cmds ; time python -u main.py ${main_args} ${options} > ${logf}_${aggregation_out}_${corruption}_${frac}_${seed} 2>&1 "
done
done
done  # aggregation
done  # seed




echo "executing..."
set +u # for parallel exec to work (unbound variables)
f_ParallelExec $njobs "$cmds"

echo "Done at $(date)"
