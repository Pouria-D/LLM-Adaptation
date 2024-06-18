#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=
#SBATCH --account=
#SBATCH --qos=
#SBATCH --mem=10G
#SBATCH -c 10
#SBATCH --array=1-500



eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate pt3

n=$SLURM_ARRAY_TASK_ID
iteration=`sed -n "${n} p" params_lr_grad_epoch.csv`      # Get n-th line (1-indexed) of the file
echo "parameters for iteration: ${iteration}"

lr=$(echo ${iteration} | cut -d "," -f 1)
grad=$(echo ${iteration} | cut -d "," -f 2)
epochs=$(echo ${iteration} | cut -d "," -f 3)
pre_seq_len=$(echo ${iteration} | cut -d "," -f 4)
dataset=$(echo ${iteration} | cut -d "," -f 5)
method=$(echo ${iteration} | cut -d "," -f 6)
epsilon=$(echo ${iteration} | cut -d "," -f 7)
batch_size=$(echo ${iteration} | cut -d "," -f 8)
steps=$(echo ${iteration} | cut -d "," -f 9)
model=$(echo ${iteration} | cut -d "," -f 10)
seq_len=$(echo ${iteration} | cut -d "," -f 11)

echo lr
echo $lr

echo grad
echo $grad

echo epochs
echo $epochs

echo pre_seq_len
echo $pre_seq_len

echo dataset
echo $dataset

echo method
echo $method

echo epsilon
echo $epsilon

echo batch_size
echo $batch_size

echo steps
echo $steps

echo model
echo $model

echo seq_len
echo ${seq_len}

dataset=${dataset}
training_type=private
batch_size=${batch_size}
gradient_accumulation_steps=${steps}
learning_rate=${lr}
method_type=${method}
epochs=${epochs}
max_grad_norm=${grad}
privacy_engine=private_transformers
pre_seq_len=${pre_seq_len}

if [[ "${privacy_engine}" == "dp_transformers" ]]; then
    echo "Private training type with dp_transformers and do not remove unused columns."
    remove_unused_columns=False
else
    echo "Remove unused columns."
    remove_unused_columns=True
fi


echo "hostname: ${HOSTNAME}"
echo batch_size: ${batch_size}
echo gradient_accumulation_steps: ${gradient_accumulation_steps}
echo dataset: ${dataset}
echo epochs: ${epochs}
echo learning_rate: ${learning_rate}
echo method_type: ${method_type}
echo remove_unused_columns: ${remove_unused_columns}
echo training_type: ${training_type}
echo max_grad_norm: ${max_grad_norm}
echo privacy_engine: ${privacy_engine}
echo pre_seq_len: ${pre_seq_len}
echo seq_len: ${seq_len}

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
echo timestamp: ${timestamp}
start_time=$(date +%s)

python3.8 run.py \
--method_type ${method_type} \
--model_name_or_path ${model} \
--task_name glue \
--dataset_name ${dataset} \
--do_train \
--do_eval \
--max_seq_length ${seq_len} \
--per_device_train_batch_size ${batch_size} \
--learning_rate ${learning_rate} \
--weight_decay 0 \
--num_train_epochs ${epochs} \
--pre_seq_len ${pre_seq_len} \
--output_dir checkpoints/${dataset}-roberta-large-${method_type}-${training_type}/ \
--overwrite_output_dir \
--hidden_dropout_prob 0.1 \
--seed 11 \
--save_strategy epoch \
--evaluation_strategy epoch \
--training_type ${training_type} \
--target_epsilon ${epsilon} \
--per_sample_max_grad_norm ${max_grad_norm} \
--remove_unused_columns ${remove_unused_columns} \
--label_name labels \
--label_names labels \
--privacy_engine ${privacy_engine} \
--gradient_accumulation_steps ${gradient_accumulation_steps}


end_time=$(date +%s)
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
echo timestamp: ${timestamp}

# elapsed time with second resolution
elapsed=$(( end_time - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

echo "hostname: ${HOSTNAME}"
echo batch_size: ${batch_size}
echo gradient_accumulation_steps: ${gradient_accumulation_steps}
echo dataset: ${dataset}
echo epochs: ${epochs}
echo learning_rate: ${learning_rate}
echo method_type: ${method_type}
echo remove_unused_columns: ${remove_unused_columns}
echo training_type: ${training_type}
echo max_grad_norm: ${max_grad_norm}
echo privacy_engine: ${privacy_engine}
echo pre_seq_len: ${pre_seq_len}

