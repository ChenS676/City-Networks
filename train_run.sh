#!/bin/bash

run_models(){
    local DATASET=$1
    local model=$2
    local device_idx=$3 # GPU idx
    local start_core=$4 # Starting CPU/GPU index
    local k=$5          # Number of CPU cores to use for each model
    # Experiment configs
    local layers=(2)
    local batch_size=200000
    local n_hidden=64
    local lr=5e-4
    local epochs=30000
    local window=100
    local exp_name='May08'
    # W&B configs
    local wandb_project='gnn-benchmark'
    local wandb_entity=''          # set to your W&B team/username, or leave blank for default

    # Loop through #layer and execute with pre-specified CPU/GPU
    for (( idx=0; idx<${#layers[@]}; idx++ )); do
        local end_core=$((start_core + k - 1))
        local num_layers=${layers[idx]}

        echo "Data: $DATASET, Model: $model, Num_layers: $num_layers"
        # The results will save to ./results/exp_name
        echo "Running on GPU $device_idx with CPU cores $start_core-$end_core"
        taskset -c $start_core-$end_core python train.py \
            --device $device_idx \
            --method $model \
            --dataset $DATASET \
            --runs 3 \
            --seed 0 \
            --save_model \
            --experiment_name $exp_name \
            --num_layers $num_layers \
            --hidden_channels $n_hidden \
            --lr $lr \
            --batch_size $batch_size \
            --epochs $epochs \
            --weight_decay 1e-5 \
            --dropout 0.2 \
            --display_step $window \
            --num_transformer_heads 4 \
            --transformer_weight_decay 1e-2 \
            --transformer_dropout 0.2 \
            --num_hops 2 \
            --local_gnn_type GCN \
            --concat_heads \
            --wandb_project $wandb_project \
            ${wandb_entity:+--wandb_entity $wandb_entity}
    done
}

# Uncomment to execute in parallel
DATASETS=(
    #"london"
    "la"
    # "paris"
    # "shanghai"
)

# Given a machine with 4 GPUs and 40 cores,
# the tasks can be executed in parallel
for data in ${DATASETS[@]}; do
    # run_models $data gcn    0 10 10 &
    # run_models $data cheb    0 0 10 &
    # run_models $data sage    0 0 10 &

    # GPU 1 — cores 20-39, partitioned evenly (5 each, no overlap)
    # run_models $data sgformer    1 20  5 &
    run_models $data nagphormer  1 0  5 &
    run_models $data graphgps    1 6  5 &
    run_models $data gt          1 11  5 &
done

