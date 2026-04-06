#!/bin/bash

run_models(){
    local DATASET=$1
    local model=$2
    local device_idx=$3
    local start_core=$4
    local k=$5

    local layers=(2)
    local batch_size=512
    local n_hidden=64
    local lr=5e-4
    local epochs=30000
    local window=100
    local exp_name='May08'

    local wandb_project='gnn-benchmark'
    local wandb_entity=''

    for (( idx=0; idx<${#layers[@]}; idx++ )); do
        local end_core=$((start_core + k - 1))
        local num_layers=${layers[idx]}

        echo "Data: $DATASET, Model: $model, Num_layers: $num_layers"
        echo "Running on GPU $device_idx with CPU cores $start_core-$end_core"

        CUDA_LAUNCH_BLOCKING=1 \
        # taskset -c $start_core-$end_core python train.py \
        # # BEFORE
        # AFTER
        python train.py \
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

DATASETS=(
    # "la"
    # "paris"
    "shanghai"
)

for data in "${DATASETS[@]}"; do
    # echo "run_models $data nagphormer 1 64 16 &"
    echo "run_models $data graphgps   0 0 16 &"

    # run_models "$data" nagphormer 0 0 16 &
    # pid1=$!

    run_models "$data" graphgps 0 0 15 &
    pid2=$!

    # wait $pid1
    wait $pid2
done
