#!/bin/bash


# for shanghai, la, paris

CUDA_LAUNCH_BLOCKING=1 uv run train_hrw.py \
    --method            hierarchial_rw \
    --dataset           la \
    --device            0 \
    --runs              3 \
    --seed              1 \
    --epochs            30000 \
    --experiment_name   HierarchialRW_CityNet_Jun25 \
    --display_step      100 \
    --weight_decay      1e-5 \
    --save_model \
    --hierarchial_rw_walk_length      8 \ 
    --hierarchial_rw_num_walks        4 \  
    --hierarchial_rw_recurrent_steps  2 \
    --hierarchial_rw_hidden_dim       128 \
    --hierarchial_rw_num_layers       1 \
    --hierarchial_rw_num_heads        16 \
    --hierarchial_rw_attn_dropout     0.1 \
    --hierarchial_rw_ffn_dropout      0.1 \
    --hierarchial_rw_resid_dropout    0.1 \
    --hierarchial_rw_drop_path        0.05 \
    --hierarchial_rw_muon_min_lr      1e-4 \
    --hierarchial_rw_muon_max_lr      1e-3 \
    --hierarchial_rw_adam_lr          5e-4 \
    --hierarchial_rw_node2vec_p       1.0 \
    --hierarchial_rw_node2vec_q       1.0 \
    --hierarchial_rw_batch_size       512 \
    --wandb_project     gnn-benchmark-hierarchial-rw
