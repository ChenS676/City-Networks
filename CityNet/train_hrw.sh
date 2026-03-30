#!/bin/bash
# =============================================================================
# train.sh
# CityNetwork Benchmark — GNN baselines + HierarchialRW
#
# Usage:
#   bash train.sh                   # run everything defined below
#   bash train.sh paris sgformer 0  # ad-hoc: dataset model gpu
#
# Parallelism:
#   Each run_models / run_hierarchial_rw call launched with & runs in its own process.
#   taskset -c <cores> pins it to specific CPU cores to avoid contention.
#   Remove taskset if you don't want CPU pinning.
#
# Output:
#   results/<exp_name>/<dataset>-<method>.csv   (incremental, crash-safe)
#   results/<exp_name>/checkpoints/             (best model .pt per seed)
# =============================================================================

# ---------------------------------------------------------------------------
# Shared experiment config
# ---------------------------------------------------------------------------
EXP_NAME='HierarchialRW_CityNet_Jun25'
EPOCHS=30000
RUNS=3
SEED=0
DISPLAY_STEP=100
WANDB_PROJECT='gnn-benchmark-hierarchial-rw'
WANDB_ENTITY=''          # set to your W&B username/team, or leave empty

# ---------------------------------------------------------------------------
# run_models — standard GNN methods  (gcn | sage | cheb | sgformer | ...)
#
#   $1 DATASET     : paris | shanghai | la | london
#   $2 MODEL       : gcn | sage | cheb | sgformer | nagphormer | graphgps | gt
#   $3 GPU         : GPU index (0-based)
#   $4 START_CORE  : first CPU core for taskset
#   $5 NUM_CORES   : number of CPU cores to allocate
# ---------------------------------------------------------------------------
run_models() {
    local DATASET=$1
    local MODEL=$2
    local GPU=$3
    local START_CORE=$4
    local NUM_CORES=$5

    local END_CORE=$(( START_CORE + NUM_CORES - 1 ))

    # Per-model hyperparams
    local LAYERS=2
    local HIDDEN=64
    local LR=5e-4
    local BATCH=200000
    local WD=1e-5
    local DROPOUT=0.2
    local NUM_HEADS=1
    local TRANS_WD=1e-2
    local TRANS_DROPOUT=0.5

    echo "[GNN] Dataset=$DATASET  Model=$MODEL  GPU=$GPU  Cores=$START_CORE-$END_CORE"

    taskset -c "$START_CORE-$END_CORE" python train.py \
        --method           "$MODEL"     \
        --dataset          "$DATASET"   \
        --device           "$GPU"       \
        --runs             "$RUNS"      \
        --seed             "$SEED"      \
        --epochs           "$EPOCHS"    \
        --experiment_name  "$EXP_NAME"  \
        --num_layers       "$LAYERS"    \
        --hidden_channels  "$HIDDEN"    \
        --lr               "$LR"        \
        --batch_size       "$BATCH"     \
        --weight_decay     "$WD"        \
        --dropout          "$DROPOUT"   \
        --display_step     "$DISPLAY_STEP" \
        --num_heads        "$NUM_HEADS" \
        --transformer_weight_decay  "$TRANS_WD"      \
        --transformer_dropout       "$TRANS_DROPOUT" \
        --save_model                 \
        --wandb_project    "$WANDB_PROJECT" \
        ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"}
}

# ---------------------------------------------------------------------------
# run_hierarchial_rw — HierarchialRW  (anonymized random-walk Transformer)
#
#   $1 DATASET     : paris | shanghai | la | london
#   $2 GPU         : GPU index (0-based)
#   $3 START_CORE  : first CPU core for taskset
#   $4 NUM_CORES   : number of CPU cores to allocate
# ---------------------------------------------------------------------------
run_hierarchial_rw() {
    local DATASET=$1
    local GPU=$2
    local START_CORE=$3
    local NUM_CORES=$4

    local END_CORE=$(( START_CORE + NUM_CORES - 1 ))

    # HierarchialRW architecture
    local WALK_LENGTH=8
    local NUM_WALKS=16
    local RECURRENT_STEPS=1
    local HIDDEN_DIM=128
    local NUM_LAYERS=1
    local NUM_HEADS=16

    # HierarchialRW regularisation
    local ATTN_DROP=0.1
    local FFN_DROP=0.1
    local RESID_DROP=0.1
    local DROP_PATH=0.05

    # HierarchialRW optimiser
    local MUON_MIN_LR=1e-4
    local MUON_MAX_LR=1e-3
    local ADAM_LR=5e-4

    # Node2Vec walk params (p=q=1 → uniform BFS/DFS balance)
    local P=1.0
    local Q=1.0

    # Node batch size for walk encoding (reduce if OOM; use 4 for recurrent_steps>1)
    local HIERARCHIAL_RW_BATCH=512

    echo "[HierarchialRW] Dataset=$DATASET  GPU=$GPU  Cores=$START_CORE-$END_CORE"

    taskset -c "$START_CORE-$END_CORE" python train.py \
        --method           hierarchial_rw        \
        --dataset          "$DATASET"   \
        --device           "$GPU"       \
        --runs             "$RUNS"      \
        --seed             "$SEED"      \
        --epochs           "$EPOCHS"    \
        --experiment_name  "$EXP_NAME"  \
        --display_step     "$DISPLAY_STEP" \
        --weight_decay     1e-5         \
        --save_model                    \
        \
        --hierarchial_rw_walk_length      "$WALK_LENGTH"      \
        --hierarchial_rw_num_walks        "$NUM_WALKS"         \
        --hierarchial_rw_recurrent_steps  "$RECURRENT_STEPS"  \
        --hierarchial_rw_hidden_dim       "$HIDDEN_DIM"        \
        --hierarchial_rw_num_layers       "$NUM_LAYERS"        \
        --hierarchial_rw_num_heads        "$NUM_HEADS"         \
        --hierarchial_rw_attn_dropout     "$ATTN_DROP"         \
        --hierarchial_rw_ffn_dropout      "$FFN_DROP"          \
        --hierarchial_rw_resid_dropout    "$RESID_DROP"        \
        --hierarchial_rw_drop_path        "$DROP_PATH"         \
        --hierarchial_rw_muon_min_lr      "$MUON_MIN_LR"       \
        --hierarchial_rw_muon_max_lr      "$MUON_MAX_LR"       \
        --hierarchial_rw_adam_lr          "$ADAM_LR"           \
        --hierarchial_rw_node2vec_p       "$P"                 \
        --hierarchial_rw_node2vec_q       "$Q"                 \
        --hierarchial_rw_batch_size       "$HIERARCHIAL_RW_BATCH"       \
        \
        --wandb_project    "$WANDB_PROJECT" \
        ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"}
}

# ---------------------------------------------------------------------------
# Ad-hoc single run:  bash train.sh <dataset> <model> <gpu>
# ---------------------------------------------------------------------------
if [ $# -eq 3 ]; then
    DATASET=$1
    MODEL=$2
    GPU=$3
    if [ "$MODEL" = "hierarchial_rw" ]; then
        run_hierarchial_rw "$DATASET" "$GPU" 0 8
    else
        run_models "$DATASET" "$MODEL" "$GPU" 0 8
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Full benchmark — edit datasets and GPU/core assignments below
#
# Layout assumption: 4 GPUs, 40 CPU cores (adjust to your machine)
#   GPU 0 → cores  0-9   (10 cores)
#   GPU 1 → cores 10-19
#   GPU 2 → cores 20-29
#   GPU 3 → cores 30-39
#
# To check your core count:  nproc
# To skip CPU pinning:        remove the taskset call inside run_models/run_hierarchial_rw
# ---------------------------------------------------------------------------

DATASETS=(
    "paris"
    # "shanghai"
    # "la"
    # "london"
)

for DATA in "${DATASETS[@]}"; do

    # ── HierarchialRW ────────────────────────────────────────────────────────────────
    run_hierarchial_rw  "$DATA"           0  0  8 &

done

# Wait for all background jobs to finish
wait
echo "All runs complete. Results → results/${EXP_NAME}/"
