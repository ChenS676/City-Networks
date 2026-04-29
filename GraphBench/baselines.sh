#!/bin/bash
# ============================================================
# run_baselines.sh
# Single A100-SXM4-40GB (40 GB VRAM) + dynamic CPU pinning.
#
# Usage:
#   bash run_baselines.sh                   # maxclique_easy
#   bash run_baselines.sh maxclique_hard    # override dataset
#
# In a SLURM job:
#   #SBATCH --gres=gpu:1
#   #SBATCH --cpus-per-task=16   (or however many you allocate)
#   bash run_baselines.sh
# ============================================================

# ── Global hyper-parameters ──────────────────────────────────
EPOCHS=50
BATCH_SIZE=256
TEST_BATCH_SIZE=64
HIDDEN_DIM=256
NUM_LAYERS=4
NUM_HEADS=8
LAP_PE_DIM=16
DROPOUT=0.1
LR=1e-4
WEIGHT_DECAY=0.0
GRAD_CLIP=0.5
TRAIN_SUBSET=0.1
LOG_EVERY=20
NUM_HOPS=3
GPS_ATTN=multihead

DATA_ROOT="./data_graphbench"
WANDB_PROJECT="bench_maxclique_baselines"
DATASET="${1:-maxclique_medium}"
GPU=0   # single A100

# ── CPU layout ───────────────────────────────────────────────
# Detect total available cores at runtime (works on login node
# and inside SLURM — SLURM sets $SLURM_CPUS_PER_TASK if allocated)
TOTAL_CORES="${SLURM_CPUS_PER_TASK:-$(nproc)}"

# Models to run (order = lightest → heaviest, maximises early results)
MODELS=(gt nagphormer gps)
NUM_MODELS=${#MODELS[@]}

# Divide cores evenly; each model gets at least 1
CORES_PER_MODEL=$(( TOTAL_CORES / NUM_MODELS ))
CORES_PER_MODEL=$(( CORES_PER_MODEL < 1 ? 1 : CORES_PER_MODEL ))

# DataLoader workers = cores_per_model - 1 (leave 1 for the main process)
# Clamp to [0, 8]
DL_WORKERS=$(( CORES_PER_MODEL - 1 ))
DL_WORKERS=$(( DL_WORKERS < 0 ? 0 : DL_WORKERS ))
DL_WORKERS=$(( DL_WORKERS > 8 ? 8 : DL_WORKERS ))

echo "========================================================"
echo "Dataset   : $DATASET"
echo "GPU       : A100 ($GPU)"
echo "CPU cores : $TOTAL_CORES total  →  $CORES_PER_MODEL per model"
echo "DL workers: $DL_WORKERS"
echo "Models    : ${MODELS[*]}"
echo "========================================================"

mkdir -p logs

# ── Per-model runner ─────────────────────────────────────────
# Args: $1=model  $2=first_cpu_core
run_experiment() {
    local MODEL=$1
    local FIRST_CORE=$2
    local LAST_CORE=$(( FIRST_CORE + CORES_PER_MODEL - 1 ))
    # Clamp to valid range
    local MAX_CORE=$(( TOTAL_CORES - 1 ))
    LAST_CORE=$(( LAST_CORE > MAX_CORE ? MAX_CORE : LAST_CORE ))

    echo ""
    echo "[$(date +%H:%M:%S)] ── START $MODEL"
    echo "  CPU cores : $FIRST_CORE-$LAST_CORE"
    echo "  GPU       : $GPU"

    taskset -c "$FIRST_CORE-$LAST_CORE" \
    uv run baselines.py \
        --model              "$MODEL"          \
        --dataset_name       "$DATASET"        \
        --data_root          "$DATA_ROOT"      \
        --epochs             $EPOCHS           \
        --batch_size         $BATCH_SIZE       \
        --test_batch_size    $TEST_BATCH_SIZE  \
        --hidden_dim         $HIDDEN_DIM       \
        --num_layers         $NUM_LAYERS       \
        --num_heads          $NUM_HEADS        \
        --lap_pe_dim         $LAP_PE_DIM       \
        --dropout            $DROPOUT          \
        --adam_max_lr        $LR               \
        --weight_decay       $WEIGHT_DECAY     \
        --grad_clip_norm     $GRAD_CLIP        \
        --train_subset_ratio $TRAIN_SUBSET     \
        --log_every          $LOG_EVERY        \
        --num_hops           $NUM_HOPS         \
        --gps_attn_type      $GPS_ATTN         \
        --wandb_project      "$WANDB_PROJECT"  \
        2>&1 | tee "logs/${DATASET}_${MODEL}.log"

    local EXIT=$?
    if [ $EXIT -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ── DONE  $MODEL  ✓"
    else
        echo "[$(date +%H:%M:%S)] ── FAIL  $MODEL  exit=$EXIT"
    fi
}

# ── Sequential execution with pinned CPU slices ───────────────
# With a single GPU, true parallelism would cause VRAM contention.
# Instead we run one model at a time and pin each to its own
# CPU core slice so the OS scheduler doesn't migrate threads.
#
# Core assignment:
#   Model 0 (gcn)       → cores 0   .. C-1
#   Model 1 (sage)      → cores C   .. 2C-1
#   ...
#   Model N-1 (gps)     → cores (N-1)*C .. N*C-1  (clamped)
#
# If TOTAL_CORES < NUM_MODELS all models share core 0 (nproc=1 case).

CORE=0
for MODEL in "${MODELS[@]}"; do
    run_experiment "$MODEL" "$CORE" &
    CORE=$(( CORE + CORES_PER_MODEL ))
    # Wrap around if we run out (handles nproc=1 case)
    CORE=$(( CORE >= TOTAL_CORES ? 0 : CORE ))
done

echo ""
echo "========================================================"
echo "[$(date +%H:%M:%S)] All experiments complete."
echo "Results : ${DATASET}/baseline_results.csv"
echo "Logs    : logs/"
echo "========================================================"