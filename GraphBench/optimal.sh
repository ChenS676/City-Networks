#!/bin/bash
# ============================================================
# run_baselines.sh
# Single A100-SXM4-40GB — iterates over multiple datasets,
# runs all models per dataset sequentially with CPU pinning.
#
# Usage:
#   bash run_baselines.sh                          # all datasets
#   bash run_baselines.sh maxclique_easy           # one dataset
#   bash run_baselines.sh "maxclique_easy maxclique_hard"  # subset
#
# SLURM:
#   #SBATCH --gres=gpu:1
#   #SBATCH --cpus-per-task=16
#   bash run_baselines.sh
# ============================================================

# ── Global hyper-parameters ──────────────────────────────────
EPOCHS=50
BATCH_SIZE=512
TEST_BATCH_SIZE=128
HIDDEN_DIM=256
NUM_LAYERS=4
NUM_HEADS=8
LAP_PE_DIM=16
DROPOUT=0.1
# LR scaled with batch size: bs=256→lr=1e-4, bs=512→lr=2e-4
LR=2e-4
WEIGHT_DECAY=0.0
GRAD_CLIP=0.5
TRAIN_SUBSET=0.1
LOG_EVERY=20
NUM_HOPS=3
GPS_ATTN=multihead
DATA_ROOT="./data_graphbench"
WANDB_PROJECT="bench_maxclique_baselines"

# ── Python interpreter ────────────────────────────────────────
PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: venv not found at $PYTHON — run: uv sync"
    exit 1
fi
echo "Python : $PYTHON"

# ── Datasets ─────────────────────────────────────────────────
# Override via first argument, otherwise run all three
if [ -n "$1" ]; then
    read -ra DATASETS <<< "$1"
else
    DATASETS=(maxclique_easy maxclique_medium maxclique_hard)
fi

# ── Models ───────────────────────────────────────────────────
MODELS=(gcn sage gin gt nagphormer gps)
NUM_MODELS=${#MODELS[@]}

# ── CPU layout ────────────────────────────────────────────────
TOTAL_CORES="${SLURM_CPUS_PER_TASK:-$(nproc)}"
CORES_PER_MODEL=$(( TOTAL_CORES / NUM_MODELS ))
CORES_PER_MODEL=$(( CORES_PER_MODEL < 1 ? 1 : CORES_PER_MODEL ))

echo "========================================================"
echo "Datasets  : ${DATASETS[*]}"
echo "Models    : ${MODELS[*]}"
echo "CPU cores : $TOTAL_CORES total -> $CORES_PER_MODEL per model"
echo "GPU       : A100"
echo "LR        : $LR  (scaled for bs=$BATCH_SIZE)"
echo "========================================================"

mkdir -p logs

# ── Step 1: Pre-compute LapPE for every dataset once ─────────
# Safe to re-run — skips if cache already exists
for DATASET in "${DATASETS[@]}"; do
    echo "[$(date +%H:%M:%S)] LapPE pre-compute: $DATASET"
    "$PYTHON" pre_compute.py \
        --dataset_name "$DATASET" \
        --data_root    "$DATA_ROOT" \
        --lap_pe_dim   $LAP_PE_DIM
    if [ $? -ne 0 ]; then
        echo "ERROR: LapPE pre-compute failed for $DATASET — aborting."
        exit 1
    fi
done
echo "[$(date +%H:%M:%S)] All LapPE caches ready."
echo ""

exit
# ── Per-model runner ──────────────────────────────────────────
# Args: $1=dataset  $2=model  $3=first_cpu_core
run_experiment() {
    local DATASET=$1
    local MODEL=$2
    local FIRST_CORE=$3
    local MAX_CORE=$(( TOTAL_CORES - 1 ))
    local LAST_CORE=$(( FIRST_CORE + CORES_PER_MODEL - 1 ))
    FIRST_CORE=$(( FIRST_CORE > MAX_CORE ? MAX_CORE : FIRST_CORE ))
    LAST_CORE=$(( LAST_CORE   > MAX_CORE ? MAX_CORE : LAST_CORE  ))

    local LOG="logs/${DATASET}_${MODEL}.log"

    echo "[$(date +%H:%M:%S)] -- START  ds=$DATASET  model=$MODEL  cores=$FIRST_CORE-$LAST_CORE"

    if [ "$TOTAL_CORES" -gt 1 ]; then
        TASKSET_CMD="taskset -c $FIRST_CORE-$LAST_CORE"
    else
        TASKSET_CMD=""
    fi

    $TASKSET_CMD \
    "$PYTHON" baselines.py \
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
        --use_cached_lap_pe  true              \
        --wandb_project      "$WANDB_PROJECT"  \
        2>&1 | tee "$LOG"

    local EXIT=${PIPESTATUS[0]}
    if [ $EXIT -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] -- DONE   ds=$DATASET  model=$MODEL"
    else
        echo "[$(date +%H:%M:%S)] -- FAIL   ds=$DATASET  model=$MODEL  exit=$EXIT  (log: $LOG)"
    fi
    return $EXIT
}

# ── Main loop: dataset -> model (sequential, CPU-pinned) ──────
FAILED=()

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "==== Dataset: $DATASET ===="

    CORE=0
    for MODEL in "${MODELS[@]}"; do
        run_experiment "$DATASET" "$MODEL" "$CORE"
        [ $? -ne 0 ] && FAILED+=("${DATASET}/${MODEL}")

        CORE=$(( CORE + CORES_PER_MODEL ))
        CORE=$(( CORE >= TOTAL_CORES ? 0 : CORE ))
    done

    echo "[$(date +%H:%M:%S)] Finished $DATASET -> ${DATASET}/baseline_results.csv"
done

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "[$(date +%H:%M:%S)] All experiments complete."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
    exit 1
else
    echo "  All runs succeeded"
fi
echo "  Logs: logs/"
echo "========================================================"