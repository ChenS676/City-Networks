# for MODEL in gcn sage gin; do
#     for DATASET in maxclique_easy maxclique_medium maxclique_hard; do
#         uv run baselines.py --model $MODEL --dataset_name $DATASET \
#             --hidden_dim 256 --num_layers 4 --lap_pe_dim 16
#     done
# done

# for MODEL in gt gps; do
#     for DATASET in maxclique_easy maxclique_medium maxclique_hard; do
#         uv run baselines.py --model $MODEL --dataset_name $DATASET \
#             --hidden_dim 256 --num_layers 4 --num_heads 8 --lap_pe_dim 16 \
#             --gps_attn_type multihead
#     done
# done

# for DATASET in maxclique_easy maxclique_medium maxclique_hard; do
#     uv run baselines.py --model nagphormer --dataset_name $DATASET \
#         --hidden_dim 256 --num_layers 4 --num_heads 8 --lap_pe_dim 16 --num_hops 3
# done
#!/bin/bash
# ============================================================
# run_baselines.sh
# Parallel experiment launcher for baselines.py
# Mirrors the structure of the CityNet train.sh:
#   - one run_experiment() function per model config
#   - models launched in background (&) per GPU
#   - wait on PIDs so the script blocks until all finish
#
# Usage:
#   bash run_baselines.sh
#   bash run_baselines.sh maxclique_easy    # override dataset
# ============================================================

# ── Global hyper-parameters (edit here) ─────────────────────
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

# NAGphormer-specific
NUM_HOPS=3

# GPS-specific
GPS_ATTN=multihead

DATA_ROOT="./data_graphbench"
WANDB_PROJECT="bench_maxclique_baselines"

# Dataset — can be overridden via first CLI arg
DATASET="${1:-maxclique_easy}"

# ── Helper: run one baseline ─────────────────────────────────
# Args: $1=model  $2=gpu_id
run_experiment() {
    local MODEL=$1
    local GPU=$2

    echo "[$(date +%H:%M:%S)] START  model=$MODEL  gpu=$GPU  dataset=$DATASET"

    CUDA_VISIBLE_DEVICES=$GPU python baselines.py \
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
        --wandb_project      "$WANDB_PROJECT"  \
        --num_hops           $NUM_HOPS         \
        --gps_attn_type      $GPS_ATTN         \
        2>&1 | tee "logs/${DATASET}_${MODEL}.log"

    echo "[$(date +%H:%M:%S)] DONE   model=$MODEL  gpu=$GPU  dataset=$DATASET"
}

# ── Setup ─────────────────────────────────────────────────────
mkdir -p logs

# Count available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
echo "Detected $NUM_GPUS GPU(s). Launching experiments for dataset: $DATASET"
echo "========================================================"

# ── Parallelisation strategy ──────────────────────────────────
# GCN, SAGE, GIN are lightweight → pack two per GPU
# GT, GPS, NAGphormer are heavier → one per GPU
#
# With 1 GPU:  all models run sequentially (safest)
# With 2 GPUs: lightweight models share GPU-0, heavy on GPU-1
# With 4 GPUs: maximum parallelism, one wave per GPU pair

if [ "$NUM_GPUS" -ge 4 ]; then
    echo "4-GPU layout: 6 models across 4 GPUs (2 per GPU)"

    # GPU 0: GCN + SAGE
    run_experiment gcn    0 &  PID_GCN=$!
    run_experiment sage   0 &  PID_SAGE=$!

    # GPU 1: GIN + GT
    run_experiment gin    1 &  PID_GIN=$!
    run_experiment gt     1 &  PID_GT=$!

    # GPU 2: GPS
    run_experiment gps    2 &  PID_GPS=$!

    # GPU 3: NAGphormer
    run_experiment nagphormer 3 &  PID_NAG=$!

    wait $PID_GCN $PID_SAGE $PID_GIN $PID_GT $PID_GPS $PID_NAG

elif [ "$NUM_GPUS" -ge 2 ]; then
    echo "2-GPU layout: lightweight models on GPU-0, heavy on GPU-1"

    # Wave 1: lightweight (GCN, SAGE in parallel on GPU 0; GIN on GPU 1)
    run_experiment gcn    0 &  PID_GCN=$!
    run_experiment sage   0 &  PID_SAGE=$!
    run_experiment gin    1 &  PID_GIN=$!
    wait $PID_GCN $PID_SAGE $PID_GIN

    echo "[$(date +%H:%M:%S)] Wave 1 done — starting Wave 2"

    # Wave 2: heavy (GT on GPU 0; GPS on GPU 1)
    run_experiment gt     0 &  PID_GT=$!
    run_experiment gps    1 &  PID_GPS=$!
    wait $PID_GT $PID_GPS

    echo "[$(date +%H:%M:%S)] Wave 2 done — starting Wave 3"

    # Wave 3: NAGphormer (GPU 0)
    run_experiment nagphormer 0 &  PID_NAG=$!
    wait $PID_NAG

else
    echo "1-GPU layout: sequential execution"

    for MODEL in gcn sage gin gt gps nagphormer; do
        run_experiment "$MODEL" 0
    done
fi

echo "========================================================"
echo "[$(date +%H:%M:%S)] All experiments complete for dataset: $DATASET"
echo "Results written to: ${DATASET}/baseline_results.csv"
echo "Logs in: logs/"