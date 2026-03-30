your_project/
│
├── train.py                           ← REPLACE existing train.py
│   ├── MupConfig                        μP init dataclass
│   ├── RotaryPositionalEmbeddings       RoPE implementation
│   ├── TransformerLayer                 Single transformer block (RMSNorm, RoPE, SwiGLU, drop-path)
│   ├── HierarchialRWEncoder             Walk-sequence transformer encoder
│   ├── HierarchialRWNodeClassifier      Encoder + linear classification head
│   ├── anonymize_rws()                  Replace node IDs with first-occurrence indices
│   ├── get_random_walk_batch()          Node2Vec walk sampling → anonymize → fetch features
│   ├── train_gnn()                      One epoch — GNN methods via NeighborLoader
│   ├── eval_gnn()                       Evaluation — GNN methods
│   ├── train_hierarchial_rw_epoch()     One epoch — HierarchialRW walk-batch loop
│   ├── eval_hierarchial_rw()            Evaluation — HierarchialRW
│   ├── parser_add_hierarchial_rw_args() Adds --hierarchial_rw_* CLI flags
│   ├── trapezoidal_lr()                 Warmup → plateau → cooldown LR schedule
│   ├── save_result_row()                Incremental crash-safe CSV writer
│   └── main()                           Per-seed runner (forks on --method)
│
├── train.sh                           ← REPLACE existing run.sh / train.sh
│   ├── run_models()                     Launches gcn | sage | cheb | sgformer via taskset
│   └── run_hierarchial_rw()             Launches HierarchialRW with all --hierarchial_rw_* flags
│
├── hierarchial_rw_e2e.py              ← NEW — standalone link prediction (no other local imports)
│   ├── MupConfig                        (same as above, self-contained copy)
│   ├── RotaryPositionalEmbeddings
│   ├── TransformerLayer
│   ├── HierarchialRWEncoder (Transformer)
│   ├── LinkPredictorMLP                 Element-wise product + MLP scoring head
│   ├── binary_cross_entropy_loss()
│   ├── anonymize_rws()
│   ├── get_random_walk_batch()
│   ├── evaluate_hits()                  OGB ogbl-collab evaluator
│   ├── evaluate_mrr()                   OGB ogbl-citation2 evaluator
│   ├── evaluate_auc()                   sklearn roc_auc + average_precision
│   ├── get_metric_score()               Combines all metrics
│   ├── score_edges()                    Batched edge scoring at inference
│   ├── run_evaluation()                 Full pos/neg eval pass
│   ├── evaluate_and_log()               Eval + W&B + early stopping + checkpoint
│   ├── get_config()                     All argparse arguments
│   ├── trapezoidal_lr_schedule()
│   ├── save_profiling_csv()             Writes per-epoch + summary profiling CSVs
│   └── __main__                         Data → splits → model → train loop → results
│
├── hierarchial_rw_model.py            ← NEW — model definition only (imported by train_hierarchial_rw.py)
│   ├── MupConfig
│   ├── RotaryPositionalEmbeddings
│   ├── TransformerLayer
│   ├── Transformer                      (the encoder, exported)
│   ├── LinkPredictorMLP
│   ├── binary_cross_entropy_loss()
│   ├── anonymize_rws()
│   └── get_random_walk_batch()
│
├── train_hierarchial_rw.py            ← NEW — link prediction trainer (imports hierarchial_rw_model.py)
│   ├── from hierarchial_rw_model import (all model components)
│   ├── str_to_bool() / get_config()     Argparse
│   ├── load_graph_arxiv23()
│   ├── downsample_edges()
│   ├── sample_negative_edges()
│   ├── trapezoidal_lr_schedule()
│   ├── get_metric_score()               evaluate_hits + evaluate_mrr + evaluate_auc (inline)
│   ├── test_edge()
│   ├── evaluate_link_prediction()
│   ├── evaluate_and_log()
│   ├── save_profiling_csv()
│   └── __main__                         Data → splits → model → train loop → profiling + results
│
├── benchmark/                         ← EXISTING — unchanged
│   ├── configs.py                       parse_method() — maps --method string to model class
│   └── utils.py                         eval_acc(), count_parameters(), plot_logging_info()
│
├── citynetworks/                      ← EXISTING — unchanged
│   └── ...                              CityNetwork dataset class
│
├── data/                              ← auto-created on first run by PyG / OGB
│   ├── Cora/
│   ├── CiteSeer/
│   ├── paris/
│   ├── shanghai/
│   └── ...
│
└── results/                           ← auto-created on first run
    └── <experiment_name>/
        ├── paris-gcn.csv                Per-run + aggregate results (appended)
        ├── paris-sage.csv
        ├── paris-hierarchial_rw.csv
        ├── profiling/                   (link prediction runs only)
        │   ├── profiling_<run_id>.csv   Per-epoch memory + timing
        │   └── profiling_summary.csv    One row per run, appended
        └── checkpoints/
            ├── paris-gcn-seed0.pt
            └── paris-hierarchial_rw-seed0.pt