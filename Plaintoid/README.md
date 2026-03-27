# HeART — Heterogeneous Anonymized Random-walk Transformer

**Link Prediction on Graphs**

---

## Overview

HeART is a Transformer-based graph encoder for link prediction. Rather than using message-passing (GNNs), it encodes node neighbourhoods by sampling anonymized random walks and processing them through a standard Transformer with Rotary Positional Embeddings (RoPE). The anonymization step replaces absolute node IDs with relative first-occurrence indices, allowing the model to learn purely structural patterns that generalize across graphs.

### Key design choices

- No GNN / no message passing — structure is captured via random walks
- Node2Vec-style walk sampling, controllable via `p` and `q`
- Walk anonymization for structure-only, ID-agnostic representations
- RoPE applied at anonymized positions, not raw sequence positions
- SwiGLU FFN, pre-RMSNorm, stochastic depth (drop path)
- muP (Maximal Update Parametrization) initialization for stable scaling
- Muon optimizer (Nesterov + orthogonalization) for matrix weights, and Adam for gains, biases, and MLP parameters
- Optional recurrent walk expansion for larger receptive fields

---

## Project structure

```text
linkpred/
│
├── model.py          # Model definition — all architecture & method code
│   │
│   ├── MupConfig                    muP initialization dataclass
│   ├── RotaryPositionalEmbeddings   RoPE implementation
│   ├── TransformerLayer             Single Transformer block
│   ├── Transformer                  Full encoder (stacks TransformerLayers)
│   ├── LinkPredictorMLP             MLP-based link scoring head
│   ├── binary_cross_entropy_loss    BCE loss over pos/neg edge scores
│   ├── anonymize_rws                Walk anonymization function
│   └── get_random_walk_batch        Walk sampling + anonymization + feature fetch
│
├── heart.py          # Training, evaluation, and logging script
│   │
│   ├── get_config()                 Argparse configuration
│   ├── load_graph_arxiv23()         Loader for Arxiv 2023 graph
│   ├── downsample_edges()           Edge downsampling utility
│   ├── sample_negative_edges()      On-the-fly negative edge sampling
│   ├── trapezoidal_lr_schedule()    Warmup → plateau → cooldown LR schedule
│   ├── get_metric_score()           Computes MRR, AUC, AP, Hits@K
│   ├── test_edge()                  Batched edge scoring (inference)
│   ├── evaluate_link_prediction()   Full pos/neg evaluation pass
│   ├── evaluate_and_log()           Eval + W&B logging + early stopping
│   └── __main__                     Data loading, model init, training loop
│
└── README.txt              # This file