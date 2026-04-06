# 🧪 ProbGT — Probabilistic Graph Transformer for Molecular Property Prediction

## 📌 Overview

**ProbGT** is a research framework for molecular property prediction that combines:

* Self-supervised pretraining (SSL) on 3D molecular conformer ensembles
* Supervised fine-tuning on downstream tasks (regression & classification)
* Multi-modal representations:

  * 1D: SMILES tokens
  * 2D: GNN / subgraphs
  * 3D: SchNet / GEMNet
* Subgraph-augmented Transformer architecture (ESAN-style)

The framework targets datasets such as:

* GEOM
* KRAKEN
* DRUGS (MARCEL)
* MoleculeNet (BACE, BBBP, HIV, etc.)
* ZINC
* EXP

---

## 📁 Repository Structure

```bash
ProbGT-boshra-c-free-v2/
├── main.py                     # Batch runner over datasets & seeds
├── pretrain.py                 # SSL pretraining logic
├── finetune.py                 # Supervised fine-tuning
├── baseline.py                 # Baseline models
├── load_evaluation.py          # Evaluate saved checkpoints
├── playground.py               # Experimental sandbox
├── sweep.py / sweep_run.py     # W&B hyperparameter sweeps
├── tokenizer.py                # SMILES tokenizer
├── tokenizer.json              # Tokenizer vocabulary
│
├── cfgs/
│   ├── default.yaml
│   ├── c-free-iclr/
│   │   ├── drugs/
│   │   ├── kraken/
│   │   └── molnet/
│   ├── downstream/
│   ├── experiments/
│   └── molmix/
│
├── layers/
│   ├── models.py
│   ├── ssl_models.py
│   ├── ssl_model_3d.py
│   ├── gnn.py
│   ├── attention.py
│   ├── encoders.py
│   ├── esan_models.py
│   ├── model_3d.py
│   ├── schnet.py
│   ├── gemnet.py
│   ├── gemnet_layers/
│   └── gemnet_utils/
│
├── preprocessing/
│   ├── dataloaders.py
│   ├── subgraphs.py
│   ├── datasets/
│   │   ├── geom.py
│   │   ├── moleculenet.py
│   │   ├── zinc.py
│   │   ├── tu.py
│   │   ├── planarsatpairsdataset.py
│   │   └── marcel/
│   │       ├── kraken.py
│   │       └── drugs.py
│   │   └── marcel_ensemble/
│   │       ├── ensemble.py
│   │       ├── kraken.py
│   │       ├── drugs.py
│   │       ├── ee.py
│   │       ├── bde.py
│   │       ├── samplers.py
│   │       └── multibatch.py
│   └── utils/
│       ├── molecule_feats.py
│       ├── molecule_to_data.py
│       ├── augment.py
│       ├── augment_conformers.py
│       ├── create_subgraphs.py
│       ├── scaffolds.py
│       ├── misc.py
│       └── target_metric.py
│
├── utils/
│   ├── misc.py
│   ├── training.py
│   ├── metrics.py
│   ├── evaluation.py
│   ├── masking.py
│   ├── prop.py
│   └── fragmentation/
│
└── slurm_*.sbatch / .sh
```

---

## ⚙️ Required Libraries

### Environment Setup

```bash
conda env create -f env_pyg.yaml
conda activate pyg
```

---

## 🚀 Usage

All scripts follow:

```bash
python <script>.py --cfg <config.yaml> [key=value overrides]
```

---

### 1. 🔁 Pretraining (SSL)

```bash
python pretrain.py --cfg cfgs/c-free-iclr/pretrain-molmix-3d.yaml
```

**Key options:**

* `data.dataset`: e.g. geom
* `data.n_conformers`: default 3
* `data.policy`: subgraph strategy
* `data.subgraph_types`: ["3-ego", "4-ego"]
* `data.with_3d`: true/false
* `max_epoch`, `batch_size`, `lr`
* `scheduler_type`
* `save_ckpt`
* `wandb.use_wandb`

---

### 2. 🎯 Fine-Tuning

**With pretrained model:**

```bash
python finetune.py --cfg cfgs/c-free-iclr/fft-molmix-no-sched.yaml
```

**From scratch:**

```bash
python finetune.py --cfg cfgs/c-free-iclr/fft-molmix-scratch-no-sched.yaml
```

**MoleculeNet:**

```bash
python main.py --cfg cfgs/c-free-iclr/molnet/fft-3d-molmix.yaml
```

**Key options:**

* `data.dataset`
* `load_ckpt`
* `backbone_file`
* `freeze_backbone`
* `eval_test`
* `train_subset_perc`
* `norm_target`
* `model.with_3d`
* `model.model_3d`

---

### 3. 🔄 Batch Experiments

```bash
python main.py
```

Or:

```bash
python main.py --cfg cfgs/c-free-iclr/drugs/fft-molmix3d.yaml
```

---

### 4. ⚡ CLI Overrides

```bash
python finetune.py --cfg <cfg> \
    data.dataset=kraken_b5 \
    batch_size=32 \
    lr=1e-4 \
    wandb.use_wandb=false
```

---

### 5. 🧹 Data Preparation

```bash
python main.py --cfg cfgs/experiments/prepare-data.yaml
```

Or:

```bash
sbatch slurm_prepare_data.sbatch <log_prefix>
```

---

### 6. 📊 Disable W&B

```yaml
wandb:
  use_wandb: false
```

or

```bash
wandb.use_wandb=false
```

---

## 📝 Notes

* GPU strongly recommended (CPU is very slow for 3D models)
* FlashAttention requires:

  * Compute capability ≥ 7.5
  * CUDA ≥ 11.6
* Default dataset path: `./data/datasets`
* Checkpoints: `best-models/`
* Tokenizer path: `tokenizer.json`

---

If you want, I can further **shorten this for GitHub**, or make a **NeurIPS-style project page version** (more narrative, less engineering-heavy).
