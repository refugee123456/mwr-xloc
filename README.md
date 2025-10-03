# Interpretable Few-Shot Transfer Learning for Microwave Radiometry Diagnosis  
**Cross-Location Adaptation with Graph Attention and Prototypical Metric Learning**

This repository contains the official implementation of our cross-location few-shot transfer pipeline for **Medical Microwave Radiometry (MWR)**.  
We model each exam as a *bilateral graph*, extract case-level embeddings with a **Q-K-V graph attention encoder** and perform **prototype-distance softmax** classification.  
During meta-test on a new anatomical location, we adapt with **instance recalibration**, **lightweight fine-tuning** and an **AdaBN-style VTAN normalization**—all designed to be **interpretable** and **data-efficient**.

---

## About

**Why MWR?** Early functional changes (micro-vascular remodeling, inflammation, metabolic heat) often precede morphological imaging findings. MWR captures these physiological signals but suffers from site/device drift, small datasets and class imbalance.

**What this repo provides**

- **Structured physiological representation**  
  *13-dim node features* (deep/shallow temps, left-right asymmetry, local/global z-scores) on a **bilateral graph** with mirrored edges.

- **Graph attention with tri-branch aggregation**  
  Local / Symmetry / Global branches are fused to yield a case embedding with clear clinical semantics.

- **Prototype-distance softmax (metric head)**  
  Distance to class prototypes with a learnable temperature; decision boundaries are affine and **auditable**.

- **Episode-level domain adaptation**  
  - *Instance Recalibration*: RBF-weighted support prototypes to suppress hubs/outliers.  
  - *Lightweight Fine-tuning*: compactness + margin + CORAL alignment (no extra linear head).  
  - *VTAN normalization*: per-dimension affine normalization (AdaBN-like) using support statistics.

- **Fair, imbalance-aware evaluation**  
  Reports **ACC / SPEC / SENS / G-mean / MCC / ROC-AUC** for each episode.

---

## Key Features

- **Cross-location few-shot**: train on Breast (source), adapt to **Leg** / **Lung** (target) with *K ∈ {3,5,10}* per class.
- **Interpretable geometry**: prototypes and fusion weights provide *why* the model decided.
- **Lightweight & stable**: no PyG dependency; works with vanilla PyTorch.
- **Reproducible**: YAML-first configs, fixed seeds, deterministic samplers.

---

## Repository Structure
```text
mwr-xloc/
├─ README.md                        # Usage guide: overview, dependencies, install, run, and how to reproduce results
│
├─ configs/                         # All YAML configs (organized by stage)
│  ├─ dataset/                      # Stage-1: raw table → graph preprocessing parameters
│  │  ├─ breast.yaml                # Breast preprocessing: reference points, node-feature recipe, export root
│  │  ├─ leg.yaml                   # Leg preprocessing: trimmed-mean settings, #points per side, export root
│  │  └─ lung.yaml                  # Lung preprocessing: reference points, node-feature recipe, export root
│  │
│  ├─ pretrain/                     # Stage-2: source-domain pretraining (supervised CE + optional contrastive)
│  │  ├─ pretrain_none.yaml         # Baseline CE classification with MetricHead; no contrastive loss
│  │  ├─ pretrain_contrastive.yaml  # CE + Siamese margin contrastive on graph-level features
│  │  ├─ pretrain_triplet_hard.yaml # CE + Triplet (hard mining) on graph-level features
│  │  ├─ pretrain_triplet_semi.yaml # CE + Triplet (semi-hard mining) on graph-level features
│  │  └─ pretrain_npairs.yaml       # CE + N-pairs contrastive objective
│  │
│  ├─ meta_test/                                              # Stage-3: cross-domain few-shot meta-test (target domain)
│  │  ├─ meta_test_fintune/                                   # (D) FT + Recal (+VTAN) — VTAN on; inner-loop fine-tuning enabled
│  │  │  ├─ metatest_leg_{3,5,10}_shot.yaml                   # Few-shot configs when target=Leg
│  │  │  └─ metatest_lung_{3,5,10}_shot.yaml                  # Few-shot configs when target=Lung
│  │  ├─ meta_test_without_vtan/                              # (C) FT + Recal (no VTAN) — fine-tuning on support, VTAN disabled
│  │  │  ├─ metatest_leg_{3,5,10}_shot.yaml
│  │  │  └─ metatest_lung_{3,5,10}_shot.yaml
│  │  ├─ simple_meta_test/                                    # (B) No-FT + Recal (+VTAN) — prototype recalibration + VTAN only
│  │  │  ├─ metatest_leg_{3,5,10}_shot.yaml
│  │  │  └─ metatest_lung_{3,5,10}_shot.yaml
│  │  └─ simple_meta_test_without_VATN/                       # (A) No-FT + Recal (no VTAN) — pure recalibration; no VTAN/no FT
│  │     ├─ metatest_leg_{3,5,10}_shot.yaml
│  │     └─ metatest_lung_{3,5,10}_shot.yaml
│  │     
│  │
│  └─ episode_viz/                  # Single-episode visualization presets (paper-style figures)
│     └─ leg_episode_3shot.yaml     # Deterministic episode draw + plotting options (UMAP/TSNE, DPI, fonts)
│
├─ data/
│  ├─ raw/                          # Raw CSV/Excel files (as exported from clinical systems)
│  │  ├─ breast.csv                 # Breast table with per-site measurements and labels
│  │  ├─ legs.csv                   # Leg table with per-site measurements and labels
│  │  └─ lungs.csv                  # Lung table with per-site measurements and labels
│  └─ processed/                    # Preprocessed graph datasets (produced by run_preprocess.py)
│     ├─ breast/ | leg/ | lung/     # One folder per anatomical site (dataset)
│     │  ├─ manifest.json           # Dataset manifest: ids, label y, file paths, #nodes, meta
│     │  ├─ splits.json             # Train/val/test indices (auto-generated if missing)
│     │  └─ <sample_id>/            # One directory per sample (graph)
│     │     ├─ nodes.npy            # [N,13] node feature matrix (z-scores, asymmetries, etc.)
│     │     ├─ edge_index.npy       # [2,E] undirected edges for bilateral fully-connected graphs
│     │     ├─ pairs.npy            # [M,2] left-right mirrored node index pairs (for Sym branch)
│     │     └─ meta.json            # Per-sample auxiliary metadata (timestamps, missingness, etc.)
│
├─ src/                             # Core code (three-stage pipeline)
│  ├─ scripts/                      # CLI entrypoints for each stage and for visualization
│  │  ├─ run_preprocess.py          # Load dataset/*.yaml → build graphs under data/processed/<site>/
│  │  ├─ run_pretrain.py            # Load pretrain/*.yaml → train PretrainNet; save best.ckpt + logs
│  │  ├─ run_metatest.py            # Load meta_test/*/*.yaml → sample episodes; run adaptation & report
│  │  └─ plot_episode_viz.py        # Draw a deterministic episode; plot 2×2 panels + save per-setting metrics
│  │
│  ├─ runners/                      # Orchestrators (parse YAML → construct models/loops → logging)
│  │  ├─ pretrain_runner.py         # Source training loop, early-stopping, checkpointing, CSV logging
│  │  └─ meta_eval_runner.py        # Episode sampler, PFAMetaAdapter calls, metric aggregation across episodes
│  │
│  ├─ algorithms/
│  │  ├─ pretrain.py                # PretrainNet: GraphEncoder → Local/Sym/Global branches → Fusion → MetricHead
│  │  ├─ pfa.py                     # Adaptation blocks: instance recalibration, prototype calc, VTAN, logits
│  │  └─ adaptation/
│  │     └─ pfa_finetuner.py        # PFAMetaAdapter: inner-loop FT (compactness/margin/CORAL) + inference
│  │                                 #  Handles BN-eval mode, gradient clipping, adaptive gamma, proto weights
│  │
│  ├─ models/
│  │  ├─ encoders/
│  │  │  └─ graph_encoder.py        # Q-K-V graph attention/MLP stacks to produce node embeddings
│  │  ├─ aggregators/               # Graph-to-case aggregation branches and fusion
│  │  │  ├─ local_branch.py         # Local attention pooling (per-site saliency → weighted sum)
│  │  │  ├─ sym_branch.py           # Symmetry aggregation using mirror pairs to emphasize asymmetry
│  │  │  ├─ global_branch.py        # Global pooling guided by sparse “disperse/aggregate” scores
│  │  │  └─ fusion.py               # Softmax-gated fusion of Local/Sym/Global into a 128-D case vector
│  │  ├─ heads/
│  │  │  └─ metric_head.py          # Distance-softmax classifier with learnable temperature (gamma)
│  │  └─ layers/
│  │     ├─ resnet_mlp.py           # Residual MLP blocks with norm/activation/dropout
│  │     └─ norm.py                 # Lightweight norm/activation helpers + linear initialization
│  │
│  ├─ losses/
│  │  ├─ metric.py                  # metric_ce_loss: CE on distance-softmax logits
│  │  └─ contrastive.py             # Siamese / Triplet (hard, semi-hard) / N-pairs objectives
│  │
│  ├─ evaluation/
│  │  ├─ metrics.py                 # Binary metrics: ACC, SPEC, SENS, G-mean, MCC, ROC-AUC
│  │  └─ reporter.py                # Unified printer/writer for train/test (console + CSV)
│  │
│  ├─ data/
│  │  ├─ datasets/
│  │  │  └─ mwr_dataset.py          # Read manifest; assemble PyTorch items for each sample/graph
│  │  ├─ collate.py                 # Batch-wise big-graph collation (nodes/edge_index/pairs/batch_ptr)
│  │  ├─ episodic_sampler.py        # Few-shot episode sampler (balanced support; query per class)
│  │  └─ processing/                # Stage-1 preprocessing implementations
│  │     ├─ base_processor.py       # Base interface: fit(df) and process_row(row) → (nodes, edges, pairs, meta)
│  │     ├─ ambient_norm.py         # Ambient/reference normalization fit + apply (linear slope; mean impute)
│  │     ├─ graph_builder.py        # Bilateral fully-connected edges, mirror links, optional self-loops
│  │     ├─ node_features_common.py # 13-D node features: shallow/deep temps, asymmetries, local/global z-scores
│  │     ├─ breast_processor.py     # Breast-specific preprocessing rules (drop-on-missing, reference correction)
│  │     ├─ leg_processor.py        # Leg-specific rules (trimmed means, pseudo-reference, linear correction)
│  │     ├─ lung_processor.py       # Lung-specific rules (group means, reference correction; Diagnosis→y)
│  │     └─ registry.py             # Name → Processor mapping (breast/leg/lung)
│  │
│  └─ utils/                        # General utilities used across the codebase
│     ├─ checkpoint.py              # Save/load checkpoints (incl. strict/partial load helpers)
│     ├─ config.py                  # YAML loader, output directory creation, pretty printing
│     ├─ logging.py                 # Logging helpers (levels, formatters, file/console handlers)
│     ├─ math_utils.py              # Numeric helpers (safe ops, pairwise distances, reductions)
│     ├─ reporter.py                # Row-wise CSV writer used during long runs
│     ├─ seed.py                    # Reproducible global seeding across NumPy/PyTorch/random
│     └─ timer.py                   # Simple timers and ETA estimation per stage/step
│
└─ outputs/                         # Auto-created training/evaluation outputs (one subfolder per run)
   ├─ pretrain_*/*                  # Source training logs, best.ckpt, train/val log.csv, config snapshot
   ├─ meta_eval/*                   # Episode-wise metrics (episodes_metrics.csv), summary.txt, log.csv
   └─ episode_viz/*                 # Single-episode figure (PNG) + metrics.{csv,txt} + config snapshot


```

---

## Dependencies

Tested with **Python 3.10**, **CUDA 11.8**, **PyTorch ≥ 2.1**.

- Core: `torch`, `numpy`, `pandas`, `pyyaml`, `tqdm`
- Evaluation / reduction / plots: `scikit-learn`, `matplotlib`
- Optional (visualization): `umap-learn` (UMAP), `scipy` (KDE contours)

> If an optional package is missing, the code falls back gracefully (e.g., UMAP → t-SNE → PCA; contours disabled).

**Install (example)**

```bash
# 1) environment
conda create -n mwr-xloc python=3.10 -y
conda activate mwr-xloc

# 2) PyTorch (choose CUDA build to match your system)
pip install torch==2.1.* --index-url https://download.pytorch.org/whl/cu118

# 3) libraries
pip install numpy pandas pyyaml tqdm scikit-learn matplotlib umap-learn scipy
```
---

## Quick Start

> You can **freely swap any YAML** shown below with another one under `configs/**`.  
> Whenever you want to run a *different* experiment (e.g., other dataset, other loss, other shot size), simply **replace the `--cfg` path** with the YAML that matches your choice.

---

### 1) Data Preparation (Stage-1)

Place the raw tables in `data/raw/` with the **exact file names**:
```text
data/raw/
├─ breast.csv
├─ legs.csv
└─ lungs.csv
```

Then run the preprocessing script with the dataset YAML you want:

```bash
# Breast / Leg / Lung (choose one YAML)
python -m src.scripts.run_preprocess --cfg configs/dataset/leg.yaml

# outputs → data/processed/leg/ (manifest.json, splits.json, <sample_id>/*)
```

- `configs/dataset/*.yaml` controls **reference points**, **per-dataset normalization**, **export paths**, etc.

- To preprocess another dataset, **only change the YAML path**, e.g.:
  - `--cfg configs/dataset/breast.yaml`
  - `--cfg configs/dataset/lung.yaml`













