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
mwr-xloc/
├─ README.md # How to use the project (Dependencies / Install / Run / Results)
│
├─ configs/ # All YAML configs (organized by stage)
│ ├─ dataset/ # Stage-1: raw table → graph preprocessing
│ │ ├─ breast.yaml # Breast: reference points / node construction / export dir
│ │ ├─ leg.yaml # Leg: trimmed mean / number of points / export dir
│ │ └─ lung.yaml # Lung: reference points / export dir
│ │
│ ├─ pretrain/ # Stage-2: source-domain pretraining (CE + optional contrastive)
│ │ ├─ pretrain_none.yaml # CE only (MetricHead)
│ │ ├─ pretrain_contrastive.yaml # Siamese margin
│ │ ├─ pretrain_triplet_hard.yaml # Triplet-hard
│ │ ├─ pretrain_triplet_semi.yaml # Triplet semi-hard
│ │ └─ pretrain_npairs.yaml # N-pairs
│ │
│ ├─ meta_test/ # Stage-3: cross-domain few-shot meta-test (target domain)
│ │ ├─ meta_test_fintune/ # (D) FT + Recal (+VTAN) — VTAN enabled by default
│ │ │ └─ metatest_{breast,leg,lung}{3,5,10}shot.yaml
│ │ ├─ meta_test_without_vtan/ # (C) FT + Recal (no VTAN)
│ │ │ └─ metatest{breast,leg,lung}{3,5,10}shot.yaml
│ │ ├─ simple_meta_test/ # (B) No-FT + Recal (+VTAN)
│ │ │ └─ metatest{breast,leg,lung}{3,5,10}shot.yaml
│ │ └─ simple_meta_test_without_VATN/ # (A) No-FT + Recal (no VTAN)
│ │ └─ metatest{breast,leg,lung}{3,5,10}shot.yaml
│ │
│ └─ episode_viz/ # Single-episode visualization (paper figure)
│ └─ leg_episode_5shot.yaml
│
├─ data/
│ ├─ raw/ # Raw CSV/Excel files you provide
│ │ ├─ breast.csv │ legs.csv │ lungs.csv
│ └─ processed/ # Graph datasets after preprocessing
│ ├─ breast/ | leg/ | lung/
│ │ ├─ manifest.json # Sample list (y, path, #nodes, etc.)
│ │ ├─ splits.json # If no split* files found, splits are auto-generated (stratifiable)
│ │ └─ <sample_id>/
│ │ ├─ nodes.npy # [N,13] node features
│ │ ├─ edge_index.npy # [2,E] graph edges
│ │ ├─ pairs.npy # [M,2] left-right mirror index pairs
│ │ └─ meta.json # sample metadata
│
├─ src/ # —— Core code —— (three-stage pipeline)
│ ├─ scripts/ # CLI entrypoints
│ │ ├─ run_preprocess.py # Read configs/dataset/* → create processed graphs
│ │ ├─ run_pretrain.py # Read configs/pretrain/* → train & save best.ckpt
│ │ ├─ run_metatest.py # Read configs/meta_test//.yaml → few-shot evaluation
│ │ └─ plot_episode_viz.py # Single episode viz (2×2 panels for the four settings + CSV/TXT)
│ │
│ ├─ runners/ # Orchestration
│ │ ├─ pretrain_runner.py # Data loading, training loop, early stop, best.ckpt saving
│ │ └─ meta_eval_runner.py # Episode sampling, PFAMetaAdapter, metric aggregation
│ │
│ ├─ algorithms/
│ │ ├─ pretrain.py # PretrainNet: GraphEncoder → Local/Sym/Global → Fusion → MetricHead
│ │ ├─ pfa.py # Lightweight adaptation: recalibration (RBF/cos), weighted protos,
│ │ │ # VTAN normalization, adaptive gamma, distance-softmax logits
│ │ └─ adaptation/
│ │ └─ pfa_finetuner.py # PFAMetaAdapter: inner-loop fine-tuning on Support + inference on Query
│ │ # - compactness, prototype margin, CORAL alignment, recalibration, VTAN, BN-Eval
│ │
│ ├─ models/
│ │ ├─ encoders/graph_encoder.py # Graph encoder (multi-head attention / MLP stacks)
│ │ ├─ aggregators/ # Three branches + fusion
│ │ │ ├─ local_branch.py
│ │ │ ├─ sym_branch.py
│ │ │ ├─ global_branch.py
│ │ │ └─ fusion.py
│ │ ├─ heads/metric_head.py # Distance-softmax classifier (learnable gamma)
│ │ └─ layers/{resnet_mlp.py, norm.py}
│ │
│ ├─ losses/
│ │ ├─ metric.py # metric_ce_loss (distance-softmax CE)
│ │ └─ contrastive.py # Siamese / Triplet / N-pairs losses
│ │
│ ├─ evaluation/
│ │ ├─ metrics.py # ACC / SPEC / SENS / G-mean / MCC / AUC
│ │ └─ reporter.py # Console + CSV logger
│ │
│ ├─ data/
│ │ ├─ datasets/mwr_dataset.py # Load manifest; auto-build splits if needed
│ │ ├─ collate.py # Batch big-graph collation (nodes/edge_index/pairs/batch_ptr)
│ │ ├─ episodic_sampler.py # Few-shot episode sampler (balanced support)
│ │ └─ processing/ # Stage-1 implementations
│ │ ├─ base_processor.py │ ambient_norm.py │ graph_builder.py │ node_features_common.py
│ │ ├─ breast_processor.py │ leg_processor.py │ lung_processor.py
│ │ └─ registry.py
│ │
│ └─ utils/ # Generic utilities (your current set of files)
│ ├─ checkpoint.py # Save/load checkpoints (best.ckpt, etc.)
│ ├─ config.py # load_yaml / ensure_dir / pretty printing
│ ├─ logging.py # Unified logging (file/console levels & formats)
│ ├─ math_utils.py # Numeric helpers (stable ops, distances, etc.)
│ ├─ reporter.py # Step-wise metric recording to CSV (used by evaluation.reporter)
│ ├─ seed.py # Global seeding (PyTorch/NumPy/random)
│ └─ timer.py # Timing / ETA utilities (stage & step granularity)
│
└─ outputs/ # Auto-created training/test outputs
├─ pretrain_/ # Train logs, best.ckpt, log.csv
├─ meta_eval/* # Meta-test summary: episodes_metrics.csv, episodes_summary.txt, log.csv
└─ episode_viz/* # Single-episode viz: PNG + metrics.csv/txt
