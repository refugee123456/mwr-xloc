# Interpretable Few-Shot Transfer Learning for Microwave Radiometry Diagnosis
**Cross-Location Adaptation with Graph Attention and Prototypical Metrics (MWR-XLoC)**

This repository implements **MWR-XLoC**, an interpretable cross-location few-shot pipeline for **Microwave Radiometry (MWR)** diagnosis.  
We convert raw MWR spreadsheets into **graphs**, encode them with a light **graph-attention/MLP** backbone, and classify via a **prototype distance (metric-softmax)** head.  
At meta-test time (new body location), we adapt with **Instance Recalibration**, optional **VTAN** normalization, and a short **inner-loop fine-tune**. The repo also includes paper-style **2×2 episode visualizations** and complete **per-episode metrics**.

---

## 📌 Highlights

- **Physiology-aware graphs** from tabular MWR: 13-D node features, symmetry pairs, reference-based normalization.
- **Transparent classifier**: Prototypical metric head with learnable temperature γ and adaptive γ (optional).
- **Cross-location adaptation**:
  - **Instance Recalibration** (RBF / Cosine): class-wise instance smoothing on the support set.
  - **VTAN**: query re-projection normalization with optional scaling.
  - **Few-shot fine-tune**: compaction / prototype-margin / CORAL losses (inner loop *with gradients*).
  - **Best-loss snapshot for inference** during fine-tuning (implemented in `PFAMetaAdapter`).
- **Reproducible visualizations**: UMAP/t-SNE/PCA, KDE density, Times fonts, clear legend; outputs `PNG + CSV/TXT`.

---

## 🔎 What’s in this repo?


