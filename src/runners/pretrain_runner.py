# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional
import os, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.algorithms.pretrain import PretrainNet, compute_losses, proto_bootstrap_once
from src.data.datasets.mwr_dataset import MWRGraphDataset
from src.data.collate import collate_graphs
from src.evaluation.metrics import (
    binary_metrics_from_logits, auc_roc_prob
)
from src.evaluation.reporter import Reporter
from src.utils.config import load_yaml, pretty, ensure_dir
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import set_seed


# ---- small helpers ----
def _to_device_batch(b: Dict, device: torch.device) -> Dict:
    out = {}
    def _t(x, dtype=None, long=False):
        if torch.is_tensor(x):
            t = x.to(device)
        else:
            t = torch.as_tensor(x, device=device)
        if long: return t.long()
        if dtype is not None: return t.to(dtype)
        return t

    out["nodes"] = _t(b.get("nodes", b.get("x")), dtype=torch.float32)
    out["edge_index"] = _t(b["edge_index"], long=True)
    # If collate_graphs does not provide batch_ptr, fall back to “single-graph”
    out["batch_ptr"]  = _t(b.get("batch_ptr", [0, int(out["nodes"].size(0))]), long=True)

    if b.get("pairs", None) is not None:
        out["pairs"] = _t(b["pairs"], long=True)
    else:
        out["pairs"] = None

    y = b.get("y", None)
    if y is not None:
        y = _t(y, dtype=torch.float32)
        if y.dim() == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)
        out["y"] = y
    return out


def _build_loader(
    root: str | Path,
    split: str,
    bs: int,
    shuffle: bool,
    device: torch.device,
    *,
    # Auto split-by-ratio configs (passed to MWRGraphDataset.from_cache)
    split_ratios: Optional[Tuple[float, float, float]] = None,
    stratify: bool = True,
    include_unknown_in_split: bool = True,
    keep_unknown: bool = True,
    seed: int = 42,  # Note: this seed is only for Dataset (affects automatic splitting)
    # Optional: DataLoader arguments
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    ds = MWRGraphDataset.from_cache(
        root=root,
        split=split,
        keep_unknown=keep_unknown,
        random_resample=False,  # If you want balanced upsampling for the train set, change to True here
        augment=False,
        seed=seed,                             # ← Random seed used for automatic splitting
        split_ratios=split_ratios,             # Key: will auto-generate splits.json when no split files exist
        stratify=stratify,
        include_unknown_in_split=include_unknown_in_split,
    )
    if pin_memory is None:
        pin_memory = (device.type == "cuda")
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        collate_fn=collate_graphs
    )


def _epoch_pass(
    model: PretrainNet, loader: DataLoader, device: torch.device,
    amp: bool, train: bool, contrastive_cfg: Dict, desc: str
):
    """
    Returns:
      scalars_epoch: {"loss": ...}
      mets:          Binary classification metrics at @thr=0.5 (including AUC)
      logits, y
    """
    model.train(mode=train)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type=="cuda"))
    opt = getattr(model, "_optim", None)

    loss_sum = 0.0; n_seen = 0
    all_logits, all_y = [], []

    it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in it:
        batch = _to_device_batch(batch, device)

        if train and model._warmup_proto:
            # During warmup epochs, perform a one-time prototype bootstrap on the first batch (current impl: only once)
            proto_bootstrap_once(model, batch, momentum=0.0)
            model._warmup_proto = False

        with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type=="cuda")):
            loss, scalars, logits, y, _ = compute_losses(model, batch, contrastive_cfg)

        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if model._clip_grad is not None and model._clip_grad > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), model._clip_grad)
            scaler.step(opt); scaler.update()

        bs = int(y.numel())
        loss_sum += float(scalars["loss"]) * bs
        n_seen   += bs
        all_logits.append(logits); all_y.append(y)

        it.set_postfix(loss=f"{(loss_sum/max(1,n_seen)):.4f}")

    logits = torch.cat(all_logits, dim=0)
    y      = torch.cat(all_y, dim=0).long()
    # Fixed threshold = 0.5
    mets = binary_metrics_from_logits(logits, y, thr=0.5, with_auc=True)
    scalars_epoch = dict(loss=loss_sum/max(1,n_seen))
    return scalars_epoch, mets, logits, y


def _pick_key(mets: Dict[str, float], name: str) -> float:
    """Pick the key metric from the dict; name ∈ {gmean,mcc,acc,auc,spec,sens}"""
    name = (name or "gmean").lower()
    if name not in mets:
        raise KeyError(f"Unknown metric '{name}', available={list(mets.keys())}")
    return float(mets[name])


# ---- main entry ----
def run_from_yaml(cfg_path: str):
    cfg = load_yaml(cfg_path)
    main = cfg.get("main", {})
    model_cfg = cfg.get("model", {})
    ctr_cfg   = cfg.get("contrastive", {})
    data_cfg  = cfg.get("data", {})  # Data configuration section

    # env
    seed = int(main.get("seed", 42))
    set_seed(seed)  # Global random seed for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp = bool(main.get("amp", True))
    out_dir = Path(main.get("out_dir", "outputs/pretrain_breast"))
    ensure_dir(out_dir)

    reporter = Reporter(out_dir)
    reporter.print_env(torch.__version__, torch.cuda.is_available(), str(device))
    reporter.print_config(cfg)

    # data
    data_root = main.get("data_root", "") or os.environ.get("MWR_DATASET_OUT", "data/processed/breast")
    bs = int(main.get("batch_size", 32))

    # ---- Read auto split-by-ratio related configs ----
    split_ratios = data_cfg.get("split_ratios", None)
    if split_ratios is not None:
        if isinstance(split_ratios, (list, tuple)) and len(split_ratios) == 3:
            split_ratios = (float(split_ratios[0]), float(split_ratios[1]), float(split_ratios[2]))
        else:
            raise ValueError("data.split_ratios must be a 3-element list/tuple, e.g., [0.7, 0.15, 0.15]")

    stratify = bool(data_cfg.get("stratify", False))  # ← Default False: purely random split
    include_unknown_in_split = bool(data_cfg.get("include_unknown_in_split", True))
    keep_unknown = bool(data_cfg.get("keep_unknown", True))

    # Seed used specifically for dataset splitting
    split_seed = int(data_cfg.get("split_seed", seed))

    # DataLoader details (optional)
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory_cfg = data_cfg.get("pin_memory", None)
    pin_memory = None if pin_memory_cfg is None else bool(pin_memory_cfg)

    # Loaders (if no split files, the first construction will generate splits.json)
    tr_loader = _build_loader(
        data_root, "train", bs, shuffle=True, device=device,
        split_ratios=split_ratios, stratify=stratify,
        include_unknown_in_split=include_unknown_in_split,
        keep_unknown=keep_unknown, seed=split_seed,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    va_loader = _build_loader(
        data_root, "val", bs, shuffle=False, device=device,
        split_ratios=split_ratios, stratify=stratify,
        include_unknown_in_split=include_unknown_in_split,
        keep_unknown=keep_unknown, seed=split_seed,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    te_loader = _build_loader(
        data_root, "test", bs, shuffle=False, device=device,
        split_ratios=split_ratios, stratify=stratify,
        include_unknown_in_split=include_unknown_in_split,
        keep_unknown=keep_unknown, seed=split_seed,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # model
    model = PretrainNet(
        node_dim=int(model_cfg.get("node_dim", 13)),
        enc_hid=int(model_cfg.get("enc_hid", 64)),
        enc_out=int(model_cfg.get("enc_out", 64)),
        num_heads=int(model_cfg.get("num_heads", model_cfg.get("heads", 4))),
        embed_dim=int(model_cfg.get("embed_dim", 128)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        metric_gamma=float(model_cfg.get("metric_gamma", 10.0)),
    ).to(device)

    # optimizer
    lr = float(main.get("lr", 4e-4))
    wd = float(main.get("weight_decay", 1e-4))
    model._optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model._clip_grad = float(main.get("clip_grad", 0.5))
    model._warmup_proto = (int(main.get("proto_warmup", 1)) > 0)

    # Learning rate scheduler (optional)
    sched_name = str(main.get("sched", "") or "").lower()
    lr_patience = int(main.get("lr_patience", 10))
    min_lr = float(main.get("min_lr", 1e-8))
    scheduler = None
    if sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model._optim, mode="max", factor=0.5, patience=lr_patience,
            threshold=1e-4, min_lr=min_lr
        )

    # Selection metric (default gmean; can be set via YAML main.select_metric)
    select_metric = str(main.get("select_metric", "gmean")).lower()
    reporter.info(f"[model selection] metric={select_metric}, thr(fixed)=0.5")

    # train loop
    epochs = int(main.get("epochs", 200))
    patience = int(main.get("patience", 50))
    best_key = -1.0; best_state = None; best_thr = 0.5; no_improve = 0

    pbar = tqdm(range(1, epochs+1), desc="Epochs", dynamic_ncols=True)
    for ep in pbar:
        tr_scalar, tr_mets, _, _ = _epoch_pass(model, tr_loader, device, amp, True,  ctr_cfg, f"Train {ep:03d}")
        va_scalar, va_mets, va_logits, va_y = _epoch_pass(model, va_loader, device, amp, False, ctr_cfg, f"Val   {ep:03d}")

        # —— Key: use a fixed threshold 0.5 throughout; select the best model by metrics at @0.5 ——
        thr_used = 0.5
        key = _pick_key(va_mets, select_metric)

        reporter.log_epoch(ep, tr_scalar, tr_mets, va_scalar, va_mets,
                           thr_opt=thr_used, key=key, key_name=select_metric, val_thr=thr_used)

        if scheduler is not None:
            scheduler.step(key)  # Scheduler also uses the @0.5 key

        if key > best_key:
            best_key = key; no_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_thr = thr_used  # Fixed at 0.5
            save_checkpoint(
                out_dir / "best.ckpt",
                dict(model=best_state, epoch=ep,
                     best_metric=select_metric, best_score=best_key,
                     best_thr=best_thr, cfg=cfg)
            )
            reporter.info("update the model")
        else:
            no_improve += 1
            if no_improve >= patience:
                reporter.info("Early stop triggered.")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test (also at @0.5)
    te_scalar, te_mets, te_logits, te_y = _epoch_pass(model, te_loader, device, amp, False, ctr_cfg, "Test")
    # Recompute AUC (for safety), can be skipped since _epoch_pass already uses with_auc=True
    te_auc = auc_roc_prob(torch.sigmoid(te_logits), te_y)
    te_mets["auc"] = te_auc
    reporter.log_test(te_scalar, te_mets, thr_used=best_thr)

    reporter.finish()
