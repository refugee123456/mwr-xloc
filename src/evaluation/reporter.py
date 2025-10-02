# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
from pathlib import Path
import csv, json, time

class Reporter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "log.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            
            cw.writerow([
                "epoch","phase","loss","acc","spec","sens","gmean","mcc","auc",
                "thr_opt","key","key_name","val_thr"
            ])

    # console helpers
    def info(self, msg: str): print(f"[info] {msg}")
    def print_env(self, torch_ver: str, cuda: bool, device: str):
        self.info(f"torch={torch_ver}, cuda={cuda}, device={device}")
    def print_config(self, cfg: Dict):
        print("[cfg] " + json.dumps(cfg, ensure_ascii=False, indent=2))

    def log_epoch(self, ep: int,
                  tr_scalar: Dict[str,float], tr_mets: Dict[str,float],
                  va_scalar: Dict[str,float], va_mets: Dict[str,float],
                  thr_opt: float, key: float, key_name: str = "gmean", val_thr: float = 0.5):
        # console（明确标注 val@thr）
        print(
            f"[epoch {ep:03d}] "
            f"train_loss={tr_scalar['loss']:.4f} | "
            f"val_loss={va_scalar['loss']:.4f} | "
            f"val@thr={val_thr:.3f}: acc={va_mets['acc']:.4f} spec={va_mets['spec']:.4f} "
            f"sens={va_mets['sens']:.4f} gmean={va_mets['gmean']:.4f} "
            f"mcc={va_mets['mcc']:.4f} auc={va_mets.get('auc',0.0):.4f} | "
            f"thr_used={thr_opt:.3f} key({key_name})={key:.4f}"
        )
        # csv
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow([ep,"train",tr_scalar['loss'],"","","","","","","","","",""])
            cw.writerow([ep,"val",va_scalar['loss'],va_mets['acc'],va_mets['spec'],
                        va_mets['sens'],va_mets['gmean'],va_mets['mcc'],va_mets.get('auc',0.0),
                        thr_opt,key,key_name,val_thr])

    def log_test(self, te_scalar: Dict[str,float], te_mets: Dict[str,float], thr_used: float = 0.5):
        print(
            f"[TEST] loss={te_scalar['loss']:.4f} | "
            f"acc={te_mets['acc']:.4f} spec={te_mets['spec']:.4f} "
            f"sens={te_mets['sens']:.4f} gmean={te_mets['gmean']:.4f} "
            f"mcc={te_mets['mcc']:.4f} auc={te_mets.get('auc',0.0):.4f} | "
            f"thr_used={thr_used:.3f}"
        )
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow(["TEST","test",te_scalar['loss'],te_mets['acc'],te_mets['spec'],
                        te_mets['sens'],te_mets['gmean'],te_mets['mcc'],te_mets.get('auc',0.0),
                        thr_used,"","test","{:.3f}".format(thr_used)])

    def finish(self):
        self.info(f"training finished; logs @ {self.csv_path}")
