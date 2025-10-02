# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import time, os

class Reporter:
    def __init__(self, out_dir: str):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.log_file = self.out / "train_log.txt"; self.t0 = time.time()

    def _append(self, s: str):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    def _fmt_m(self, m: Dict[str, float]) -> str:
        return ("acc={acc:.4f} spec={spec:.4f} sens={sens:.4f} "
                "gmean-loss={gmean_loss:.4f} mcc={mcc:.4f} auc={auc:.4f}").format(**m)

    def print_env(self, env: Dict[str, Any]):
        line = "[env] " + ", ".join(f"{k}={v}" for k, v in env.items())
        print(line); self._append(line)

    def print_cfg(self, cfg: Dict[str, Any]):
        for k, v in cfg.items():
            print(f"[cfg] {k}:"); self._append(f"[cfg] {k}:")
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f"  - {kk}: {vv}"); self._append(f"  - {kk}: {vv}")
            else:
                print("  ", v); self._append("  " + str(v))

    def log_epoch(self, ep: int, tr_loss: Dict[str, float], tr_m: Dict[str, float],
                  va_loss: Dict[str, float], va_m: Dict[str, float], best_mcc: float):
        line = (f"[epoch {ep:03d}] "
                f"train[total={tr_loss['total']:.4f} ce={tr_loss['ce']:.4f} ctr={tr_loss['ctr']:.4f} | {self._fmt_m(tr_m)}] | "
                f"val[total={va_loss['total']:.4f} ce={va_loss['ce']:.4f} ctr={va_loss['ctr']:.4f} | {self._fmt_m(va_m)}] | "
                f"best-mcc={best_mcc:.4f}")
        print(line); self._append(line)

    def log_test(self, loss: Dict[str, float], m: Dict[str, float]):
        line = f"[TEST] total={loss['total']:.4f} ce={loss['ce']:.4f} ctr={loss['ctr']:.4f} | {self._fmt_m(m)}"
        print(line); self._append(line)

    def save_ckpt(self, path: str, payload: Dict[str, Any]):
        import torch, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)
        self._append(f"[ckpt] saved to {path}")
