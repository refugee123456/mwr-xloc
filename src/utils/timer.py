# -*- coding: utf-8 -*-
"""
Simple timing helpers: Stopwatch, AverageMeter, ETA.
"""

from __future__ import annotations
import time
from dataclasses import dataclass


class Stopwatch:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start = time.time()
        self._last = self._start

    def lap(self) -> float:
        now = time.time()
        dt = now - self._last
        self._last = now
        return dt

    def total(self) -> float:
        return time.time() - self._start


@dataclass
class AverageMeter:
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    cnt: int = 0

    def update(self, v: float, n: int = 1) -> None:
        self.val = v
        self.sum += v * n
        self.cnt += n
        self.avg = self.sum / max(1, self.cnt)


class ETA:
    def __init__(self, total_steps: int) -> None:
        self.total = total_steps
        self.done = 0
        self.t0 = time.time()

    def step(self, n: int = 1) -> str:
        self.done += n
        elapsed = max(1e-6, time.time() - self.t0)
        rate = self.done / elapsed
        remain = (self.total - self.done) / max(1e-6, rate)
        return f"{remain:,.1f}s remaining @ {rate:.2f} it/s"
