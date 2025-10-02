# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from src.runners.pretrain_runner import run_from_yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="configs/pretrain.yaml")
    args = ap.parse_args()
    run_from_yaml(args.cfg)

if __name__ == "__main__":
    main()
