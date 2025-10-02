# -*- coding: utf-8 -*-
import argparse

import sys
sys.path.append('/root/autodl-tmp/mwr-xloc')
from src.runners.meta_eval_runner import run_from_yaml




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="path to yaml")
    args = ap.parse_args()
    run_from_yaml(args.cfg)

if __name__ == "__main__":
    main()
