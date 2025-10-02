# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Type
from .base_processor import BaseProcessor
from .breast_processor import BreastProcessor
from .leg_processor import LegProcessor
from .lung_processor import LungProcessor

REGISTRY: Dict[str, Type[BaseProcessor]] = {
    "breast": BreastProcessor,
    "lung":  LungProcessor,  
    "leg":   LegProcessor,
}

def get_processor(name: str) -> BaseProcessor:
    if name not in REGISTRY:
        raise KeyError(f"Unknown processor: {name}")
    return REGISTRY[name]()
