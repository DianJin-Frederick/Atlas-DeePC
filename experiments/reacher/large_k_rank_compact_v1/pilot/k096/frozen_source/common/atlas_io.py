"""Shared NPZ serialization for atlas arrays and their recorded configuration."""

from dataclasses import asdict
import json
import numpy as np


def save_atlas(path, atlas, config):
    np.savez(path, **atlas, config=np.array(json.dumps(asdict(config)), dtype=object))


def load_atlas(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files if key != "config"}
