"""Project paths and relocated source lookup; never substitute historical code."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_MODULES = json.loads((Path(__file__).resolve().parents[1] / "module_map.json").read_text())


def project_root():
    return ROOT


def source_path(name):
    """Find the current source for a legacy filename used in manifest keys."""
    stem = Path(name).stem
    if stem not in _MODULES:
        # A notebook kernel may outlive an update to the module registry.
        _MODULES.update(json.loads((Path(__file__).resolve().parents[1] / "module_map.json").read_text()))
        if stem not in _MODULES:
            raise KeyError(f"Unknown Atlas-DeePC source: {name}")
    return ROOT / "src" / _MODULES[stem]


def source_hashes(names):
    """Include shared helpers when signing newly generated experiment caches."""
    from atlas_deepc.common.files import file_digest
    paths = {name: source_path(name) for name in names}
    paths.update({f"common/{p.name}": p for p in Path(__file__).parent.glob("*.py")})
    return {name: file_digest(path) for name, path in paths.items()}
