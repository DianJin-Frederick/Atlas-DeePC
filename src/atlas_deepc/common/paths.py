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


def experiment_path(recorded):
    """Locate retained experiment artifacts after project and rank-panel moves."""
    path = Path(recorded)
    parts = path.parts
    if "experiments" not in parts:
        return path
    relative = Path(*parts[parts.index("experiments"):])
    old = Path("experiments/reacher/four_baselines_projection_chordal_lamdu0_clarabel_v1")
    if relative.is_relative_to(old):
        relative = (Path("experiments/reacher/projection_chordal_rank_lamg0_lamdu0_clarabel_v1/rank060")
                    / relative.relative_to(old))
    local = ROOT / relative
    return local if local.exists() or not path.exists() else path
