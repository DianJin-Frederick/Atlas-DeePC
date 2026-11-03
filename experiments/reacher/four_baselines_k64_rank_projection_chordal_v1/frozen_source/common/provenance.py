"""Read-only historical reports are distinct from strict experiment resumption."""

import warnings
from pathlib import Path

from atlas_deepc.common.files import file_digest


SOURCE_FIELDS = frozenset({"source_sha256", "dependency_sha256", "runner_sha256", "sources"})


def validate_report_signature(saved, current):
    """Require the same scientific settings/data; disclose changed code for reports.

    Never use this to authorize a new rollout or cache reuse. Experiment runners
    retain their exact signature comparisons.
    """
    old = {k: v for k, v in saved.items() if k not in SOURCE_FIELDS}
    new = {k: v for k, v in current.items() if k not in SOURCE_FIELDS}
    if old != new:
        raise ValueError("Saved report settings, data or numerical environment differ")
    if saved != current:
        warnings.warn("Historical results: source code has changed since these runs. "
                      "Viewing saved results only; current-code reproduction is not verified.",
                      RuntimeWarning, stacklevel=2)


def validate_report_files(checks):
    """Verify data exactly and disclose source drift without claiming reproducibility."""
    changed = []
    for name, expected in checks.items():
        path = Path(name)
        if file_digest(path) == expected:
            continue
        if path.suffix != ".py":
            raise ValueError(f"Historical data/artifact hash mismatch: {path}")
        changed.append(str(path))
    if changed:
        warnings.warn("Historical source files differ from the recorded run: " + ", ".join(changed)
                      + ". Saved data/results remain unchanged; do not resume into this archive.",
                      RuntimeWarning, stacklevel=2)
    return changed
