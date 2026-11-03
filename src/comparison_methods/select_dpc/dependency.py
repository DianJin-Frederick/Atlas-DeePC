"""Standard-library-only discovery of the optional external comparison runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
URL = "https://github.com/naefjo/choose-wisely-paper.git"
COMMIT = "8afa3b36ca47570400b0b8ff82328ab25ee3fdee"
HOME_VARIABLE = "ATLAS_DEEPC_SELECT_DPC_HOME"


def home():
    return Path(os.environ.get(HOME_VARIABLE, Path.home() / ".local/share/atlas-deepc/select-dpc")).expanduser().absolute()


def source_dir():
    return home() / "source"


def interpreter():
    return home() / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def setup_hint():
    return "Run: python3.10 scripts/setup_select_dpc.py (from atlas_deepc/code/)"


def verify_source(source=None):
    source = source or source_dir()
    if not (source / ".git").exists():
        raise RuntimeError(f"Optional Select-DPC checkout is missing at {source}. {setup_hint()}")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != COMMIT or dirty:
        raise RuntimeError("Select-DPC must use the pinned, unmodified upstream revision; existing files were not changed.")
    return revision


def require_installation():
    marker = home() / "installation.json"
    if not interpreter().is_file() or not marker.is_file():
        raise RuntimeError(f"Select-DPC is optional and not installed at {home()}. {setup_hint()}")
    record = json.loads(marker.read_text())
    requirements = ROOT / record["requirements"]
    if record["commit"] != COMMIT or record["requirements_sha256"] != hashlib.sha256(requirements.read_bytes()).hexdigest():
        raise RuntimeError(f"Select-DPC setup is outdated. {setup_hint()}")
    verify_source()
    return interpreter()
