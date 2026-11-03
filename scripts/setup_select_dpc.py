"""Install/check optional Select-DPC outside this repository. Never runs a study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from comparison_methods.select_dpc import dependency


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def check(python=None):
    python = python or dependency.require_installation()
    code = """import importlib.metadata as m, json, sys
import cvxpy, gymnasium, mujoco, numpy, scipy, torch, sklearn
assert sys.prefix != sys.base_prefix
assert 'CLARABEL' in cvxpy.installed_solvers()
from comparison_methods.select_dpc.upstream import load_upstream
dc, args, scheduler = load_upstream()
assert (args.deepc_dims.p, args.deepc_dims.m, args.deepc_dims.T_past, args.deepc_dims.T_fut) == (8, 2, 2, 15)
print(json.dumps({'python': sys.executable, 'packages': {p:m.version(p) for p in ('numpy','scipy','cvxpy','clarabel','gymnasium','mujoco','torch')}, 'upstream_imports': 'passed'}))
"""
    run([python, "-c", code], env=runtime_env())
    run([python, "-m", "pip", "check"])


def runtime_env():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONNOUSERSITE="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, help="External source/environment location; also set ATLAS_DEEPC_SELECT_DPC_HOME when running notebooks")
    parser.add_argument("--python", default=sys.executable, help="Python 3.10 used to create the isolated environment")
    parser.add_argument("--check", action="store_true", help="Validate existing installation without downloading or running rollouts")
    args = parser.parse_args()
    if args.home:
        os.environ[dependency.HOME_VARIABLE] = str(args.home.expanduser().absolute())
    if dependency.home().resolve().is_relative_to(ROOT.parent):
        raise ValueError("Select an installation location outside the Atlas-DeePC project.")
    if args.check:
        check()
        return
    details = json.loads(subprocess.check_output([args.python, "-c", "import sys,platform,json; print(json.dumps([list(sys.version_info[:2]),sys.platform,platform.machine()]))"], text=True))
    if details[0] != [3, 10]:
        raise RuntimeError("Use Python 3.10 to match the saved comparison runtime.")
    requirements = ROOT / ("configs/select_dpc_macos_arm64_py310.lock" if details[1:] == ["darwin", "arm64"] else "requirements-select-dpc.txt")
    if details[1:] != ["darwin", "arm64"]:
        print("Using pinned direct dependencies; this platform has not been validated against the macOS results.", flush=True)
    dependency.home().mkdir(parents=True, exist_ok=True)
    # Lock the setup directory, never overwrite or reset another checkout.
    import fcntl
    with (dependency.home() / "setup.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = dependency.source_dir()
        if not source.exists():
            run(["git", "-c", "gc.auto=0", "init", source])
            run(["git", "-C", source, "remote", "add", "origin", dependency.URL])
            run(["git", "-C", source, "-c", "gc.auto=0", "fetch", "--depth", "1", "origin", dependency.COMMIT])
            run(["git", "-C", source, "checkout", "--detach", dependency.COMMIT])
        dependency.verify_source()
        environment = dependency.home() / "venv"
        if environment.exists() and not (environment / "pyvenv.cfg").is_file():
            raise RuntimeError(f"Refusing to overwrite an unrelated directory: {environment}")
        if not environment.exists():
            run([args.python, "-m", "venv", environment])
        if "include-system-site-packages = false" not in (environment / "pyvenv.cfg").read_text().lower():
            raise RuntimeError("Comparison environment must not inherit system packages.")
        python = dependency.interpreter()
        run([python, "-m", "pip", "install", "--only-binary=:all:", "--report", dependency.home() / "pip-install-report.json", "-r", requirements])
        run([python, "-m", "pip", "check"])
        record = dict(commit=dependency.COMMIT, upstream_url=dependency.URL,
                      requirements=str(requirements.relative_to(ROOT)),
                      requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
                      python=str(python), platform=details)
        # Publish the installation marker only after import and dataset checks pass.
        check(python)
        temporary = dependency.home() / "installation.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(dependency.home() / "installation.json")
    print("Optional comparison runtime is ready. No experiment rollouts were started.")


if __name__ == "__main__":
    main()
