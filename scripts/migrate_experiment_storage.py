"""Move saved bulky artifacts to an SSD, preserving existing paths with links.

No experiment is run. Offline inputs, source, and current notebooks stay local.
The optional experiment-json scope moves JSON result records too. Separate plans
and append-only checksum journals support interrupted moves.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / ".external_experiment_data"
PLAN = ROOT / "docs/storage_migration_plan.json"
JOURNAL = ROOT / "docs/storage_migration_checksums.jsonl"
DEFAULT_DESTINATION = Path("/Volumes/Crucial X10/atlas_deepc_data")
BULKY = {".npz", ".npy", ".joblib", ".mp4", ".mov", ".zip"}
ARCHIVED = BULKY | {".ipynb", ".png", ".pdf", ".svg"}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def stamp(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino)


def append_record(journal, record):
    with journal.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def relocate(source, target, link_target, journal, relative):
    if source.is_symlink():
        if source.resolve() != target.resolve() or not target.is_file():
            raise RuntimeError(f"Unexpected or unavailable link: {source}")
        return
    before = stamp(source)
    checksum = digest(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise RuntimeError(f"Refusing destination symlink: {target}")
    if target.exists():
        if digest(target) != checksum:
            raise RuntimeError(f"Destination conflict; original retained: {target}")
    else:
        fd, temporary = tempfile.mkstemp(prefix=".copy-", dir=target.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as output, source.open("rb") as original:
                shutil.copyfileobj(original, output, 4 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if digest(temporary) != checksum:
                raise RuntimeError(f"Copy checksum mismatch: {source}")
            # Hard links are not available on exFAT. The destination must be unused.
            if target.exists():
                raise RuntimeError(f"Destination appeared during copy: {target}")
            temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)
    if stamp(source) != before or digest(source) != checksum:
        raise RuntimeError(f"Source changed during copy; original retained: {source}")
    if digest(target) != checksum:
        raise RuntimeError(f"Destination verification failed: {target}")
    append_record(journal, dict(path=relative, bytes=before[0], sha256=checksum))
    # Replace the local original only after a durable verified copy and journal.
    temporary_link = source.with_name(source.name + ".ssd-link")
    if temporary_link.exists() or temporary_link.is_symlink():
        if not temporary_link.is_symlink() or os.readlink(temporary_link) != link_target:
            raise RuntimeError(f"Unexpected temporary link: {temporary_link}")
        temporary_link.unlink()
    temporary_link.symlink_to(link_target)
    temporary_link.replace(source)


def create_plan(destination, scope="bulky"):
    entries = []
    folders = ("experiments",) if scope == "experiment-json" else (
        "experiments/reacher", "experiments/f1tenth", "archive")
    for folder in folders:
        allowed = {".json"} if scope == "experiment-json" else (ARCHIVED if folder == "archive" else BULKY)
        for base, directories, files in os.walk(ROOT / folder):
            directories[:] = sorted(d for d in directories
                                    if not (Path(base) / d).is_symlink())
            for name in sorted(files):
                path = Path(base) / name
                if path.is_symlink() or path.suffix not in allowed:
                    continue
                entries.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size))
    local_inputs = {str(p.relative_to(ROOT)): digest(p)
                    for p in (ROOT / "data").rglob("*") if p.is_file() and not p.is_symlink()}
    return dict(destination=str(destination), entries=entries, offline_inputs=local_inputs)


def require_ssd(destination):
    volume = Path("/Volumes/Crucial X10")
    if not volume.is_mount() or not destination.is_relative_to(volume):
        raise RuntimeError("Connect Crucial X10; refusing a missing or different volume.")
    if destination.resolve() != destination:
        raise RuntimeError("Destination must not traverse symlinks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "migrate", "verify", "status"))
    parser.add_argument("--scope", choices=("bulky", "experiment-json"), default="bulky",
                        help="Use a separate resumable plan for JSON experiment records")
    args = parser.parse_args()
    plan_path, journal_path = PLAN, JOURNAL
    if args.scope == "experiment-json":
        plan_path = ROOT / "docs/storage_json_migration_plan.json"
        journal_path = ROOT / "docs/storage_json_migration_checksums.jsonl"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        if args.action != "plan":
            raise RuntimeError("Create a migration plan first.")
        plan = create_plan(DEFAULT_DESTINATION, args.scope)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    destination = Path(plan["destination"])
    entries = plan["entries"]
    for row in entries:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or path.parts[0] not in {"experiments", "archive"}:
            raise RuntimeError(f"Invalid planned path: {path}")
    if args.action in {"plan", "status"}:
        linked = sum((ROOT / row["path"]).is_symlink() for row in entries)
        print(json.dumps(dict(files=len(entries), bytes=sum(r["bytes"] for r in entries),
                              linked=linked, destination=str(destination),
                              ssd_connected=destination.is_dir()), indent=2))
        return
    require_ssd(destination)
    if args.action == "migrate":
        remaining = sum(r["bytes"] for r in entries if not (ROOT / r["path"]).is_symlink())
        if shutil.disk_usage(destination.parent).free < remaining + 1024**3:
            raise RuntimeError("Insufficient SSD space.")
        destination.mkdir(parents=True, exist_ok=True)
        if ANCHOR.is_symlink():
            if ANCHOR.resolve() != destination:
                raise RuntimeError("Existing storage anchor points elsewhere.")
        elif ANCHOR.exists():
            raise RuntimeError("Storage anchor already exists and is not a link.")
        else:
            ANCHOR.symlink_to(destination, target_is_directory=True)
        for index, row in enumerate(entries, 1):
            source = ROOT / row["path"]
            target = destination / row["path"]
            if not source.is_symlink() and source.stat().st_size != row["bytes"]:
                raise RuntimeError(f"File changed since planning: {source}")
            relocate(source, target, os.path.relpath(ANCHOR / row["path"], source.parent),
                     journal_path, row["path"])
            if index % 100 == 0 or index == len(entries):
                print(f"Migrated {index}/{len(entries)} files", flush=True)
    records = {r["path"]: r for r in map(json.loads, journal_path.read_text().splitlines())}
    for index, row in enumerate(entries, 1):
        source, target = ROOT / row["path"], destination / row["path"]
        if not source.is_symlink() or source.resolve() != target:
            raise RuntimeError(f"Missing storage link: {source}")
        if target.stat().st_size != row["bytes"] or digest(target) != records[row["path"]]["sha256"]:
            raise RuntimeError(f"Checksum verification failed: {target}")
        if index % 500 == 0:
            print(f"Verified {index}/{len(entries)} files", flush=True)
    for relative, checksum in plan["offline_inputs"].items():
        path = ROOT / relative
        if path.is_symlink() or digest(path) != checksum:
            raise RuntimeError(f"Offline input changed: {path}")
    for local in (plan_path, journal_path):
        shutil.copyfile(local, destination / local.name)
    print(f"Verified all {len(entries)} external files and {len(plan['offline_inputs'])} unchanged local inputs.")


if __name__ == "__main__":
    main()
