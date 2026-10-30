"""Inventory local dependencies without executing notebooks or experiments."""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODULE_MAP = json.loads((SRC / "atlas_deepc/module_map.json").read_text())


def source_file(name):
    return SRC / MODULE_MAP.get(Path(name).stem, name)
REPRODUCTION_ENTRY_POINTS = {
    "targeted_search.py", "selected_rank_1000_clarabel.py",
    "ablation_nearest_terminal_reference.py",
    "render_measurement_norm_noise_videos.py",
}
SUPERSEDED_BUILDERS = {
    "build_baseline_contextual_atlas_notebook.py", "build_contextual_cluster_notebook.py",
    "build_basis_routing_ablation_notebook.py", "build_baseline_waypoint_notebook.py",
    "update_comparison_with_subspace.py", "update_comparison_with_iterative_reference.py",
    "build_four_method_comparison_notebook.py",
}


def sources(path):
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text())
        return ["".join(c.get("source", [])) for c in notebook["cells"]]
    return [path.read_text(errors="replace")]


def local_dependencies(text):
    dependencies = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                parts = name.split(".")
                path = SRC.joinpath(*parts).with_suffix(".py")
                if path.is_file():
                    dependencies.add(path)
                package = SRC / parts[0]
                if (package / "__init__.py").is_file():
                    dependencies.update(package.rglob("*.py"))
    # Subprocess entry points and source-signature checks also need their files.
    for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*\.py", text):
        if source_file(name).is_file():
            dependencies.add(source_file(name))
    return dependencies


def main():
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    notebooks += sorted(ROOT.glob("*.ipynb"))
    roots = set(notebooks)
    roots.update((ROOT / "notebooks/motivation").glob("*.py"))
    roots.update(source_file(name) for name in REPRODUCTION_ENTRY_POINTS if source_file(name).exists())
    retained = set()
    reasons = {}
    pending = list(roots)
    while pending:
        path = pending.pop()
        if path in retained:
            continue
        retained.add(path)
        for source in sources(path):
            try:
                dependencies = local_dependencies(source)
            except SyntaxError:
                dependencies = set()
            for dependency in dependencies:
                reasons.setdefault(str(dependency.relative_to(ROOT)), []).append(str(path.relative_to(ROOT)))
                pending.append(dependency)
    # Preserve generators of active notebooks and the maintained test suite.
    active_names = {p.name for p in notebooks}
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        if path.name in SUPERSEDED_BUILDERS:
            continue
        if path.name.startswith(("build_", "update_", "execute_")) and any(n in text for n in active_names):
            pending.append(path)
        elif path.name.startswith("test_"):
            deps = local_dependencies(text)
            if deps:
                pending.append(path)
    while pending:
        path = pending.pop()
        if path in retained:
            continue
        retained.add(path)
        for source in sources(path):
            for dependency in local_dependencies(source):
                reasons.setdefault(str(dependency.relative_to(ROOT)), []).append(str(path.relative_to(ROOT)))
                pending.append(dependency)
    text_by_path = {str(p.relative_to(ROOT)): "\n".join(sources(p)) for p in retained}
    website = ROOT.parent / "website" if ROOT.name == "code" else ROOT / "website"
    for path in website.glob("*.json"):
        text_by_path[str(path.relative_to(website.parent))] = path.read_text()
    experiment_dirs = []
    for parent in [ROOT / "experiments/reacher", ROOT / "experiments/f1tenth",
                   ROOT / "experiments/reacher/routed_deepc_iid"]:
        if not parent.exists():
            continue
        for path in sorted(parent.iterdir()):
            if path.is_dir():
                refs = [name for name, text in text_by_path.items() if path.name in text]
                experiment_dirs.append({"path": str(path.relative_to(ROOT)), "references": sorted(refs)})
    report = {
        "active_notebooks": [str(p.relative_to(ROOT)) for p in notebooks],
        "retained_code": sorted(str(p.relative_to(ROOT)) for p in retained if p.suffix == ".py"),
        "unreferenced_src": sorted(str(p.relative_to(ROOT)) for p in SRC.rglob("*.py") if p not in retained),
        "code_reasons": reasons,
        "experiment_directories": experiment_dirs,
        "note": "Static candidates only. Inspect indirect artifact paths before deleting anything.",
    }
    output = ROOT / "docs/notebook_dependency_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"notebooks": len(notebooks), "retained_code": len(report["retained_code"]),
                      "unreferenced_src": report["unreferenced_src"],
                      "unreferenced_directories": [r["path"] for r in experiment_dirs if not r["references"]]}, indent=2))


if __name__ == "__main__":
    main()
