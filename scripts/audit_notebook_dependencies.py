"""Screen notebook dependencies without executing experiments or reading SSD data.

Follow imports, package initializers, literal subprocess modules, and source-file
references. Candidates require review: arbitrary dynamic imports and callbacks
cannot be proved unused by static analysis. This tool never deletes files.
"""
import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRECTORIES = ("src", "scripts", "notebooks", "output", "exports", "tmp")
SUPPORT_SCRIPTS = (
    "setup_select_dpc.py", "run_select_dpc.py", "collect_f1tenth_offline_data.py",
    "migrate_experiment_storage.py",
    "verify_notebook_reports.py", "audit_notebook_dependencies.py",
)


def inventory(root):
    paths = set()
    for name in CODE_DIRECTORIES:
        directory = root / name
        if not directory.is_symlink():
            paths.update(p for p in directory.rglob("*.py") if not p.is_symlink())
    paths.update(p for p in (root / "notebooks").rglob("*.ipynb") if not p.is_symlink())
    return paths


def sources(path):
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text())["cells"]
        return (["".join(c.get("source", [])) for c in cells if c["cell_type"] == "code"],
                "\n".join("".join(c.get("source", [])) for c in cells))
    text = path.read_text()
    return [text], text


def parse(source):
    try:
        return ast.parse(source)
    except SyntaxError:
        from IPython.core.inputtransformer2 import TransformerManager
        return ast.parse(TransformerManager().transform_cell(source))


def dependency_graph(root, paths):
    modules, stems, filenames = {}, defaultdict(set), defaultdict(set)
    for path in paths:
        if path.suffix != ".py":
            continue
        stems[path.stem].add(path)
        filenames[path.name].add(path)
        if path.is_relative_to(root / "src"):
            name = ".".join(path.relative_to(root / "src").with_suffix("").parts)
            modules[name.removesuffix(".__init__")] = path
    graph, missing = {}, {}
    for path in sorted(paths):
        dependencies, unresolved = set(), set()

        def add_module(name, required=False):
            if name in modules:
                dependencies.add(modules[name])
            elif "." not in name:
                dependencies.update(stems.get(name, ()))
            elif required and name.startswith(("atlas_deepc.", "comparison_methods.")):
                unresolved.add(name)
            parts = name.split(".")
            for end in range(1, len(parts)):
                parent = modules.get(".".join(parts[:end]))
                if parent is not None and parent.name == "__init__.py":
                    dependencies.add(parent)

        chunks, text = sources(path)
        for chunk in chunks:
            for node in ast.walk(parse(chunk)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        add_module(alias.name, required=True)
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        package = list(path.relative_to(root / "src").parent.parts)
                        base = ".".join(package[:len(package)-node.level+1] + ([base] if base else []))
                    add_module(base, required=True)
                    for alias in node.names:
                        name = base + "." + alias.name
                        add_module(name)
                        parent = modules.get(base)
                        if (parent and parent.name == "__init__.py" and name not in modules
                                and alias.name != "*" and alias.name not in parent.read_text()):
                            unresolved.add(name)
        # CLI/importlib paths and source signatures are dependencies too.
        for name in re.findall(r"(?:atlas_deepc|comparison_methods)(?:\.[A-Za-z_]\w*)+", text):
            add_module(name)
        for name in re.findall(r"\b[A-Za-z_]\w*\.py\b", text):
            if name != "__init__.py":
                dependencies.update(filenames.get(name, ()))
        if path.is_relative_to(root / "src"):
            dependencies.update(parent / "__init__.py" for parent in path.parents
                                if parent / "__init__.py" in paths)
        # source_hashes signs all common helpers, not only imported helpers.
        if path == root / "src/atlas_deepc/common/paths.py":
            dependencies.update(p for p in paths if p.parent == path.parent)
        dependencies.discard(path)
        graph[path] = dependencies
        if unresolved:
            missing[path] = unresolved
    return graph, modules, missing


def closure(graph, roots):
    seen, pending = set(), list(roots)
    while pending:
        path = pending.pop()
        if path not in seen:
            seen.add(path)
            pending.extend(graph.get(path, set()) - seen)
    return seen


def process_modules():
    """Protect live and suspended experiments without signaling them."""
    result = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True, check=True)
    return set(re.findall(r"(?:^|\s)-m\s+((?:atlas_deepc|comparison_methods)(?:\.[A-Za-z_]\w*)+)", result.stdout))


def audit(root=ROOT, active_modules=()):
    paths = inventory(root)
    graph, modules, missing = dependency_graph(root, paths)
    notebooks = {p for p in paths if p.suffix == ".ipynb"}
    support = {root / "scripts" / name for name in SUPPORT_SCRIPTS} & paths
    notebook_used = closure(graph, notebooks)
    active_roots = {modules[name] for name in active_modules if name in modules}
    active = closure(graph, active_roots)
    used = closure(graph, notebooks | support | active_roots)
    tests = {p for p in paths if p.name.startswith("test_")}
    retained_tests, mixed_tests = set(), {}
    for path in tests:
        deps = {p for p in graph[path] if p.name != "__init__.py" and p not in tests}
        if deps <= used:
            retained_tests.add(path)
        elif deps & used:
            mixed_tests[path] = deps - used
    retained = used | retained_tests
    retained.update(p.parent / "__init__.py" for p in retained_tests
                    if p.parent / "__init__.py" in paths)
    candidates = {p for p in paths if p.suffix == ".py" and p not in retained}
    rel = lambda p: str(p.relative_to(root))
    return {
        "notebooks": sorted(map(rel, notebooks)),
        "support_scripts": sorted(map(rel, support)),
        "notebook_dependencies": sorted(map(rel, notebook_used)),
        "protected_process_modules": sorted(active_modules),
        "protected_process_files": sorted(map(rel, active)),
        "retained_code": sorted(rel(p) for p in retained if p.suffix == ".py"),
        "retained_tests": sorted(map(rel, retained_tests)),
        "mixed_tests": {rel(p): sorted(map(rel, deps)) for p, deps in sorted(mixed_tests.items())},
        "deletion_candidates": sorted(map(rel, candidates)),
        "unresolved_retained_imports": {rel(p): sorted(names) for p, names in sorted(missing.items())
                                        if p in retained},
        "dependencies": {rel(p): sorted(map(rel, deps)) for p, deps in sorted(graph.items())},
        "excluded": ["experiments", "archive", "data", "third_party", ".external_experiment_data"],
        "note": "Review candidates before deletion. No experiment data is read or changed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional local JSON inventory; defaults to stdout.")
    args = parser.parse_args()
    report = audit(active_modules=process_modules())
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
        print(json.dumps({key: len(report[key]) for key in
                          ("notebooks", "retained_code", "deletion_candidates", "unresolved_retained_imports")}))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
