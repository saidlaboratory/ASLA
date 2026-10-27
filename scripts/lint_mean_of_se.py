"""Flag means taken over standard errors or standard deviations that feed a ratio.

``mean(se)`` summarises a typical standard error; it is the wrong denominator for a
ratio of scales, which needs ``sqrt(mean(se ** 2))``. This defect class recurred four
times in this project (practice record, instances 3, 5, 6 and 12).

A hit is a mean (``np.mean``, ``np.nanmean``, ``np.average``, ``statistics.mean``,
``.mean()``) whose argument names a standard error or deviation (an identifier
containing ``se``, ``sd``, ``std``, ``sigma``, ``stderr`` or ``standard_error`` as an
underscore-separated part, or a ``.std(...)`` call) and that

* sits inside a division, or
* is assigned to a name that is used in a division in the same function.

A line carrying ``# lint: mean-se-intended`` with a reason is exempt. Means of
squares (``np.square``, ``** 2``) are not flagged.

Usage: ``python scripts/lint_mean_of_se.py [paths...]``; exits 1 if any unannotated
hit is found, and prints every hit (annotated ones marked).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARK = "lint: mean-se-intended"
SCALE_PARTS = {"se", "ses", "sd", "sds", "std", "stds", "sigma", "sigmas", "stderr", "standard", "error", "errors"}
MEAN_FUNCS = {"mean", "nanmean", "average"}
SKIP_DIRS = {".venv", ".git", "node_modules", "data", "__pycache__", ".pytest_cache"}


def _name_parts(node: ast.AST) -> set[str]:
    parts: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            parts.update(re.split(r"_+", sub.id.lower()))
        elif isinstance(sub, ast.Attribute):
            parts.update(re.split(r"_+", sub.attr.lower()))
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            parts.update(re.split(r"[_\W]+", sub.value.lower()))
    return parts


def _is_scale(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Pow):
            return False
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in {"square"}:
            return False
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in {"pow", "power"}
            and any(isinstance(a, ast.Constant) and a.value == 2 for a in sub.args)
        ):
            return False
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "std":
            return True
    parts = _name_parts(node)
    if {"standard", "error"} <= parts or {"standard", "errors"} <= parts:
        return True
    return bool(parts & (SCALE_PARTS - {"standard", "error", "errors"}))


def _mean_argument(call: ast.Call) -> ast.AST | None:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in MEAN_FUNCS:
        if call.args and isinstance(func.value, ast.Name) and func.value.id in {"np", "numpy", "statistics"}:
            return call.args[0]
        if not call.args:  # x.mean()
            return func.value
        return call.args[0]
    if isinstance(func, ast.Name) and func.id in MEAN_FUNCS and call.args:
        return call.args[0]
    return None


class _Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.hits: list[int] = []
        self._div_stack = 0

    def _scan_function(self, body: list[ast.stmt]) -> None:
        assigned: dict[str, int] = {}
        divided: set[str] = set()
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.AST):
                for call in ast.walk(node.value):
                    if isinstance(call, ast.Call):
                        arg = _mean_argument(call)
                        if arg is not None and _is_scale(arg):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    assigned[target.id] = call.lineno
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                for side in (node.left, node.right):
                    for sub in ast.walk(side):
                        if isinstance(sub, ast.Name):
                            divided.add(sub.id)
                        if isinstance(sub, ast.Call):
                            arg = _mean_argument(sub)
                            if arg is not None and _is_scale(arg):
                                self.hits.append(sub.lineno)
        for name, line in assigned.items():
            if name in divided:
                self.hits.append(line)

    def visit_Module(self, node: ast.Module) -> None:
        top = [n for n in node.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        self._scan_function(top)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        inner = [n for n in node.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self._scan_function(inner)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]


def scan_file(path: Path) -> list[tuple[int, bool, str]]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    scanner = _Scanner()
    scanner.visit(tree)
    lines = text.splitlines()
    out = []
    for line in sorted(set(scanner.hits)):
        source = lines[line - 1] if 0 < line <= len(lines) else ""
        annotated = MARK in source or (line >= 2 and MARK in lines[line - 2])
        out.append((line, annotated, source.strip()))
    return out


def python_files(paths: list[Path]) -> list[Path]:
    files = []
    for root in paths:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
            continue
        for path in root.rglob("*.py"):
            if not any(part in SKIP_DIRS for part in path.relative_to(root).parts):
                files.append(path)
    return sorted(files)


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [REPO]
    failures = 0
    for path in python_files(roots):
        for line, annotated, source in scan_file(path):
            rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
            tag = "intended" if annotated else "HIT"
            failures += not annotated
            print(f"{rel}:{line}: {tag}: {source}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
