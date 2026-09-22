"""Enforce that reported numbers come from their source, not from a keyboard.

Three defects in this project shared a shape: a number annotated as loaded from
a file, but actually retyped. The annotation was a convention and conventions do
not fail. These tests are the enforcement.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from asla.provenance import ProvenanceError, Source, keys_written_by

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
RESULTS = REPO / "results"

# Scripts whose job is to render a committed document. Every number they emit
# must come from JSON, so a numeric literal in one of them is a provenance leak.
RENDERERS = (
    "render_abstention_results.py",
    "render_allocation_results.py",
    "render_confidence_failures.py",
    "render_consolidated_findings.py",
    "render_curvature_results.py",
    "render_readme_findings.py",
    "render_macros.py",
)

# Literals that are structural rather than reported quantities: array indices,
# formatting precision, percentages of one, and the neutral elements.
ALLOWED_LITERALS = frozenset({0, 1, 2, 3, 4, 100, -1})

# Numbers that are part of a stated definition rather than a measured result.
# A value earns a place here only if it is a PRE-REGISTERED threshold, a nominal
# level, or a scenario parameter the document names as its own premise -- never
# a quantity read off the data. Each entry is justified in the comment above it.
ALLOWED_FLOATS: dict[str, frozenset[float]] = {
    # Pre-registered verdict thresholds from PREDICTIONS_TASK_ABSTENTION.md
    # (A1 coverage 0.85/0.88/0.70, A2 level 0.70/0.85, A3 seeds 10/20, A4
    # 0.30/0.60, A5 0.07/0.10), the headline delta, and a monotonicity epsilon.
    "render_abstention_results.py": frozenset({0.05, 0.02, 0.07, 0.1, 0.3, 0.6, 0.7, 0.85, 0.88, 10.0, 20.0, 1e-9}),
    # Pre-registered thresholds from PREDICTIONS_TASK_ALLOCATION.md.
    "render_allocation_results.py": frozenset({0.02, 0.25, 0.45, 0.9, 0.5, 0.8}),
    # The scenario the Student-t section states as its premise: delta, the
    # comparison count, and the seed ladder whose quantiles it tabulates.
    # Plus SMALL_SCALE_CEILING_FLOPS (the 90M rung, the boundary the section
    # reports about) and TOTAL_ABSTENTION (the float tolerance for a rate of 1).
    "render_confidence_failures.py": frozenset({0.05, 5.0, 10.0, 20.0, 50.0, 300.0, 5.8e18, 0.999}),
    # The same stated scenario, quoted in the README's section 6.
    "render_readme_findings.py": frozenset({0.05, 50.0, 300.0}),
    # N_INSTANCES and N_FLATTERING: counts of documented artifact instances in
    # the section itself. Structural facts about the prose, not quantities read
    # off the data, and centralised precisely so the four places that state them
    # cannot drift apart.
    "render_consolidated_findings.py": frozenset({5.0, 6.0, 8.0}),
    "render_curvature_results.py": frozenset(),
    "render_macros.py": frozenset(),
}


def _numeric_literals(path: Path) -> list[tuple[int, float]]:
    """Every numeric literal in a file, with its line number."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, float]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                continue
            found.append((node.lineno, float(node.value)))
    return found


@pytest.mark.parametrize("name", RENDERERS)
def test_renderers_contain_no_unexplained_numeric_literals(name: str) -> None:
    """A number in a rendering path must come from JSON, not from the source.

    This is the check the `c4_ssr_ratio = 10.03` and `A4_measured = -0.0042`
    defects would have failed: both were annotated as loaded and were typed.
    """

    path = SCRIPTS / name
    if not path.exists():  # pragma: no cover - renderer set may change
        pytest.skip(f"{name} not present")
    allowed = ALLOWED_LITERALS | {int(v) for v in ALLOWED_FLOATS[name] if float(v).is_integer()}
    offenders = [
        (line, value)
        for line, value in _numeric_literals(path)
        if value not in allowed and value not in ALLOWED_FLOATS[name]
    ]
    assert not offenders, (
        f"{name} contains numeric literals in a rendering path: {offenders}. "
        "Load the value from its source JSON via asla.provenance.Source, or add it "
        "to ALLOWED_FLOATS with a justification if it is a stated definition."
    )


def test_source_reads_by_key_path() -> None:
    source = Source.load("theory_v2/theory_v2.json")
    value = source.number(
        "FINDING_misspecification_is_structured.evidence.1_misspecification_is_real.residual_scatter_over_seed_se"
    )
    assert value > 0


def test_source_raises_on_missing_key() -> None:
    source = Source.load("theory_v2/theory_v2.json")
    with pytest.raises(ProvenanceError, match="no key"):
        source.number("FINDING_misspecification_is_structured.evidence.does_not_exist")


def test_source_raises_on_missing_file() -> None:
    with pytest.raises(ProvenanceError, match="no such results file"):
        Source.load("nowhere/nothing.json")


def test_source_rejects_wrong_type() -> None:
    source = Source.load("theory_v2/theory_v2.json")
    with pytest.raises(ProvenanceError, match="not a number"):
        source.number("V4_overall")


def test_source_rejects_bool_as_number() -> None:
    """A bool is an int in Python; it must not pass as a reported number."""

    source = Source.load("curvature/curvature_gate.json")
    with pytest.raises(ProvenanceError, match="not a number"):
        source.number("gate.gate_passes")


# (results file, the script that writes it)
PRODUCERS = {
    "abstention/abstention_1B.json": "run_abstention_study.py",
    "allocation/allocation_1B.json": "run_allocation_study.py",
    "curvature/curvature_gate.json": "run_curvature_gate.py",
    "theory/theory_check.json": "run_theory_check.py",
    "theory_v2/theory_v2.json": "run_theory_v2.py",
}


@pytest.mark.parametrize("relative,script", sorted(PRODUCERS.items()))
def test_every_shipped_top_level_key_is_written_by_its_producer(relative: str, script: str) -> None:
    """No shipped JSON may contain a top-level field no code path produces.

    This is the `run_abstention_study.py` defect: the committed JSON carried a
    `scored` baseline that the script never wrote, because an ad-hoc run had
    left it there. The document then cited a number the script could not
    regenerate.
    """

    path = RESULTS / relative
    source = SCRIPTS / script
    if not path.exists() or not source.exists():  # pragma: no cover
        pytest.skip(f"{relative} or {script} not present")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):  # pragma: no cover - all shipped files are objects
        pytest.skip("payload is not an object")
    text = source.read_text(encoding="utf-8")
    orphans = [key for key in payload if f'"{key}"' not in text]
    assert not orphans, (
        f"{relative} has top-level keys {orphans} that never appear in {script}. "
        "Either the script no longer writes them, or the file was edited outside it."
    )


def test_keys_written_by_walks_nested_paths() -> None:
    payload = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    assert keys_written_by(payload) == {"a", "a.b", "a.c", "a.c.d", "e"}


# --- The paper itself: no hand-typed numbers, no undefined macros ---

PAPER = REPO / "paper" / "main.tex"
MACROS = REPO / "paper" / "macros.tex"

# Numerals that are structural rather than reported quantities: section numbers
# in citation identifiers, LaTeX sizing, and the arXiv ids of cited work.
PAPER_ALLOWED_CONTEXTS = (
    "arXiv:",
    "%% nolint",
)


def _paper_numeric_literals(text: str) -> list[tuple[int, str]]:
    import re

    offenders = []
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("%"):
            continue
        if any(token in line for token in PAPER_ALLOWED_CONTEXTS):
            continue
        for match in re.finditer(r"(?<![\w\\.])\d+(?:\.\d+)?%?", line):
            offenders.append((number, match.group(0)))
    return offenders


def test_paper_has_no_hand_typed_numbers() -> None:
    """Every figure in the paper must arrive through a generated macro.

    A retyped 4.22 survived into five documents and a draft before it was found
    to be wrong. This is the check that would have caught it at the paper.
    """

    if not PAPER.exists():  # pragma: no cover - paper not always present
        pytest.skip("paper not present")
    offenders = _paper_numeric_literals(PAPER.read_text(encoding="utf-8"))
    assert not offenders, (
        f"paper/main.tex contains hand-typed numbers {offenders[:10]}; route them through scripts/render_macros.py"
    )


def test_paper_macros_are_all_defined() -> None:
    """A macro the paper uses must exist in the generated file."""

    import re

    if not PAPER.exists() or not MACROS.exists():  # pragma: no cover
        pytest.skip("paper or macros not present")
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", MACROS.read_text(encoding="utf-8")))
    latex_builtins = set(
        """documentclass usepackage input title author date begin end maketitle section
        label paragraph emph ref cdot textbf textit item cite footnote hyperref
        includegraphics caption centering toprule midrule bottomrule S Phi Delta """.split()
    )
    used = set(re.findall(r"\\([A-Za-z]+)", PAPER.read_text(encoding="utf-8")))
    missing = sorted((used - latex_builtins) & {u for u in used if u not in latex_builtins} - defined)
    # Only flag macros that look like ours: camelCase and not a known builtin.
    missing = [m for m in missing if any(c.isupper() for c in m)]
    assert not missing, f"paper references undefined macros: {missing}"


def test_every_results_directory_is_tracked() -> None:
    """A results directory outside the .gitignore allowlist is silently untracked.

    This has now happened twice. The second time it hid the provenance evidence
    for the project's own headline correction, and the paper's macros read from
    those files, so a clone could not regenerate the paper.
    """

    allowlist = {
        line.strip()[len("!results/") :].rstrip("/")
        for line in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("!results/")
    }
    present = {p.name for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))}
    missing = sorted(present - allowlist)
    assert not missing, f"results directories not allowlisted in .gitignore: {missing}"
