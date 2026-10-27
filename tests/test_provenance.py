"""Enforce that reported numbers come from their source, not from a keyboard.

Three defects in this project shared a shape: a number annotated as loaded from
a file, but actually retyped. The annotation was a convention and conventions do
not fail. These tests are the enforcement.
"""

from __future__ import annotations

import ast
import json
import re
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
    "render_consolidated_findings.py": frozenset({5.0, 6.0, 10.0, 13.0}),  # 10: the base for per-decade slopes
    "render_curvature_results.py": frozenset(),
    # QUOTED_M and QUOTED_DELTAS: the grid points the prose quotes (a stated choice, checked against the JSON grid)
    "render_macros.py": frozenset({5.0, 10.0, 25.0}),
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
        includegraphics caption centering toprule midrule bottomrule S Phi Delta
        STATE FOR ENDFOR printAffiliationsAndNotice ifXeTeX Cref FloatBarrier""".split()
    )
    used = set(re.findall(r"\\([A-Za-z]+)", PAPER.read_text(encoding="utf-8")))
    appendix = REPO / "paper" / "appendix.tex"
    if appendix.exists():
        used |= set(re.findall(r"\\([A-Za-z]+)", appendix.read_text(encoding="utf-8")))
    missing = sorted((used - latex_builtins) & {u for u in used if u not in latex_builtins} - defined)
    # Only flag macros that look like ours: camelCase and not a known builtin.
    missing = [m for m in missing if any(c.isupper() for c in m)]
    assert not missing, f"paper references undefined macros: {missing}"


# --- Spelled-out numbers must agree with their macros ---
#
# Digits are already banned from prose. A number written as a word ("twenty-five candidates") is still a claim, and
# it can disagree with the macro it sits beside. Every spelled-out number of three or more must be bound, on the
# same line, by a trailing annotation ``%% words: <item>, <item>`` with one item per spelled number, in order:
#   \macro      the macro's value must equal the spelled number;
#   ~\macro     an approximate count ("a few hundred"): the macro must have the same number of digits;
#   - (reason)  structural (a model name, a rule's name, the paper's own outline), with the reason stated.
# "one" and "two" are left out: they are almost always structural ("one pair", "two forms").

_UNITS = "one two three four five six seven eight nine".split()
_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}  # fmt: skip
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_NUMBER = re.compile(
    r"\b(?:(?P<few>few\s+)?(?:a\s+)?(?P<hundred>hundred)"
    r"|(?P<tens>" + "|".join(_TENS) + r")(?:-(?P<unit>" + "|".join(_UNITS) + r"))?"
    r"|(?P<word>" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r"))\b",
    re.IGNORECASE,
)


def _spelled_numbers(prose: str) -> list[tuple[int, bool]]:
    """(value, approximate) for each spelled-out number of three or more."""

    out = []
    for m in _NUMBER.finditer(prose):
        if m.group("hundred"):
            out.append((100, bool(m.group("few"))))
        elif m.group("tens"):
            out.append(
                (
                    _TENS[m.group("tens").lower()] + (_UNITS.index(m.group("unit").lower()) + 1 if m.group("unit") else 0),
                    False,
                )
            )
        elif _WORDS[m.group("word").lower()] >= 3:
            out.append((_WORDS[m.group("word").lower()], False))
    return out


def _macro_values(text: str) -> dict[str, str]:
    return dict(re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*)\}  % ", text, flags=re.MULTILINE))


def _as_number(value: str) -> float | None:
    cleaned = re.sub(r"[${}\\,]", "", value).strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def spelled_number_violations(tex: str, macro_values: dict[str, str]) -> list[str]:
    problems = []
    for number, line in enumerate(tex.splitlines(), start=1):
        if line.strip().startswith("%"):
            continue
        prose, _, comment = re.split(r"(?<!\\)%", line, maxsplit=1)[0], None, ""
        parts = re.split(r"(?<!\\)%", line, maxsplit=1)
        prose, comment = parts[0], (parts[1] if len(parts) > 1 else "")
        found = _spelled_numbers(prose)
        m = re.search(r"words:\s*([^;]*)", comment)
        items = [i.strip() for i in m.group(1).split(",")] if m else []
        items = [i for i in items if i]
        if not found and not items:
            continue
        if len(items) != len(found):
            problems.append(f"line {number}: {len(found)} spelled number(s), {len(items)} binding(s): {prose.strip()[:70]}")
            continue
        for (value, approx), item in zip(found, items):
            b = re.fullmatch(r"(~)?\\(\w+)", item)
            if item.startswith("-"):
                if not re.search(r"\(.+\)", item):
                    problems.append(f"line {number}: structural binding without a reason")
            elif not b or b.group(2) not in macro_values:
                problems.append(f"line {number}: binding {item!r} is not a defined macro")
            else:
                v = _as_number(macro_values[b.group(2)])
                if v is None:
                    problems.append(f"line {number}: \\{b.group(2)} is not numeric")
                elif b.group(1) or approx:
                    if not (b.group(1) and approx) or len(str(int(v))) != len(str(value)):
                        problems.append(f"line {number}: approximate {value} against \\{b.group(2)}={v:g}")
                elif v != value:
                    problems.append(f"line {number}: prose says {value}, \\{b.group(2)} is {v:g}")
    return problems


def test_spelled_numbers_lint_catches_disagreement() -> None:
    values = {"seeds": "3", "worlds": "300", "label": "1B"}
    assert spelled_number_violations("Three seeds.  %% words: \\seeds", values) == []
    assert spelled_number_violations("ten seeds  %% words: \\seeds", values)  # disagrees
    assert spelled_number_violations("three seeds", values)  # unbound
    assert spelled_number_violations("a few hundred worlds  %% words: ~\\worlds", values) == []
    assert spelled_number_violations("rule of three  %% words: - (name of a rule)", values) == []
    assert spelled_number_violations("rule of three  %% words: -", values)  # no reason
    assert spelled_number_violations("one or two forms", values) == []
    assert spelled_number_violations("twenty-five candidates  %% words: \\worlds", values)


APPENDIX = REPO / "paper" / "appendix.tex"


def test_appendix_has_no_hand_typed_numbers() -> None:
    if not APPENDIX.exists():  # pragma: no cover
        pytest.skip("appendix not present")
    offenders = _paper_numeric_literals(APPENDIX.read_text(encoding="utf-8"))
    assert not offenders, f"paper/appendix.tex contains hand-typed numbers {offenders[:10]}"


def test_appendix_spelled_numbers_match_their_macros() -> None:
    if not APPENDIX.exists() or not MACROS.exists():  # pragma: no cover
        pytest.skip("appendix or macros not present")
    problems = spelled_number_violations(
        APPENDIX.read_text(encoding="utf-8"), _macro_values(MACROS.read_text(encoding="utf-8"))
    )
    assert not problems, "spelled-out numbers not bound to agreeing macros:\n" + "\n".join(problems)


def test_paper_spelled_numbers_match_their_macros() -> None:
    if not PAPER.exists() or not MACROS.exists():  # pragma: no cover
        pytest.skip("paper or macros not present")
    problems = spelled_number_violations(
        PAPER.read_text(encoding="utf-8"), _macro_values(MACROS.read_text(encoding="utf-8"))
    )
    assert not problems, "spelled-out numbers not bound to agreeing macros:\n" + "\n".join(problems)


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


def test_prose_correction_span_matches_across_line_breaks() -> None:
    import importlib

    apply = importlib.import_module("scripts.apply_prose_corrections")
    raw = "Before. There is\nan open   discrepancy, which we\nare investigating. After."
    entry = {"id": "x", "existing": "There is an open discrepancy, which we are investigating.", "action": "replace",
             "replacement": "Resolved."}  # fmt: skip
    out, line = apply.apply_text(raw, entry)
    assert out == "Before. Resolved. After." and line == 1
    deleted, _ = apply.apply_text(raw, dict(entry, action="delete"))
    assert deleted == "Before. After."


UNIT_PHRASES = ("per decade", "per natural log", "per rung")
_TABLE_ENVS = re.compile(r"\\begin\{(table\*?|longtable)\}(.*?)\\end\{\1\}", re.S)


def _rendered_tables() -> list[tuple[str, str, set[str]]]:
    """(file, table text, rows macros used) for every table in the paper."""

    out = []
    for path in (PAPER, REPO / "paper" / "appendix.tex"):
        if not path.exists():
            continue
        for match in _TABLE_ENVS.finditer(path.read_text(encoding="utf-8")):
            text = match.group(2)
            out.append((path.name, text, set(re.findall(r"\\(\w+Rows)\b", text))))
    return out


def test_slope_tables_state_the_unit_their_values_carry() -> None:
    """A table's stated slope unit must match the unit macro rendered from the results file.

    Practice record, instance 13: per-natural-log slopes were printed under a hand-typed
    "per decade" header. Every rows macro that renders slopes has a sibling ``...Units`` macro
    derived from the JSON's unit field; the header and caption must state exactly those units.
    """

    macros = MACROS.read_text(encoding="utf-8")
    units = dict(re.findall(r"\\newcommand\{\\(\w+Rows)Units\}\{([^}]*)\}", macros))
    checked = 0
    for name, text, rows in _rendered_tables():
        prose = re.sub(r"\s+", " ", re.sub(r"\\\\\s*", " ", text))
        stated = {u for u in UNIT_PHRASES if u in prose}
        declared = set()
        for row in rows:
            if row in units:
                declared |= {u.strip() for u in units[row].split(";")}
        if stated or declared:
            checked += 1
            assert declared, f"{name}: table states {sorted(stated)} but its rows macro has no Units macro: {rows}"
            assert stated == declared, f"{name}: table states {sorted(stated)}, results carry {sorted(declared)}"
    assert checked >= 3
