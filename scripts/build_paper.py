"""Build the anonymized ICML PDF and a standalone arXiv source tarball from paper/.

* ``build/icml/main.pdf``: the anonymized submission, compiled from paper/ as is.
* ``build/arxiv/``: a flat source tree for arXiv: main.tex with the ICML ``preprint``
  option and the author block from ``paper/authors_preprint.tex``, the bibliography
  inlined from a generated .bbl (arXiv does not run BibTeX), the generated macros, the
  appendix, the style files and the figures. ``build/arxiv.tar.gz`` packs it, and the
  tarball is test-compiled in an empty directory with nothing else available.

Compilation uses tectonic (XeTeX). arXiv compiles with pdfLaTeX; the paper loads
fontspec only under XeTeX, so pdfLaTeX takes the style's own Times path.

Usage: ``python scripts/build_paper.py`` (needs tectonic on PATH or at /opt/homebrew/bin).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
BUILD = REPO / "build"
TECTONIC = shutil.which("tectonic") or "/opt/homebrew/bin/tectonic"
STYLE_FILES = ("icml2026.sty", "algorithm.sty", "algorithmic.sty", "fancyhdr.sty")
AUTHOR_BLOCK = re.compile(r"\\begin\{icmlauthorlist\}.*?\\icmlcorrespondingauthor\{[^}]*\}\{[^}]*\}", re.S)


def _tectonic(workdir: Path, keep: bool = False) -> None:
    args = [TECTONIC, "--keep-logs"] + (["--keep-intermediates"] if keep else []) + ["main.tex"]
    result = subprocess.run(args, cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"tectonic failed in {workdir}:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")


def build_icml() -> Path:
    out = BUILD / "icml"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(PAPER, out, ignore=shutil.ignore_patterns("*.log", "*.blg", "*.aux", "main.pdf"))
    _tectonic(out)
    return out / "main.pdf"


def _preprint_main(bbl: str) -> str:
    source = (PAPER / "main.tex").read_text(encoding="utf-8")
    authors = (PAPER / "authors_preprint.tex").read_text(encoding="utf-8").strip()
    if source.count(r"\usepackage{icml2026}") != 1 or not AUTHOR_BLOCK.search(source):
        raise SystemExit("main.tex no longer has the expected style line or author block")
    source = source.replace(r"\usepackage{icml2026}", r"\usepackage[preprint]{icml2026}")
    source = AUTHOR_BLOCK.sub(lambda _: authors, source, count=1)
    bib = re.compile(r"\\bibliography\{references\}\s*\\bibliographystyle\{icml2026\}")
    if not bib.search(source):
        raise SystemExit("main.tex no longer has the expected bibliography lines")
    return bib.sub(lambda _: "\\bibliographystyle{icml2026}\n" + bbl, source, count=1)


def build_arxiv() -> Path:
    out = BUILD / "arxiv"
    staging = BUILD / "arxiv_bbl"
    for path in (out, staging):
        if path.exists():
            shutil.rmtree(path)
    # 1. A full compile with BibTeX, only to obtain the .bbl.
    shutil.copytree(PAPER, staging, ignore=shutil.ignore_patterns("*.log", "*.blg", "*.aux", "main.pdf"))
    _tectonic(staging, keep=True)
    bbl = (staging / "main.bbl").read_text(encoding="utf-8")
    # 2. The flat source tree, with the bibliography inlined.
    out.mkdir(parents=True)
    (out / "main.tex").write_text(_preprint_main(bbl), encoding="utf-8")
    for name in ("macros.tex", "appendix.tex", *STYLE_FILES):
        shutil.copy2(PAPER / name, out / name)
    (out / "figures").mkdir()
    for fig in sorted((PAPER / "figures").glob("*.pdf")):
        shutil.copy2(fig, out / "figures" / fig.name)
    tarball = BUILD / "arxiv.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(out)))
    # 3. Test: compile the tarball's contents alone, in an empty directory.
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tarball) as tar:
            tar.extractall(tmp, filter="data")
        _tectonic(Path(tmp))
        shutil.copy2(Path(tmp) / "main.pdf", BUILD / "arxiv_test.pdf")
    shutil.rmtree(staging)
    return tarball


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    icml = build_icml()
    tarball = build_arxiv()
    print(f"icml: {icml}")
    print(f"arxiv: {tarball} (test compile: {BUILD / 'arxiv_test.pdf'})")


if __name__ == "__main__":
    main()
