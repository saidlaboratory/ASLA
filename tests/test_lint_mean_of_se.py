"""The mean-of-standard-errors lint: it catches the defect class and passes the repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.lint_mean_of_se import scan_file

REPO = Path(__file__).resolve().parents[1]


def _hits(tmp_path: Path, code: str) -> list[tuple[int, bool, str]]:
    path = tmp_path / "snippet.py"
    path.write_text(code, encoding="utf-8")
    return scan_file(path)


def test_flags_mean_of_se_in_a_ratio(tmp_path):
    assert _hits(tmp_path, "import numpy as np\nratio = sd_all / np.mean(pair_ses)\n")


def test_flags_mean_of_sd_assigned_then_divided(tmp_path):
    code = "def f(x, sds):\n    typical = sds.mean()\n    return x / typical\n"
    assert _hits(tmp_path, code)


def test_passes_root_mean_square(tmp_path):
    code = "import numpy as np\nratio = sd_all / np.sqrt(np.mean(np.square(pair_ses)))\nr2 = a / np.sqrt(np.mean(se ** 2))\n"
    assert not _hits(tmp_path, code)


def test_passes_mean_not_in_a_ratio(tmp_path):
    assert not _hits(tmp_path, "import numpy as np\nreport = np.mean(pair_ses)\n")


def test_annotation_exempts(tmp_path):
    hits = _hits(tmp_path, "import numpy as np\nr = s / np.mean(ses)  # lint: mean-se-intended (described)\n")
    assert hits and all(annotated for _, annotated, _ in hits)


def test_repository_has_no_unannotated_hits():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "lint_mean_of_se.py")], cwd=REPO, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout
