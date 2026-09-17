"""Read reported numbers from their source JSON, so they cannot be retyped.

Three numbers in this project were annotated as loaded from a source file and
were in fact typed by hand: ``c4_ssr_ratio`` (10.03 against a true 10.0252),
``A4_measured`` (asserted "measured by seed bootstrap" while being a literal),
and the two-parameter cross-references. The annotations did not prevent the
defect because nothing enforced them.

:class:`Source` makes the enforcement structural. A renderer or comparison
asks for a value *by key path*; if the key is absent the call raises, and the
returned value carries the file and key it came from. A retyped constant then
fails the lint in ``tests/test_provenance.py``, which rejects numeric literals
in rendering paths, rather than passing silently.

Usage::

    source = Source.load("theory_v2/theory_v2.json")
    ratio = source.number("FINDING_misspecification_is_structured.evidence."
                          "1_misspecification_is_real.residual_scatter_over_seed_se")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


class ProvenanceError(KeyError):
    """Raised when a requested value is absent from, or the wrong type in, its source."""


@dataclass(frozen=True)
class Source:
    """A committed results file, addressed by dotted key path."""

    relative: str
    payload: Any

    @classmethod
    def load(cls, relative: str) -> Source:
        path = RESULTS / relative
        if not path.exists():
            raise ProvenanceError(f"no such results file: {relative}")
        return cls(relative=relative, payload=json.loads(path.read_text(encoding="utf-8")))

    def _resolve(self, key_path: str) -> Any:
        node: Any = self.payload
        for part in key_path.split("."):
            if isinstance(node, list):
                try:
                    node = node[int(part)]
                    continue
                except (ValueError, IndexError) as exc:
                    raise ProvenanceError(f"{self.relative}: no index {part!r} in {key_path!r}") from exc
            if not isinstance(node, dict) or part not in node:
                raise ProvenanceError(f"{self.relative}: no key {part!r} in {key_path!r}")
            node = node[part]
        return node

    def number(self, key_path: str) -> float:
        """Return a numeric value, raising if it is absent or not a number."""

        value = self._resolve(key_path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProvenanceError(f"{self.relative}: {key_path!r} is {type(value).__name__}, not a number")
        return float(value)

    def integer(self, key_path: str) -> int:
        value = self._resolve(key_path)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProvenanceError(f"{self.relative}: {key_path!r} is {type(value).__name__}, not an int")
        return int(value)

    def text(self, key_path: str) -> str:
        value = self._resolve(key_path)
        if not isinstance(value, str):
            raise ProvenanceError(f"{self.relative}: {key_path!r} is {type(value).__name__}, not a string")
        return value

    def get(self, key_path: str) -> Any:
        """Return any value, raising only if the key is absent."""

        return self._resolve(key_path)


def keys_written_by(payload: Any, prefix: str = "") -> set[str]:
    """Return every dotted key path present in a payload.

    Used by the shipped-results check to confirm that a JSON file contains no
    field beyond what its producing script writes.
    """

    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.add(path)
            found |= keys_written_by(value, path)
    return found


__all__ = ["ProvenanceError", "Source", "keys_written_by"]
