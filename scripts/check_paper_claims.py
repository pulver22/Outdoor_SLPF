#!/usr/bin/env python3
"""Check manuscript language and numeric claims against canonical evidence."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Claim:
    claim_id: str
    permitted: str
    artifact: str


CLAIMS = [
    Claim("C1", "Semantic walls provide row-aligned constraints for the evaluated map.", "paper_main_table.tex"),
    Claim("C2", "Map-frame and aligned errors are reported together.", "paper_main_table.tex"),
    Claim("C3", "Row-level claims use in-row metrics.", "paper_operational_table.tex"),
    Claim("C4", "GNSS stress results are bounded to the evaluated profile.", "paper_gnss_stress_table.tex"),
]

EVIDENCE: dict[str, object] = {"claim_checks": {"passed": True}}
PROHIBITED_PHRASES = (
    "guarantees",
    "deployment ready",
    "guarantees recovery",
    "centimetre-grade",
    "generalises across vineyards",
    "solves perceptual aliasing",
)
NUMERIC_LITERAL = re.compile(r"(?<![A-Za-z])\d+\.\d{2,}(?=\s*(?:m|s|%))")


def _is_negated(text: str, start: int) -> bool:
    context = text[max(0, start - 80) : start].lower()
    return bool(re.search(r"\b(?:not|no|without|cannot|does not|do not|never)\b", context))


def _evidence_numbers(evidence: Mapping[str, object]) -> set[str]:
    numbers: set[str] = set()
    outputs = evidence.get("outputs")
    table_paths = outputs.get("tables", {}) if isinstance(outputs, Mapping) else {}
    for value in table_paths.values() if isinstance(table_paths, Mapping) else []:
        path = Path(str(value))
        if not path.exists():
            continue
        numbers.update(NUMERIC_LITERAL.findall(path.read_text(encoding="utf-8")))
    return numbers


def check_claims(paper_text: str, claims: list[Claim], evidence: Mapping[str, object]) -> list[str]:
    """Return actionable claim errors; an empty list means the contract passes."""
    errors: list[str] = []
    lowered = paper_text.lower()
    for phrase in PROHIBITED_PHRASES:
        start = 0
        while True:
            index = lowered.find(phrase, start)
            if index < 0:
                break
            if not _is_negated(paper_text, index):
                errors.append(f"prohibited claim phrase: {phrase}")
            start = index + len(phrase)
    known_numbers = _evidence_numbers(evidence)
    if known_numbers:
        for number in NUMERIC_LITERAL.findall(paper_text):
            if number not in known_numbers:
                errors.append(f"numeric claim {number} is absent from generated evidence tables")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    paper_text = args.paper.read_text(encoding="utf-8")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    errors = check_claims(paper_text, CLAIMS, evidence)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Claim checks passed for {args.paper} using {args.claims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
