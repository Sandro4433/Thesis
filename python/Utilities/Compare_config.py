"""
Compare the current workspace configuration against a known-correct reference.

Loads configuration.json from File_Exchange/ and compares it to a ground-truth
file whose path is set below.  Reports whether the two match and, if not,
lists every difference found.

Numerical values (positions, orientations, quaternions) are compared with a
configurable tolerance so that minor sensor noise doesn't cause false failures.

Usage:
    python Utilities/compare_config.py
    python Utilities/compare_config.py --truth path/to/other_correct.json
    python Utilities/compare_config.py --tolerance 0.01
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ──────────────────────────────────────────────────────────────────────
# PATHS — edit these when switching scenarios
# ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]          # .../python
REPO_ROOT   = PROJECT_DIR.parent                           # one level above python/

CONFIG_PATH = PROJECT_DIR / "File_Exchange" / "configuration.json"
TRUTH_PATH  = REPO_ROOT / "Experiments" / "Scenario_2" / "post_config.json"

# Default tolerance for numerical comparisons (positions in metres,
# orientations in degrees, quaternion components).
# 0.005 m = 5 mm — reasonable for vision-based part detection.
DEFAULT_TOLERANCE = 0.8


# ──────────────────────────────────────────────────────────────────────
# Comparison helpers
# ──────────────────────────────────────────────────────────────────────
def _sort_key(item: Any) -> str:
    """Produce a stable sort key for dicts/lists/primitives."""
    if isinstance(item, dict):
        return json.dumps(item, sort_keys=True)
    return json.dumps(item)


def _normalise_list(lst: list) -> list:
    """Return a sorted copy so element order doesn't cause false diffs."""
    try:
        return sorted(lst, key=_sort_key)
    except TypeError:
        return lst


def _numbers_close(a: float, b: float, tol: float) -> bool:
    """Check if two numbers are within tolerance."""
    return abs(a - b) <= tol


def _compare(
    a: Any,
    b: Any,
    path: str = "$",
    tol: float = DEFAULT_TOLERANCE,
) -> List[Tuple[str, Any, Any]]:
    """
    Recursively compare two JSON-like structures.

    Returns a list of (json_path, value_in_config, value_in_truth) tuples
    for every leaf difference found.  Lists of dicts are compared
    order-independently (sorted by content).

    Numerical values (int/float) are compared with tolerance `tol`.
    """
    diffs: List[Tuple[str, Any, Any]] = []

    # Both numeric — compare with tolerance
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not _numbers_close(float(a), float(b), tol):
            diffs.append((path, a, b))
        return diffs

    if type(a) is not type(b):
        diffs.append((path, a, b))
        return diffs

    if isinstance(a, dict):
        all_keys = sorted(set(list(a.keys()) + list(b.keys())))
        for k in all_keys:
            child_path = f"{path}.{k}"
            if k not in a:
                diffs.append((child_path, "<missing>", b[k]))
            elif k not in b:
                diffs.append((child_path, a[k], "<missing>"))
            else:
                diffs.extend(_compare(a[k], b[k], child_path, tol))

    elif isinstance(a, list):
        a_sorted = _normalise_list(a)
        b_sorted = _normalise_list(b)
        if len(a_sorted) != len(b_sorted):
            diffs.append((path, a_sorted, b_sorted))
        else:
            for i, (ai, bi) in enumerate(zip(a_sorted, b_sorted)):
                diffs.extend(_compare(ai, bi, f"{path}[{i}]", tol))
    else:
        if a != b:
            diffs.append((path, a, b))

    return diffs


def compare_configs(
    config: Dict[str, Any],
    truth: Dict[str, Any],
    tol: float = DEFAULT_TOLERANCE,
) -> List[Tuple[str, Any, Any]]:
    """Public entry point — returns list of differences."""
    return _compare(config, truth, tol=tol)


# ──────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────
def _fmt(val: Any) -> str:
    if isinstance(val, (dict, list)):
        return json.dumps(val, sort_keys=True)
    return repr(val)


def print_report(
    diffs: List[Tuple[str, Any, Any]],
    tol: float = DEFAULT_TOLERANCE,
) -> None:
    if not diffs:
        print(f"✅  PASS — configuration matches the ground truth.")
        print(f"   (numerical tolerance: ±{tol})")
        return

    print(f"❌  FAIL — {len(diffs)} difference(s) found  (numerical tolerance: ±{tol}):\n")
    for path, got, expected in diffs:
        line = f"  Path:     {path}\n"
        line += f"  Got:      {_fmt(got)}\n"
        line += f"  Expected: {_fmt(expected)}"

        # Show the actual delta for numeric diffs so you can judge severity
        if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
            delta = abs(float(got) - float(expected))
            line += f"\n  Delta:    {delta:.6f}"

        print(line)
        print()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare configuration.json against a ground-truth file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help=f"Path to the configuration under test (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--truth",
        type=str,
        default=str(TRUTH_PATH),
        help=f"Path to the correct/reference file (default: {TRUTH_PATH})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Numerical tolerance for positions/orientations (default: {DEFAULT_TOLERANCE})",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    truth_path  = Path(args.truth)
    tol         = args.tolerance

    # Load both files
    for label, p in [("Config", config_path), ("Truth", truth_path)]:
        if not p.exists():
            print(f"❌  {label} file not found: {p}")
            return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(truth_path, "r", encoding="utf-8") as f:
        truth = json.load(f)

    print(f"Config:    {config_path}")
    print(f"Truth:     {truth_path}")
    print(f"Tolerance: ±{tol}")
    print()

    diffs = compare_configs(config, truth, tol=tol)
    print_report(diffs, tol=tol)


if __name__ == "__main__":
    main()