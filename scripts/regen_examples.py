#!/usr/bin/env python3
"""Regenerate output files for catalog-eligible examples.

Reads catalog.toml to discover which gpkit.examples.* scripts to run (no
hardcoded list). Captures each script's stdout to
docs/source/examples/{name}_output.txt.

Non-catalog examples (autosweep, simpleflight, etc.) still have top-level
execution code; their output is captured by the conftest _import_example
fixture during pytest — no separate script needed for them.

Usage:
    uv run python scripts/regen_examples.py
"""

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "gpkit" / "examples"
OUTPUT_DIR = REPO_ROOT / "docs" / "source" / "examples"
CATALOG_PATH = REPO_ROOT / "catalog.toml"

_EXAMPLES_PREFIX = "gpkit.examples."


def _catalog_scripts():
    """Deduplicated list of example script stems registered in catalog.toml."""
    with open(CATALOG_PATH, "rb") as f:
        data = tomllib.load(f)
    seen: set[str] = set()
    stems: list[str] = []
    for entry in data.get("models", []):
        mod = entry.get("module", "")
        if mod.startswith(_EXAMPLES_PREFIX):
            stem = mod[len(_EXAMPLES_PREFIX) :]
            if stem not in seen:
                seen.add(stem)
                stems.append(stem)
    return stems


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scripts = _catalog_scripts()
    failed = []
    for name in scripts:
        script = EXAMPLES_DIR / f"{name}.py"
        if not script.exists():
            print(f"  SKIP {name} (script not found)")
            continue
        out_file = OUTPUT_DIR / f"{name}_output.txt"
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAIL {name}:\n{result.stderr}")
            failed.append(name)
        elif not result.stdout:
            print(f"  WARN {name}: produced no output, skipping write")
            failed.append(name)
        else:
            out_file.write_text(result.stdout)
            lines = result.stdout.count("\n")
            print(f"  OK   {name} → {out_file.relative_to(REPO_ROOT)} ({lines} lines)")

    if failed:
        print(
            f"\n{len(failed)} example(s) failed: {', '.join(failed)}", file=sys.stderr
        )
        sys.exit(1)
    print(f"\n{len(scripts) - len(failed)} catalog example(s) regenerated.")


if __name__ == "__main__":
    main()
