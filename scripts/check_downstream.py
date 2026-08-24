"""Run sibling model repos' test suites against this working copy of gpkit-core.

Whether a model repo picks up local gpkit-core depends on its uv.lock: some
are locked to an editable ../gpkit-core, others to a PyPI release.  Passing
--with-editable makes it local for every repo regardless, so this checks what
is on disk here rather than whatever each lockfile happens to say.

Sibling repos are discovered by looking for a catalog.toml, so no repo names
are hardcoded.

    python scripts/check_downstream.py           # every sibling found
    python scripts/check_downstream.py lunar     # just the named ones
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def find_siblings(names=()):
    "Sibling directories holding a catalog.toml, optionally filtered by name."
    found = sorted(
        p.parent for p in REPO.parent.glob("*/catalog.toml") if p.parent != REPO
    )
    if names:
        found = [p for p in found if p.name in names]
    return found


def run_one(repo: Path) -> bool:
    "Run repo's tests against local gpkit-core. Returns True if they passed."
    result = subprocess.run(
        ["uv", "run", "--with-editable", str(REPO), "pytest", "tests/", "-q"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout or result.stderr
    tail = out.strip().splitlines()
    print("  " + "\n  ".join(tail[-3:] if tail else ["(no output)"]))
    if result.returncode == 0 and " passed" not in out:
        # A suite that only skips is green without checking anything; say so
        # rather than letting it read as coverage.
        print("  NOTE: nothing actually ran here — every test skipped")
    return result.returncode == 0


def main():
    siblings = find_siblings(sys.argv[1:])
    if not siblings:
        print("no sibling repos with a catalog.toml found next to gpkit-core")
        return 0

    failed = []
    for repo in siblings:
        print(f"\n=== {repo.name} ===")
        if not run_one(repo):
            failed.append(repo.name)

    print()
    passed = [p.name for p in siblings if p.name not in failed]
    if passed:
        print(f"pass:   {', '.join(passed)}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
