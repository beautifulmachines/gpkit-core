#!/usr/bin/env bash
# loc.sh – quick tracked-file LOC report (source / tests / docs)

set -euo pipefail

count_loc () {
  # Accept any number of git-pathspecs, so we can add exclusions
  git ls-files -z -- "$@" \
    | xargs -0 -r cat \
    | wc -l \
    | tr -d '[:space:]'
}

# Source: everything in gpkit/** EXCEPT gpkit/tests/**
src_lines=$(  count_loc 'gpkit/**' ':!gpkit/tests/**')
test_lines=$( count_loc 'gpkit/tests/**')
doc_lines=$(  count_loc 'docs/**')

printf "\nLines of code in this repo (tracked files only):\n"
printf "  %-6s : %8s\n" "Source" "$src_lines"
printf "  %-6s : %8s\n" "Tests"  "$test_lines"
printf "  %-6s : %8s\n\n" "Docs"   "$doc_lines"
