#!/usr/bin/env bash
#
# Build the published site: the interactive viewer at the root, the written
# report beside it, plus the coordinated design and every sensitivity scenario.
#
# This is the single definition of a build. Vercel and GitHub Actions both call
# it, so the two deployments cannot drift apart.
#
#   PYTHON=python ./build.sh          # override the interpreter
#   OUT=dist      ./build.sh          # override the output directory
#
set -euo pipefail

PY="${PYTHON:-python3}"
OUT="${OUT:-output}"

echo "build.sh: interpreter $($PY --version 2>&1), output '$OUT'"

# main.py exits 1 when an acceptance check fails. For this study that is the
# expected result - the base case is an audit of an as-specified design and it
# finds two genuine failures. A failing CHECK is an engineering finding, not a
# broken build. Exit 2 (bad input, missing file, tool error) must still fail.
run_study() {
  local status=0
  "$PY" main.py "$@" || status=$?
  if [ "$status" -gt 1 ]; then
    echo "build.sh: main.py failed with exit $status" >&2
    exit "$status"
  fi
  return 0
}

rm -rf "$OUT"

echo "--- base case (settings as specified) ---"
run_study --outdir "$OUT" --quiet

echo "--- coordinated design (settings as recommended) ---"
run_study --settings recommended --outdir "$OUT/recommended" --no-figures --quiet

for scenario in weak_source m1_dol worst_case_start; do
  echo "--- scenario: $scenario ---"
  run_study --scenario "$scenario" --outdir "$OUT/scenario-$scenario" --no-figures --quiet
done

# The site root is the interactive viewer, so the bare domain opens the study.
# The report, figures and CSV exports stay reachable beneath it.
cp "$OUT/viewer/index.html" "$OUT/index.html"

# Tells GitHub Pages not to run the content through Jekyll. Harmless elsewhere.
touch "$OUT/.nojekyll"

echo "--- site ---"
find "$OUT" -maxdepth 2 -type f | sort
du -sh "$OUT"
