#!/usr/bin/env bash
# Publish src/daystorm as a new version of the Kaggle dataset the kernel reads.
#
# The kernel cannot clone this repo, so the source arrives as a dataset. Skipping
# this step fails silently in the worst way: the kernel runs the *previous*
# version of the code, and a newly added flag comes back as "unrecognized
# arguments" from inside a subprocess whose output nobody is reading.
set -euo pipefail

SLUG="nullpointeranish/daystorm-src"
STAGE="$(mktemp -d)/daystorm-src"
trap 'rm -rf "$(dirname "${STAGE}")"' EXIT

# One wrapper level on purpose. `-r zip` zips each top-level folder with paths
# relative to it, and Kaggle extracts those entries verbatim - so zipping
# `daystorm/` directly lands `__init__.py` and `bench/` at the dataset root and
# the package disappears. Zipping `src/` keeps `daystorm/` as the top entry.
mkdir -p "${STAGE}/src"
cp -r src/daystorm "${STAGE}/src/daystorm"
find "${STAGE}" -name __pycache__ -type d -prune -exec rm -rf {} +

# `kaggle datasets metadata` writes the fields nested under "info", which
# `kaggle datasets version` then rejects with "ID or slug must be specified in
# the metadata". Declaring the two fields it actually reads is shorter than
# reshaping its own output.
cat > "${STAGE}/dataset-metadata.json" <<JSON
{"id": "${SLUG}", "title": "daystorm-src"}
JSON

kaggle datasets version -p "${STAGE}" -m "${1:-sync src/daystorm}" -r zip
echo "--> pushed ${SLUG}; wait for it to finish processing before running the kernel"
