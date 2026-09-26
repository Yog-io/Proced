#!/usr/bin/env bash
# Gated Zenodo build runner — POSIX wrapper around run_zenodo_build.py.
#
# The real implementation is the Python script so it also runs on the Windows
# dataset PC, where bash is unavailable:
#
#   python run_zenodo_build.py -path D:\Zenodo-dataset
#   python run_zenodo_build.py -path D:\Zenodo-dataset -seed 42 -workers 8
#   python run_zenodo_build.py -path D:\Zenodo-dataset -dest D:\Zenodo-Dataset_final
#
# Usage (macOS/Linux, unchanged):
#   ./run_zenodo_build.sh /path/to/extracted/zenodo/folder
#   SEED=42 WORKERS=8 DEST_DIR=/path/to/Zenodo-Dataset_final ./run_zenodo_build.sh "$ZENODO_ROOT"
# Default DEST_DIR: sibling folder named Zenodo-Dataset_final next to $ZENODO_ROOT.
#
# Exit codes (identical to run_zenodo_build.py / AGENT_ORCHESTRATION_BRIEF):
#   0  all gates passed (or stopped cleanly with report written per §4)
#   2  pipeline validate failed and no allowed retry, or QA hard-fail
#   3  Step 0 STOP — incomplete extraction suspected (§4.1 >5% / folder gaps)
#   1  harness / unexpected error
set -uo pipefail

ROOT_REPO="$(cd "$(dirname "$0")" && pwd)"
ZENODO_ROOT="${1:-${ZENODO_ROOT:-}}"

if [[ -z "$ZENODO_ROOT" || ! -d "$ZENODO_ROOT" ]]; then
  echo "ERROR: pass extracted Zenodo root as \$1 (or set ZENODO_ROOT)." >&2
  echo "  usage: $0 /path/to/extracted/zenodo/folder" >&2
  echo "  (Windows: python run_zenodo_build.py -path <folder> -seed 42 -workers 8)" >&2
  exit 1
fi

args=(-path "$ZENODO_ROOT")
[[ -n "${SEED:-}" ]]     && args+=(-seed "$SEED")
[[ -n "${WORKERS:-}" ]]  && args+=(-workers "$WORKERS")
[[ -n "${DEST_DIR:-}" ]] && args+=(-dest "$DEST_DIR")
[[ -n "${PAIRS:-}" ]]    && args+=(-pairs "$PAIRS")

exec python3 "$ROOT_REPO/run_zenodo_build.py" "${args[@]}"
