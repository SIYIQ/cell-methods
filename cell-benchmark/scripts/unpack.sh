#!/usr/bin/env bash
# Unpack all *_outs.zip in raw/ into raw/<dataset>/outs/  (matching 10x's expected layout).
# Idempotent: skips if outs/cells.csv.gz is already present.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/raw"

shopt -s nullglob
for zip in "${RAW_DIR}"/*/*_outs.zip; do
  dir="$(dirname "$zip")"
  if [[ -f "${dir}/outs/cells.csv.gz" || -f "${dir}/cells.csv.gz" ]]; then
    echo "[skip] $(basename "$dir") already unpacked"
    continue
  fi
  echo "[unzip] $zip"
  # 10x's outs.zip already contains a top-level 'outs/' directory.
  unzip -q -n "$zip" -d "$dir"
done

echo
echo "Layout after unpack:"
for d in "${RAW_DIR}"/*/; do
  echo "  $(basename "$d"):"
  ls "$d" | sed 's/^/    /'
  [[ -d "${d}outs" ]] && { echo "    outs/:"; ls "${d}outs" | head -20 | sed 's/^/      /'; }
done
