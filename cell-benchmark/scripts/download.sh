#!/usr/bin/env bash
# Download Xenium single-cell benchmark datasets from 10x Genomics CDN.
# Resumable, parallel-safe, validates Content-Length when the server reports one.
#
# Usage:
#   ./download.sh                  # download core 5 datasets (default group)
#   ./download.sh --group all      # core + IDC_Big (~171 GB)
#   ./download.sh --group big      # only IDC_Big (~111 GB)
#   ./download.sh --dataset hSkin_Melanoma   # one specific dataset
#   ./download.sh --dry-run        # print plan, do not download
#
# Concurrency: DL_JOBS=N controls parallel downloads (default 3).
# 10x CDN supports HTTP/2 + Range, so curl -C - is fine for resume.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${ROOT_DIR}/raw"
LOG_DIR="${ROOT_DIR}/logs"
MANIFEST="${SCRIPT_DIR}/urls.tsv"
JOBS="${DL_JOBS:-3}"

# ---- worker mode: invoked by the parent via `bash $0 --worker <rel> <url>` ----
if [[ "${1:-}" == "--worker" ]]; then
  rel="$2"
  url="$3"
  out="${RAW_DIR}/${rel}"
  log="${LOG_DIR}/$(echo "$rel" | tr '/' '_').log"
  mkdir -p "$(dirname "$out")"

  # Skip if local size already matches remote Content-Length
  if [[ -f "$out" ]]; then
    local_size=$(stat -c%s "$out")
    remote_size=$(curl -sI --max-time 30 "$url" \
      | awk -F': ' 'tolower($1)=="content-length"{gsub("\r","",$2); print $2; exit}')
    if [[ -n "$remote_size" && "$local_size" == "$remote_size" ]]; then
      printf '[skip] %s (%s)\n' "$rel" "$(numfmt --to=iec --suffix=B "$local_size")"
      exit 0
    fi
  fi

  printf '[get ] %s\n' "$rel"
  curl --fail-with-body --location \
       --retry 8 --retry-delay 5 --retry-all-errors \
       -C - --speed-time 60 --speed-limit 1024 \
       --output "$out" "$url" \
       >> "$log" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    printf '[done] %s (%s)\n' "$rel" "$(numfmt --to=iec --suffix=B "$(stat -c%s "$out")")"
  else
    printf '[FAIL] %s  (see %s)\n' "$rel" "$log" >&2
  fi
  exit $rc
fi

# ---------------------------- parent (orchestrator) ----------------------------
GROUP="core"
DATASET=""
DRY_RUN=0

CORE_DATASETS=(
  Human_Breast_Cancer_Rep1
  Human_Breast_Cancer_Rep2
  hColon_Non_diseased
  mouse_Colon
  hSkin_Melanoma
)
BIG_DATASETS=(
  Human_Breast_IDC_Big_Rep1
  Human_Breast_IDC_Big_Rep2
)

usage() {
  cat <<'EOF'
Usage:
  ./download.sh [--group core|big|all] [--dataset NAME] [--dry-run]
  DL_JOBS=N to override parallel job count (default 3).
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)   GROUP="${2:?}"; shift 2;;
    --dataset) DATASET="${2:?}"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage;;
    *) echo "unknown arg: $1"; usage;;
  esac
done

mkdir -p "${RAW_DIR}" "${LOG_DIR}"

declare -A WHITELIST=()
if [[ -n "$DATASET" ]]; then
  WHITELIST["$DATASET"]=1
else
  case "$GROUP" in
    core) for d in "${CORE_DATASETS[@]}"; do WHITELIST["$d"]=1; done;;
    big)  for d in "${BIG_DATASETS[@]}";  do WHITELIST["$d"]=1; done;;
    all)  for d in "${CORE_DATASETS[@]}" "${BIG_DATASETS[@]}"; do WHITELIST["$d"]=1; done;;
    *) echo "unknown group: $GROUP"; usage;;
  esac
fi

# Build the work list as a NUL-delimited stream of "<rel>\0<url>\0" pairs.
WORK="$(mktemp)"
trap 'rm -f "$WORK"' EXIT

n=0
while IFS=$'\t' read -r dataset role rel_path url; do
  [[ -z "${dataset:-}" || "${dataset:0:1}" == "#" ]] && continue
  [[ -z "${WHITELIST[$dataset]:-}" ]] && continue
  printf '%s\t%s\t%s\n' "$dataset" "$rel_path" "$url" >> "$WORK"
  n=$((n+1))
done < "$MANIFEST"

echo "Plan: $n files into ${RAW_DIR}  (jobs=$JOBS)"
awk -F'\t' '{printf "  %-30s -> %s\n", $1, $2}' "$WORK"

if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi

# Dispatch: hand each line to xargs, which slices into (rel, url) by tab and
# passes them as separate argv to the worker. -L 1 means one record per call.
# Using printf '%s\n' to feed lines, and a small awk to emit rel\turl pairs.
awk -F'\t' '{printf "%s\t%s\n", $2, $3}' "$WORK" \
  | xargs -P "$JOBS" -L 1 -I '{}' bash -c '
      line="{}"
      rel="${line%%	*}"
      url="${line#*	}"
      exec "'"$0"'" --worker "$rel" "$url"
    '

echo
echo "All done."
du -sh "${RAW_DIR}"/* 2>/dev/null || true
