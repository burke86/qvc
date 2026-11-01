#!/bin/sh
# Run prm_lag_fit.py over prm_selection.csv (object, line), in file order, N at a time; finished objects are skipped, so it can be restarted.
# usage (from anywhere): sh prm_run.sh <out_dir> [n_parallel=3]
P=$(cd "$(dirname "$0")" && pwd)
OUT=$(mkdir -p "$1" && cd "$1" && pwd); NPAR=${2:-3}
mkdir -p "$OUT/out" "$OUT/logs"
cd "$P/../../.."  # qvc repo root (data/ paths are relative to it)
echo "start $(date '+%I:%M %p'): $(($(wc -l < "$P/prm_selection.csv") - 1)) object-lines, $NPAR at a time"
export P OUT
tail -n +2 "$P/prm_selection.csv" | cut -d, -f1,2 | tr , ' ' | xargs -P "$NPAR" -L 1 sh -c '
  [ -f "$OUT/out/$0_$1.json" ] && exit 0
  python -u "$P/prm_lag_fit.py" "$0" "$1" "$OUT/out" > "$OUT/logs/$0_$1.log" 2>&1 && grep -h DONE "$OUT/logs/$0_$1.log" || echo "FAIL $0 $1 (see $OUT/logs/$0_$1.log)"
'
echo "ALL DONE $(date '+%I:%M %p'): $(ls "$OUT"/out/*.json 2>/dev/null | wc -l) results in $OUT/out"
