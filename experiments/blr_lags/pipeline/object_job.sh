#!/bin/sh
# One object: MAP+Laplace fit, then for each selected line: lag profile -> diagnostic figure.
R=$(cd "$(dirname "$0")" && pwd)
PP=${QVC_PYTHONPATH:-}  # optional overlay, e.g. a qvc worktree src/ and an equinox 0.12.2 install
oid=$1
mkdir -p $R/fits $R/logs
cd "$R/../../.."  # qvc repo root (data/ paths are relative to it)
if [ ! -f $R/fits/${oid}_laplace.npz ]; then
  PYTHONPATH=$PP python -u $R/fit_object.py $oid $R/fits > $R/logs/${oid}_fit.log 2>&1 || { echo "FAIL fit $oid"; exit 0; }
fi
grep "^$oid," $R/selection.csv | while IFS=, read o z band logL var line; do
  mkdir -p $R/profiles/$line $R/plots/$line
  prof=$R/profiles/$line/${oid}_${band}_profile.json
  [ -f $prof ] || PYTHONPATH=$PP python -u $R/lag_profile.py $oid $band $R/fits/${oid}_laplace.npz $R/profiles/$line > $R/logs/${oid}_${line}_profile.log 2>&1 || { echo "FAIL profile $oid $line"; continue; }
  PYTHONPATH=$PP python -u $R/diagnostic_figure.py $oid $line $band $logL $R/fits/${oid}_laplace.npz $prof $R/plots/$line > $R/logs/${oid}_${line}_figure.log 2>&1 || { echo "FAIL figure $oid $line"; continue; }
  echo "done $oid $line $(grep -h DONE $R/logs/${oid}_${line}_profile.log | sed 's/.*DONE //' | cut -c1-70)"
done
