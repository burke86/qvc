#!/bin/sh
# Wait for the seeing sidecars, then run all objects (3 at a time) and keep the per-line R-L plots current.
R=$(cd "$(dirname "$0")" && pwd)
S=$R/..
while [ ! -f $S/seeing/dr16s82_sdssSeeing.parquet ] || [ ! -f $S/seeing/dr16s82_ztfSeeing.parquet ]; do sleep 30; done
echo "seeing sidecars ready at $(date '+%I:%M %p'); starting fits"
cat $R/job_order.txt | xargs -P 3 -n 1 $R/object_job.sh &
XPID=$!
last=""
while kill -0 $XPID 2>/dev/null; do
  now=$(ls $R/profiles/*/*_profile.json 2>/dev/null | wc -l)
  if [ "$now" != "$last" ]; then
    last=$now
    for line in Halpha Hbeta MgII CIV; do
      [ -n "$(ls $R/profiles/$line/*_profile.json 2>/dev/null)" ] && python $R/analyze_line.py $R $line > $R/plots/$line/summary.txt 2>&1
    done
  fi
  sleep 30
done
for line in Halpha Hbeta MgII CIV; do python $R/analyze_line.py $R $line > $R/plots/$line/summary.txt 2>&1; done
echo "ALL DONE at $(date '+%I:%M %p'): fits $(ls $R/fits/*.json | wc -l), profiles $(ls $R/profiles/*/*_profile.json | wc -l), figures $(ls $R/plots/*/*_diagnostic.pdf | wc -l)"
