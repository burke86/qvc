"""Photometric-RM selection for prm_lag_fit.py: Halpha, Hbeta, Mg II and C IV.

Cuts, in order, per line:
1. chi2_g > 20 S82 quasar with a good-quality DR16Q line (count_strong.py, dr16q_lines.line_quality());
2. in-band line/continuum ratio a > MIN_FRAC, in the band with the largest a;
3. line-band per-point variability S/N > MIN_SNR (ML excess amplitude over the typical error, line_snr.py); the
   SDSS-RM test bands span 3.6-7.9 (16-84%), so 3 is the low end of the validated range;
4. point source: drop objects flagged extended in morphology/morphology.csv (query_morphology.py: error-weighted
   psfMag - cModelMag over g, r, i > 0.145 and > 3 sigma from zero). Objects without SDSS morphology are kept.
Writes prm_selection.csv (objectId, line, z, band, frac, frac_lnerr, snr, extended), ordered Halpha, Hbeta, Mg II,
C IV (the run order) and by decreasing a within a line. frac_lnerr is the fractional DR16Q EW error, the measurement
part of the prior width on a in prm_lag_fit.py.
Run from the qvc repo root, after count_strong.py and line_snr.py:
    python experiments/blr_lags/pipeline/prm_select.py [--min-snr X] [--keep-extended]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from dr16q_lines import LINE_COL, load_dr16q, match_dr16q

S = Path(__file__).resolve().parent
LINES = ["Halpha", "Hbeta", "MgII", "CIV"]
MIN_FRAC = 0.25
MIN_SNR = float(sys.argv[sys.argv.index("--min-snr") + 1]) if "--min-snr" in sys.argv else 3.0
KEEP_EXTENDED = "--keep-extended" in sys.argv

fr = pd.read_csv(S / "chi20_line_fractions.csv", dtype={"objectId": str})
sel = fr[fr.good & fr.line.isin(LINES) & (fr.frac > MIN_FRAC)].sort_values("frac").groupby(["objectId", "line"]).tail(1)
counts = {"a": sel.line.value_counts()}
snr = pd.read_csv(S / "line_snr.csv", dtype={"objectId": str})[["objectId", "band", "snr"]]
sel = sel.merge(snr, on=["objectId", "band"], how="left")
sel = sel[sel.snr > MIN_SNR]
counts["snr"] = sel.line.value_counts()
morph = pd.read_csv(S.parent / "morphology" / "morphology.csv", dtype={"objectId": str})[["objectId", "extended"]]
sel = sel.merge(morph, on="objectId", how="left")
if not KEEP_EXTENDED:
    sel = sel[sel.extended != True]
counts["final"] = sel.line.value_counts()

cat = pd.read_parquet("data/S82/Catalog.parquet", columns=["objectId", "PLATE", "MJD", "FIBERID", "Z_SYS"])
cat["objectId"] = cat.objectId.astype(str)
d = load_dr16q()
m = match_dr16q(cat[cat.objectId.isin(sel.objectId)], d).drop_duplicates("objectId")
idx = dict(zip(m.objectId, m["index"]))
sel["z"] = sel.objectId.map(dict(zip(m.objectId, m.Z_SYS)))
sel["frac_lnerr"] = [float(d[LINE_COL[l] + "_ERR"][idx[o]][5] / d[LINE_COL[l]][idx[o]][5]) for o, l in zip(sel.objectId, sel.line)]
sel["order"] = sel.line.map({l: i for i, l in enumerate(LINES)})
out = sel.sort_values(["order", "frac"], ascending=[True, False])
out = out[["objectId", "line", "z", "band", "frac", "frac_lnerr", "snr", "extended"]]
out.to_csv(S / "prm_selection.csv", index=False)
print(pd.DataFrame(counts).reindex(LINES).fillna(0).astype(int).rename(
    columns={"a": f"a > {MIN_FRAC}", "snr": f"+ S/N > {MIN_SNR:g}", "final": "+ point source" if not KEEP_EXTENDED else "(extended kept)"}).to_string())
print(f"{len(out)} object-lines -> prm_selection.csv")
