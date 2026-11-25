"""Backfill the prior ranges into summary.json for runs made before the
driver recorded them (2026-11-25 survey). Login node is fine: it only
unpickles result.pkl, no likelihood calls.

    platon-python write_priors.py ~/scratch/platon
"""
import json
import pickle
import sys
from pathlib import Path

for pkl in sorted(Path(sys.argv[1]).expanduser().glob("*/*/result.pkl")):
    summ = pkl.parent / "summary.json"
    if not summ.exists():
        continue
    d = json.loads(summ.read_text())
    if "priors" in d:
        continue
    r = pickle.load(open(pkl, "rb"))
    d["priors"] = {n: [float(r.fit_info.all_params[n].low_lim),
                       float(r.fit_info.all_params[n].high_lim)]
                   for n in r.fit_info.fit_param_names}
    summ.write_text(json.dumps(d, indent=2))
    print("priors ->", summ)
