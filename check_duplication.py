#!/usr/bin/env python3
"""
JWST duplication check for the Steam Rising candidates -- CORRECTED.

The previous run used SearchMastJwstObservations(target_name="GJ 1214 b"),
which does a literal substring match against MAST's `target_name` column.
MAST files observations under the HOST STAR name as the PI typed it
("GJ1214", "TOI-270"), never "GJ 1214 b", so that query returns nothing for
every planet -- including ones with famous JWST data. It is a false negative
generator, not a duplication check.

mast.py says so directly (search_jwst_observations docstring):
    "MAST observations are usually filed under host-star target names, so the
     default search is coordinate-centered."

So: resolve coordinates and cone-search instead.

POSITIVE CONTROL: GJ 1214 b has a published JWST MIRI phase curve
(Kempton et al. 2023, Nature 620, 67). If the control returns zero
observations, the query is broken and every other result is meaningless.
This script refuses to report on the real targets until the control passes.

    python ~/MAIA/aster/check_duplication.py 2>&1 | tee ~/MAIA/aster/dupcheck.log
"""

import sys, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from aster_toolkit.data_acquisition.mast import search_jwst_observations

RADIUS_DEG = 0.02          # ~72", comfortably larger than any proper motion
CONTROL = "GJ 1214 b"      # must return observations or the query is broken

TARGETS = ["GJ 1214 b", "TOI-2443 b", "TOI-270 c", "TOI-178 g", "TOI-270 d"]
# TOI-270 d is included deliberately as a SECOND control: it is known-observed
# (NIRISS SOSS + NIRSpec G395H + MIRI) and shares a star with TOI-270 c, so it
# shows what a real detection looks like at the exact coordinates we care about.


def query(name):
    try:
        return search_jwst_observations(name, radius_deg=RADIUS_DEG), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def show(name, rows):
    if not rows:
        print(f"  -> NO JWST observations within {RADIUS_DEG} deg")
        return
    by = collections.defaultdict(lambda: {"n": 0, "targets": set(), "modes": set()})
    for r in rows:
        pid = str(r.get("proposal_id") or "?")
        inst = str(r.get("instrument_name") or "?")
        key = (pid, inst)
        by[key]["n"] += 1
        if r.get("target_name"):
            by[key]["targets"].add(str(r["target_name"]))
        if r.get("filters"):
            by[key]["modes"].add(str(r["filters"]))
    print(f"  -> {len(rows)} observation record(s):")
    for (pid, inst), d in sorted(by.items()):
        tgt = ", ".join(sorted(d["targets"])) or "-"
        mode = ", ".join(sorted(d["modes"])[:3]) or "-"
        print(f"     GO/GTO {pid:<8} {inst:<22} n={d['n']:<4} target_name={tgt}")
        print(f"       {'':<8} {'':<22} modes: {mode}")


print("=" * 78)
print("POSITIVE CONTROL")
print("=" * 78)
rows, err = query(CONTROL)
if err:
    sys.exit(f"CONTROL FAILED TO RUN: {err}\nCannot trust any result. Stopping.")
print(f"{CONTROL}:")
show(CONTROL, rows)

if not rows:
    sys.exit(
        "\nCONTROL RETURNED ZERO OBSERVATIONS.\n"
        "GJ 1214 b definitely has JWST data, so the query is broken -- most\n"
        "likely coordinate resolution or the cone radius. Do NOT interpret any\n"
        "target as unobserved. Fix the query first.\n"
    )
print("\nControl passed: the query does find real JWST data. Results below are meaningful.\n")

print("=" * 78)
print(f"DUPLICATION CHECK (cone search, r = {RADIUS_DEG} deg)")
print("=" * 78)
for name in TARGETS:
    print(f"\n{name}:")
    rows, err = query(name)
    if err:
        print(f"  -> QUERY ERROR: {err}")
        continue
    show(name, rows)

print("""
=============================================================================
READING THIS OUTPUT

A cone search returns everything at that POSITION, so for a multi-planet
system it returns observations of the star regardless of which planet was
the science target. TOI-270 c and TOI-270 d share coordinates.

So an empty result for TOI-270 c is only meaningful if TOI-270 d is NON-empty
(it should be: NIRISS SOSS, NIRSpec G395H, MIRI). If BOTH come back empty,
the position resolution is wrong -- not the sky.

If TOI-270 returns observations, check each proposal_id: a program targeting
planet d at the same coordinates is NOT a duplication of a planet c transit.
Transit observations are timed to a specific planet's ephemeris, so confirm
against the program abstract before concluding either way.
=============================================================================
""")
