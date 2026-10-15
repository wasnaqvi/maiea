#!/usr/bin/env python3
"""
Cold-trap gap sweep + duplication check, built on ASTER's own tools.

Fills slot 5 of the Steam Rising sample: an unobserved, mass-measured
sub-Neptune in the 450-620 K window where T_crit is expected to sit.

RUN THIS ON THE MAC (not in the Cowork VM) -- the NASA Exoplanet Archive and
MAST are blocked by the org egress policy from the sandboxed environments, so
the query fails there at the network layer, not in the code.

    cd ~/MAIA/aster
    source .venv/bin/activate      # or however you normally activate
    python gap_sweep_aster.py

Writes gap_candidates.csv next to itself.
"""

import sys, csv, math

sys.path.insert(0, ".")

from aster_toolkit.data_acquisition.exoarchive import find_exoplanets_by_condition

# ---------------------------------------------------------------- selection
# Criteria carried over from STEAM_RISING_draft_v2 §Sample, with the
# temperature window moved into the gap that actually constrains T_crit.
CONDITIONS = [
    "pl_eqt between 450 and 620",      # the unsampled half of the 383-600 K gap
    "pl_rade between 1.8 and 3.0",     # sub-Neptune, matches the controlled band
    "pl_bmasse is not null",
    "pl_bmasseerr1 is not null",
    "tran_flag = 1",
    "sy_jmag < 11.0",                  # JWST-accessible
    "st_teff < 4800",                  # K/M hosts; tighten to 4000 for M-only
]

COLUMNS = [
    "pl_name", "pl_eqt", "pl_rade", "pl_bmasse", "pl_bmasseerr1",
    "st_rad", "st_teff", "sy_jmag", "pl_orbper", "pl_trandur", "st_spectype",
]


def tsm(rp, mp, rs, teq, jmag):
    """Kempton et al. 2018 Eq. 1."""
    if rp < 1.5:
        sf = 0.190
    elif rp < 2.75:
        sf = 1.26
    elif rp < 4.0:
        sf = 1.28
    else:
        sf = 1.15
    return sf * (rp ** 3 * teq) / (mp * rs ** 2) * 10 ** (-jmag / 5)


def scale_height_km(teq, mp, rp, mu):
    kB, u, G = 1.380649e-23, 1.66053907e-27, 6.674e-11
    g = G * (mp * 5.972e24) / (rp * 6.371e6) ** 2
    return kB * teq / (mu * u * g) / 1000.0


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


rows = find_exoplanets_by_condition(
    conditions=CONDITIONS, return_columns=COLUMNS, table="pscomppars"
)
print(f"archive returned {len(rows)} rows in the gap\n")

out = []
for r in rows:
    rp, mp, rs = f(r["pl_rade"]), f(r["pl_bmasse"]), f(r["st_rad"])
    teq, j, me = f(r["pl_eqt"]), f(r["sy_jmag"]), f(r["pl_bmasseerr1"])
    if None in (rp, mp, rs, teq, j) or mp <= 0 or rs <= 0:
        continue
    t = tsm(rp, mp, rs, teq, j)
    sig = mp / me if me else float("nan")
    h2 = scale_height_km(teq, mp, rp, 2.3)
    st = scale_height_km(teq, mp, rp, 18.0)
    # ppm per scale height
    dppm = 2 * (rp * 6.371e6) * (h2 * 1e3) / (rs * 6.957e8) ** 2 * 1e6
    out.append(dict(name=r["pl_name"], teq=teq, rp=rp, mp=mp, msig=sig,
                    rs=rs, jmag=j, tsm=t, H2=h2, Hsteam=st, ppm_per_H=dppm,
                    spt=r.get("st_spectype") or "", per=r.get("pl_orbper")))

# Keep what the proposal can actually use: TSM above the Kempton threshold
# and a mass good enough to interpret mu.
out = [o for o in out if o["tsm"] >= 90 and (o["msig"] != o["msig"] or o["msig"] >= 3)]
out.sort(key=lambda o: -o["tsm"])

hdr = f"{'Planet':<18}{'Teq':>6}{'Rp':>6}{'Mp':>7}{'M sig':>7}{'TSM':>7}{'H(H2)':>8}{'ppm/H':>8}  {'SpT'}"
print(hdr); print("-" * len(hdr))
for o in out:
    print(f"{o['name']:<18}{o['teq']:6.0f}{o['rp']:6.2f}{o['mp']:7.2f}"
          f"{o['msig']:7.1f}{o['tsm']:7.1f}{o['H2']:8.0f}{o['ppm_per_H']:8.1f}  {o['spt']}")

with open("gap_candidates.csv", "w", newline="") as fh:
    if out:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
print(f"\nwrote gap_candidates.csv ({len(out)} candidates)")

# ------------------------------------------------- duplication check (MAST)
# Run this for every survivor AND for TOI-270 c before committing to it.
print("\n--- MAST duplication check ---")
try:
    from aster_toolkit.data_acquisition.mast import SearchMastJwstObservations
    check = [o["name"] for o in out] + ["TOI-270 c"]
    for name in check:
        try:
            res = SearchMastJwstObservations(target_name=name)._run()
            print(f"\n{name}:\n{res}")
        except Exception as e:
            print(f"{name}: query failed -- {type(e).__name__}: {e}")
except Exception as e:
    print(f"MAST tool unavailable: {type(e).__name__}: {e}")
    print("Fall back to the MAST Duplicate Observation Search in APT.")
