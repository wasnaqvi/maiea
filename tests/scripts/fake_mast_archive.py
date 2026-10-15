"""Shared fake MAST archive (ERS 1366 / WASP-39 b with decoys).

Used by test_exotedrf_parity.py (in-process, HTTP layer patched) and by the
GUI end-to-end test (served over real HTTP by tests/e2e/fake_mast_server.py).
"""
import math
import re
import time

WASP39 = (217.32667, -3.44450)
OTHER = (10.0, 20.0)
HD12572 = (30.9050061, 21.2813274)     # HIP 9618 b and c; MAST names it "HD-12572b"


def now_mjd() -> float:
    return time.time() / 86400.0 + 40587.0


# Release dates (MJD). Everything ERS 1366 went public in 2022. GO 8004 o003
# was released 36 days ago; o004 is locked until LOCKED_RELEASE_MJD, which a
# test may move to a few seconds from now to watch the release-day email go
# out. Access is computed per request, like MAST: PUBLIC once released.
OLD_RELEASE_MJD = 59800.0
RECENT_RELEASE_MJD = now_mjd() - 36
LOCKED_RELEASE_MJD = now_mjd() + 89

OBS = [
    # obsid, obs_id, instrument, filters, program, coords, visit prefixes
    ("101", "jw01366-o003_t001_nirspec_clear-prism-s200a1-sub512",
     "NIRSPEC/SLIT", "CLEAR;PRISM", "1366", WASP39, ["jw01366003001"]),
    ("102", "jw01366-o004_t001_nirspec_f290lp-g395h-s1600a1-sub2048",
     "NIRSPEC/SLIT", "F290LP;G395H", "1366", WASP39, ["jw01366004001", "jw01366004002"]),
    ("103", "jw01366-o001_t001_niriss_clear-gr700xd-substrip256",
     "NIRISS/SOSS", "CLEAR;GR700XD", "1366", WASP39, ["jw01366001001"]),
    ("104", "jw01366-o002_t001_nircam_f322w2-grismr-subgrism256",
     "NIRCAM/GRISM", "F322W2;GRISMR", "1366", WASP39, ["jw01366002001"]),
    ("105", "jw02783-o001_t001_nirspec_f290lp-g395h-s1600a1-sub2048",
     "NIRSPEC/SLIT", "F290LP;G395H", "2783", WASP39, ["jw02783001001"]),
    ("106", "jw01366-o099_t009_nirspec_f290lp-g395h-s1600a1-sub2048",
     "NIRSPEC/SLIT", "F290LP;G395H", "1366", OTHER, ["jw01366099001"]),
    ("201", "jw08004-o003_t002_nirspec_f290lp-g395h-s1600a1-sub2048",
     "NIRSPEC/SLIT", "F290LP;G395H", "8004", HD12572, ["jw08004003001"]),
    ("202", "jw08004-o004_t002_nirspec_f290lp-g395h-s1600a1-sub2048",
     "NIRSPEC/SLIT", "F290LP;G395H", "8004", HD12572, ["jw08004004001"]),
]

# For the fake Exoplanet Archive (TAP) the e2e GUI test points ASTER at.
PLANETS = [
    {"pl_name": "WASP-39 b", "hostname": "WASP-39", "ra": WASP39[0], "dec": WASP39[1],
     "pl_rade": 14.34, "pl_eqt": 1166, "tran_flag": 1},
    {"pl_name": "HIP 9618 b", "hostname": "HIP 9618", "ra": HD12572[0], "dec": HD12572[1],
     "pl_rade": 3.83, "pl_eqt": 641, "tran_flag": 1},
    {"pl_name": "HIP 9618 c", "hostname": "HIP 9618", "ra": HD12572[0], "dec": HD12572[1],
     "pl_rade": 3.43, "pl_eqt": 470, "tran_flag": 1},
]


def _release(obsid: str) -> float:
    return {"201": RECENT_RELEASE_MJD, "202": LOCKED_RELEASE_MJD}.get(obsid, OLD_RELEASE_MJD)


def obs_rows():
    return [
        {"obsid": o, "obs_id": oid, "instrument_name": inst, "filters": flt,
         "proposal_id": prog, "s_ra": c[0], "s_dec": c[1],
         "obs_collection": "JWST", "dataproduct_type": "timeseries",
         "calib_level": 3,
         "target_name": {WASP39: "WASP-39", HD12572: "HD-12572b"}.get(c, "OTHER"),
         "proposal_pi": "Batalha, Natasha" if prog == "8004" else "Batalha, Natalie",
         "t_min": _release(o) - 365, "t_obs_release": _release(o),
         "dataRights": "PUBLIC" if _release(o) <= now_mjd() else "EXCLUSIVE_ACCESS"}
        for o, oid, inst, flt, prog, c, _ in OBS
    ]


# Each product row carries its OWN filters. Science exposures and the
# observation-level products share the observation's; target acquisition is
# taken through other optics. NIRSpec's F110W;MIRROR is the live value (ERS
# 1366 obsid 233644122, 2026-10-15); the NIRISS/NIRCam ones are illustrative.
TA_FILTERS = {"NIRSPEC": "F110W;MIRROR", "NIRISS": "F480M", "NIRCAM": "F335M"}


def products_for(obsid):
    _, oid, inst, flt, *_r, prefixes = next(x for x in OBS if x[0] == obsid)
    ta_filters = TA_FILTERS[inst.split("/")[0]]
    dets = ("nis",) if "NIRISS" in inst else ("nrcalong",) if "NIRCAM" in inst else ("nrs1", "nrs2")
    out = []
    for p in prefixes:
        for seg in (1, 2):
            for d in dets:
                for kind, sub in (("uncal", "UNCAL"), ("rateints", "RATEINTS")):
                    name = f"{p}_04101_00001-seg{seg:03d}_{d}_{kind}.fits"
                    out.append({"obsID": obsid, "productFilename": name,
                                "dataURI": f"mast:JWST/product/{name}",
                                "productSubGroupDescription": sub,
                                "productType": "SCIENCE", "dataRights": "PUBLIC",
                                "dataproduct_type": "timeseries", "filters": flt,
                                "size": 1})
        # Target acquisition in the same visit. MAST types it 'timeseries'
        # like the science (verified live) — only its filters differ.
        ta = f"{p}_02101_00001_{dets[0]}_uncal.fits"
        out.append({"obsID": obsid, "productFilename": ta,
                    "dataURI": f"mast:JWST/product/{ta}",
                    "productSubGroupDescription": "UNCAL", "productType": "SCIENCE",
                    "dataRights": "PUBLIC", "dataproduct_type": "timeseries",
                    "filters": ta_filters, "size": 1})
    x1d = f"{oid}_x1dints.fits"
    out.append({"obsID": obsid, "productFilename": x1d, "dataURI": f"mast:JWST/product/{x1d}",
                "productSubGroupDescription": "X1DINTS", "productType": "SCIENCE",
                "dataRights": "PUBLIC", "dataproduct_type": "timeseries",
                "filters": flt, "size": 1})
    return out


def sep_deg(a, b):
    r1, d1, r2, d2 = map(math.radians, (*a, *b))
    h = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1, math.sqrt(h))))


def resolve(name):
    key = name.replace(" ", "").upper()
    if key.startswith("WASP-39"):
        return WASP39
    if key in ("HD-12572B", "HD12572B", "HD12572", "HIP9618B", "HIP9618C", "HIP9618"):
        return HD12572
    raise ValueError(name)


_BETWEEN_RE = re.compile(r"\b(\w+)\s+between\s+(-?[\d.]+)\s+and\s+(-?[\d.]+)", re.I)
_CMP_RE = re.compile(r"\b(pl_rade|pl_eqt|tran_flag)\s*(>=|<=|=|>|<)\s*(-?[\d.]+)", re.I)
_IN_RE = re.compile(r"\bpl_name\s+IN\s*\(([^)]*)\)", re.I)
_EQ_RE = re.compile(r"\b(pl_name|hostname)\s*=\s*'([^']*)'", re.I)


def tap_rows(adql: str) -> list[dict]:
    """Planets matching the handful of ADQL shapes ASTER sends: RA/Dec boxes,
    numeric bounds, pl_name IN (...), and pl_name/hostname = '...'."""
    rows = list(PLANETS)
    for col, lo, hi in _BETWEEN_RE.findall(adql):
        rows = [r for r in rows if float(lo) <= float(r[col.lower()]) <= float(hi)]
    ops = {">=": float.__ge__, "<=": float.__le__, "=": float.__eq__,
           ">": float.__gt__, "<": float.__lt__}
    for col, op, val in _CMP_RE.findall(adql):
        rows = [r for r in rows if ops[op](float(r[col.lower()]), float(val))]
    if m := _IN_RE.search(adql):
        names = set(re.findall(r"'([^']*)'", m[1]))
        rows = [r for r in rows if r["pl_name"] in names]
    eqs = _EQ_RE.findall(adql)
    if eqs:
        rows = [r for r in rows if any(r[c.lower()] == v for c, v in eqs)]
    return rows
