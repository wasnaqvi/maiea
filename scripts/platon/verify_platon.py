"""Check a PLATON 7 install: versions, data, device, and one forward model.

    platon-python ~/aster/maiea/scripts/platon/verify_platon.py

Prints READY only when every check passes. On a GPU node it also requires
JAX to be on the GPU, since a silent CPU fallback is ~100x slower and would
blow a retrieval's walltime.
"""
import os
import sys
import time

problems = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL':4s}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        problems.append(name)


import numpy as np
import jax
import platon
from platon._get_data import get_data_if_needed

check("platon version", platon.__version__ == "7.0", platon.__version__)
check("platon source", True, os.path.dirname(platon.__file__))
check("numpy>=2", int(np.__version__.split(".")[0]) >= 2, np.__version__)
check("jax", True, f"jax {jax.__version__}")

pkg = os.path.dirname(platon.__file__)
md5_file = os.path.join(pkg, "md5sum")
have_data = os.path.isdir(os.path.join(pkg, "data")) and os.path.isfile(md5_file)
md5 = open(md5_file).read().strip() if have_data else ""
check("opacity data present, md5 current", have_data and md5 == platon.__md5sum__,
      md5 or "missing -- run setup_platon_fir.sh on a login node")
if not have_data:
    sys.exit("NOT READY")
get_data_if_needed()  # no-op when present; never downloads here

on_gpu_node = bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
backend = jax.default_backend()
check("jax backend", backend == "gpu" or not on_gpu_node,
      f"{backend} {jax.devices()}" + ("" if not on_gpu_node or backend == "gpu"
                                      else "  (GPU allocated but jax fell back to CPU)"))

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.TP_profile import Profile
from platon.constants import M_jup, R_jup, R_sun

t0 = time.time()
calc = TransitDepthCalculator()
t_load = time.time() - t0
bins = np.array([[2.87e-6 + 1e-8 * i, 2.87e-6 + 1e-8 * (i + 1)] for i in range(210)])
calc.change_wavelength_bins(bins)
kw = dict(t_p_profile=Profile.isothermal(500), star_radius=0.7 * R_sun,
          planet_mass=0.03 * M_jup, planet_radius=0.22 * R_jup,
          logZ=2, CO_ratio=0.53)
calc.compute_depths(**kw)  # first call compiles
t0 = time.time()
wl, depths = calc.compute_depths(**kw)[:2]
t_run = time.time() - t0
finite = np.all(np.isfinite(depths))
check("forward model (G395H range, 100x solar, 500 K)",
      finite and 1e-4 < np.median(depths) < 1e-1,
      f"median depth {1e6 * np.median(depths):.0f} ppm, span "
      f"{1e6 * (depths.max() - depths.min()):.0f} ppm, load {t_load:.0f} s, "
      f"model {1e3 * t_run:.0f} ms")

print("READY" if not problems else "NOT READY: " + ", ".join(problems))
sys.exit(1 if problems else 0)
