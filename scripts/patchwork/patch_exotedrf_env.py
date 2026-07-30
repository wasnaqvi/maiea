#!/usr/bin/env python
"""Apply the Patchwork compatibility patches to an exoTEDRF environment.

Three upstream/packaging defects have to be worked around before exoTEDRF
2.3.1 + jwst 1.17.1 will run on DRAC Fir. Each patch is inert with
respect to numerical results -- they fix imports and a resume bug, never
the reduction itself -- but they ARE modifications to the frozen executor
and therefore belong in the Patchwork provenance record.

Run once per environment (idempotent; safe to re-run after any pip
install, which may revert a patched file):

    source ~/envs/exotedrf/bin/activate
    python scripts/patchwork/patch_exotedrf_env.py

Patches
-------
1. ``jwst/__init__.py`` -- the ``+computecanada`` wheel omits
   ``__version_commit__``, which ``jwst/stpipe/core.py`` imports at module
   level. Cosmetic provenance string; stamped into FITS headers only.

2. ``stdatamodels/exceptions.py`` -- newer CRDS imports
   ``ValidationWarning`` from here, but stdatamodels 2.2.0 (the only
   version jwst 1.17.1 permits, ``>=2.2.0,<2.3.0``) still defines it in
   ``stdatamodels.validate``. Re-export shim. Redundant once CRDS is
   pinned to 12.1.11, kept as insurance.

3. ``exotedrf/stage2.py`` BadPixStep resume bug -- when every segment's
   output already exists and ``force_redo=False``, the processing branch
   never runs, so ``to_flag`` is unbound when the step tries to save the
   hot-pixel map (UnboundLocalError). Binds it up front and skips the
   save when nothing was recomputed, so the existing hot-pixel map is
   preserved rather than overwritten with None. Without this, a reduction
   can never be resumed -- fatal for jobs that hit a walltime limit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Patch the tree that will actually RUN. When ASTER_EXOTEDRF_REPO is set the
# toolkit shadows the installed release with that checkout, so the checkout
# is what needs patching -- the exoTEDRF optimizer branch carries the same
# BadPixStep resume bug as the PyPI build.
_REPO = os.environ.get("ASTER_EXOTEDRF_REPO")
if _REPO:
    _REPO = os.path.expanduser(_REPO)
    if (Path(_REPO) / "exotedrf" / "__init__.py").exists():
        sys.path.insert(0, _REPO)


def _package_dir(name: str) -> Path | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return Path(module.__file__).parent


def _git_warning(path: Path) -> str:
    """Patching a git working tree leaves it dirty; say so explicitly."""
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return (
                f"\nNOTE: {path} is inside the git checkout {parent}.\n"
                "      These edits will show up in `git status`. Commit them\n"
                "      to your fork so the fix survives a fresh clone, or\n"
                "      re-run this script after every `git pull`."
            )
    return ""


def patch_jwst_version_commit() -> str:
    pkg = _package_dir("jwst")
    if pkg is None:
        return "SKIP  jwst not importable"
    init = pkg / "__init__.py"
    text = init.read_text()
    if "__version_commit__" in text:
        return "OK    jwst.__version_commit__ already present"
    with init.open("a") as fh:
        fh.write('\n# Patchwork: absent from the +computecanada wheel;\n'
                 '# jwst/stpipe/core.py imports it at module level.\n'
                 '__version_commit__ = ""\n')
    return "FIXED jwst.__version_commit__ appended"


def patch_stdatamodels_exceptions() -> str:
    pkg = _package_dir("stdatamodels")
    if pkg is None:
        return "SKIP  stdatamodels not importable"
    target = pkg / "exceptions.py"
    if target.exists():
        return "OK    stdatamodels.exceptions already present"
    try:
        from stdatamodels.validate import ValidationWarning  # noqa: F401
    except ImportError:
        return ("SKIP  ValidationWarning not in stdatamodels.validate either "
                "-- inspect this version by hand")
    target.write_text(
        "# Patchwork shim: newer CRDS imports ValidationWarning from here,\n"
        "# but stdatamodels 2.2.0 still defines it in .validate.\n"
        "from stdatamodels.validate import ValidationWarning  # noqa: F401\n"
    )
    return "FIXED stdatamodels/exceptions.py shim written"


# The unguarded save, and its replacement.
_BADPIX_OLD = """        if save_results is True:
            # Save hot pixel mask.
            outfile = self.output_dir + self.fileroot_noseg + 'hot_pixels.npy'
            np.save(outfile, to_flag)"""

_BADPIX_NEW = """        if save_results is True and to_flag is not None:
            # Patchwork: guard added. When every segment was skipped on a
            # resume, to_flag is unbound (UnboundLocalError) and there is
            # nothing new to save -- the existing map on disk is still
            # valid and must not be overwritten with None.
            # Save hot pixel mask.
            outfile = self.output_dir + self.fileroot_noseg + 'hot_pixels.npy'
            np.save(outfile, to_flag)"""

# NOTE: several exoTEDRF step classes share this boilerplate (BackgroundStep
# has an identical block earlier in the file), so this anchor MUST be applied
# inside the BadPixStep class only -- a plain str.replace hits BackgroundStep
# and leaves the real bug in place.
_LOOP_OLD = """        first_time = True
        for i, segment in enumerate(self.datafiles):"""

_LOOP_NEW = """        first_time = True
        # Patchwork: bind before the loop so a fully resumed step (every
        # segment already on disk, so the else-branch never runs) cannot
        # raise UnboundLocalError at the hot-pixel save below.
        to_flag = None
        for i, segment in enumerate(self.datafiles):"""

_MARKER = "Patchwork: bind before the loop"


def patch_badpixstep_resume() -> str:
    pkg = _package_dir("exotedrf")
    if pkg is None:
        return "SKIP  exotedrf not importable"
    stage2 = pkg / "stage2.py"
    backup = stage2.with_suffix(".py.patchwork-orig")
    text = stage2.read_text()

    # Always rebuild from pristine source. An earlier version of this script
    # anchored the loop edit with str.replace(..., 1), which patched
    # BackgroundStep instead of BadPixStep; restoring first repairs that.
    if backup.exists():
        text = backup.read_text()
    else:
        backup.write_text(text)

    if _BADPIX_OLD not in text:
        return ("SKIP  BadPixStep hot-pixel save does not match the expected "
                "exoTEDRF 2.3.1 form -- inspect stage2.py by hand")

    # Restrict the loop edit to the BadPixStep class body.
    cls = text.index("class BadPixStep")
    head, body = text[:cls], text[cls:]
    if _LOOP_OLD not in body:
        return ("SKIP  BadPixStep loop does not match the expected exoTEDRF "
                "2.3.1 form -- inspect stage2.py by hand")
    body = body.replace(_LOOP_OLD, _LOOP_NEW, 1)
    text = head + body
    text = text.replace(_BADPIX_OLD, _BADPIX_NEW, 1)

    # Verify the initialiser really landed inside BadPixStep, before the guard.
    check = text[text.index("class BadPixStep"):]
    if not (0 < check.index(_MARKER) < check.index("to_flag is not None")):
        return "SKIP  patch verification failed -- stage2.py left unchanged"

    stage2.write_text(text)
    return f"FIXED BadPixStep resume guard applied (backup: {backup.name})"


def main() -> int:
    print(f"python      : {sys.executable}")
    for name in ("jwst", "stdatamodels", "exotedrf", "crds", "numpy"):
        pkg = _package_dir(name)
        print(f"  {name:<13}: {pkg if pkg else '(not importable)'}")
    print()

    results = [
        patch_jwst_version_commit(),
        patch_stdatamodels_exceptions(),
        patch_badpixstep_resume(),
    ]
    for line in results:
        print(line)

    pkg = _package_dir("exotedrf")
    if pkg is not None:
        note = _git_warning(pkg)
        if note:
            print(note)

    if any(r.startswith("SKIP") for r in results):
        print("\nOne or more patches were skipped -- read the notes above.")
        return 1
    print("\nEnvironment patched. Re-run after any pip install that touches "
          "jwst, stdatamodels, or exotedrf -- and after any git pull of the "
          "exoTEDRF checkout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
