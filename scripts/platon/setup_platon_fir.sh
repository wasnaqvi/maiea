#!/bin/bash
# Build (or rebuild) the PLATON 7 environment on DRAC Fir. Run on a LOGIN
# node: it needs the internet for pip, git and the ~11 GB opacity download,
# and Fir compute nodes have none.
#
#     bash ~/aster/maiea/scripts/platon/setup_platon_fir.sh
#
# Layout (each overridable by environment variable):
#   PLATON_ENV  ~/envs/platon-env                 venv, next to ~/envs/exotedrf
#   PLATON_SRC  /project/def-ncowan/wasi/platon   source checkout + data/
#               (the data is ~11 GB zipped; home has no room, and scratch
#               is purged, so it lives on /project)
#   PLATON_REF  v7.0                              git tag, pinned
#
# Reproducibility: requirements-platon-fir.txt (next to this script) is the
# lock. If it exists, the env is built from it exactly; if not, the env is
# resolved fresh and the lock is written for you to commit. Delete the lock
# only to upgrade deliberately, and never mix retrievals across locks.
#
# Idempotent: re-running reuses the checkout, venv and data.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATON_ENV="${PLATON_ENV:-$HOME/envs/platon-env}"
PLATON_SRC="${PLATON_SRC:-/project/def-ncowan/wasi/platon}"
PLATON_REF="${PLATON_REF:-v7.0}"
PLATON_COMMIT="202e9399b969c286d510f5254b1468185eec747b"   # v7.0
LOCK="$HERE/requirements-platon-fir.txt"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Deactivate $VIRTUAL_ENV first (a venv on PATH leaks into the build)." >&2
    exit 1
fi

module --force purge
module load StdEnv/2023 python/3.13.2
unset PYTHONPATH PYTHONHOME

echo "== source: $PLATON_SRC @ $PLATON_REF"
if [[ ! -d "$PLATON_SRC/.git" ]]; then
    git clone https://github.com/ideasrule/platon.git "$PLATON_SRC"
fi
git -C "$PLATON_SRC" fetch --tags --quiet
git -C "$PLATON_SRC" -c advice.detachedHead=false checkout --quiet "$PLATON_REF"
got="$(git -C "$PLATON_SRC" rev-parse HEAD)"
if [[ "$PLATON_REF" == "v7.0" && "$got" != "$PLATON_COMMIT" ]]; then
    echo "v7.0 tag points at $got, expected $PLATON_COMMIT. Tag moved upstream; stop and check." >&2
    exit 1
fi

echo "== venv: $PLATON_ENV"
if [[ ! -f "$PLATON_ENV/bin/activate" ]]; then
    virtualenv --no-download "$PLATON_ENV"
fi
# shellcheck disable=SC1091
source "$PLATON_ENV/bin/activate"
pip install --no-index --upgrade pip --quiet

if [[ -f "$LOCK" ]]; then
    echo "== installing from lock $LOCK"
    pip install -r "$LOCK"
    pip install --no-deps --no-build-isolation -e "$PLATON_SRC"
else
    echo "== no lock: resolving fresh (jax[cuda12] comes from PyPI; the"
    echo "   Alliance wheelhouse stops at jaxlib 0.4.28, too old for PLATON 7)"
    pip install "jax[cuda12]"
    pip install -e "$PLATON_SRC"
    pip freeze --exclude-editable > "$LOCK"
    echo "== wrote $LOCK -- commit it"
fi

echo "== opacity data (first run downloads ~11 GB and checks its md5)"
cd "$PLATON_SRC"   # never import platon from a directory that could shadow it
python -c "from platon._get_data import get_data_if_needed; get_data_if_needed()"

mkdir -p "$HOME/bin"
install -m 755 "$HERE/platon-python" "$HOME/bin/platon-python"

echo "== done. Verify on a GPU node:"
echo "   sbatch ~/aster/maiea/scripts/platon/verify_platon.sbatch"
