#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${PDDLSTREAM_HOME:-${repo_root}/.paper_deps/pddlstream}"
revision="b38137e47fd4a4116a3e36bc4be691cbe5da6cb0"

mkdir -p "$(dirname "${target}")"
if [[ ! -d "${target}/.git" ]]; then
  git clone https://github.com/caelan/pddlstream.git "${target}"
fi
git -C "${target}" fetch origin "${revision}"
git -C "${target}" checkout --detach "${revision}"
# Only FastDownward is required. The research repository also declares a large
# PyBullet examples submodule that this MuJoCo port never imports.
git -C "${target}" submodule update --init downward

mkdir -p "${repo_root}/.paper_deps/ccache" "${repo_root}/.paper_deps/ccache-tmp"
CCACHE_DIR="${repo_root}/.paper_deps/ccache" \
CCACHE_TEMPDIR="${repo_root}/.paper_deps/ccache-tmp" \
  "${target}/downward/build.py"

printf 'PDDLStream ready at %s\n' "${target}"
printf 'revision: %s\n' "$(git -C "${target}" rev-parse HEAD)"
