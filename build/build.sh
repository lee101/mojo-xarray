#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"
# MOJO_BUILD_JOBS caps LLVM's worker threads (0 = compiler default). Needed on
# hosts with a tight pids cgroup, where one thread per core exhausts the budget
# and posix_spawn fails with EAGAIN.
mojo build --emit shared-lib -j "${MOJO_BUILD_JOBS:-0}" -I "$repo_dir/src" \
    "$repo_dir/src/capi.mojo" \
    -o "$repo_dir/dist/libmojo-xarray.so"
# `python/mojo_xarray/_lib.py` loads `build/capi.so` and rebuilds it when it is
# older than `src/`, so keep that copy fresh instead of paying a second build.
cp "$repo_dir/dist/libmojo-xarray.so" "$repo_dir/build/capi.so"
