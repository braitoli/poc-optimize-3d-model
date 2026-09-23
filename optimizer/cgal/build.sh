#!/usr/bin/env bash
# Builds optimizer/cgal/build/mesh_repair.
#
# Header roots are resolved in this order and nothing else is tried:
#   CGAL   $CGAL_DIR   -> installed (brew --prefix cgal)   -> third_party/CGAL-6.0.1
#   Boost  $BOOST_ROOT -> installed (brew --prefix boost)  -> third_party/boost_1_86_0
#   Eigen  $EIGEN_DIR  -> installed (brew --prefix eigen)  -> third_party/eigen-3.4.0
# (Eigen is a hard requirement of CGAL's Garland-Heckbert simplification policies.)
# Missing -> one-line error, non-zero exit.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
build="$here/build"

die() { echo "Fatal mesh_repair build error: $*" >&2; exit 1; }

# brew_prefix <formula> -> prints the prefix only when the formula is actually
# installed (`brew --prefix` happily prints paths for uninstalled kegs).
brew_prefix() {
  command -v brew >/dev/null 2>&1 || return 1
  local p
  p="$(brew --prefix "$1" 2>/dev/null)" || return 1
  [ -d "$p/include" ] || return 1
  echo "$p"
}

# resolve <probe-header> <env-value> <brew-formula> <vendored-include-dir>
# Prints the first include root that actually contains <probe-header>, or fails.
resolve() {
  local probe="$1" env_val="$2" formula="$3" vendored="$4" cand brew_root

  if [ -n "$env_val" ]; then
    for cand in "$env_val" "$env_val/include"; do
      [ -e "$cand/$probe" ] && { echo "$cand"; return 0; }
    done
    return 1
  fi

  if brew_root="$(brew_prefix "$formula")"; then
    for cand in "$brew_root/include" "$brew_root/include/eigen3"; do
      [ -e "$cand/$probe" ] && { echo "$cand"; return 0; }
    done
  fi

  [ -e "$vendored/$probe" ] && { echo "$vendored"; return 0; }
  return 1
}

cgal_inc="$(resolve CGAL/version.h "${CGAL_DIR:-}" cgal "$repo/third_party/CGAL-6.0.1/include")" \
  || die "CGAL headers not found - set \$CGAL_DIR (currently '${CGAL_DIR:-unset}'), install cgal, or vendor it at $repo/third_party/CGAL-6.0.1"

boost_inc="$(resolve boost/version.hpp "${BOOST_ROOT:-}" boost "$repo/third_party/boost_1_86_0")" \
  || die "Boost headers not found - set \$BOOST_ROOT (currently '${BOOST_ROOT:-unset}'), install boost, or vendor it at $repo/third_party/boost_1_86_0"

eigen_inc="$(resolve Eigen/Dense "${EIGEN_DIR:-}" eigen "$repo/third_party/eigen-3.4.0")" \
  || die "Eigen headers not found (the Garland-Heckbert policies need them) - set \$EIGEN_DIR (currently '${EIGEN_DIR:-unset}'), install eigen, or vendor it: curl -fsSL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar xz -C $repo/third_party"

echo "CGAL : $cgal_inc"
echo "Boost: $boost_inc"
echo "Eigen: $eigen_inc"

cmake -S "$here" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCGAL_INCLUDE_DIR="$cgal_inc" \
  -DBOOST_INCLUDE_DIR="$boost_inc" \
  -DEIGEN_INCLUDE_DIR="$eigen_inc" \
  || die "cmake configure failed"

cmake --build "$build" --parallel "$(sysctl -n hw.ncpu 2>/dev/null || echo 4)" \
  || die "build failed"

[ -x "$build/mesh_repair" ] || die "build finished but $build/mesh_repair is missing"
echo "Built $build/mesh_repair"
