#!/usr/bin/env bash
# Build Timeloop and Accelergy for the simulator backends, at the versions pinned by the
# Accelergy-Timeloop infrastructure repository, commit 6e6186f9fe8f9a1f3990f78f57c7224d22cb8cfa.
#
#   scripts/setup/install_timeloop.sh deps       apt packages, needs sudo
#   scripts/setup/install_timeloop.sh barvinok   barvinok with its bundled isl, installed under PREFIX
#   scripts/setup/install_timeloop.sh timeloop   timeloop-mapper, timeloop-model and timeloop-metrics
#   scripts/setup/install_timeloop.sh python     Accelergy, its estimation plug-ins and timeloopfe
#   scripts/setup/install_timeloop.sh check      reports whether the Timeloop backend can run
#   scripts/setup/install_timeloop.sh all        every stage in order
#
# PREFIX defaults to /usr/local, as in the official Docker image, so the binaries land on PATH.
# Sources are built under third_party/timeloop-build, which git ignores. PYTHON selects the environment
# that receives the Python packages. JOBS sets the build parallelism.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${TIMELOOP_BUILD:-$ROOT/third_party/timeloop-build}"
PREFIX="${PREFIX:-/usr/local}"
JOBS="${JOBS:-8}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SUDO="$([ "$(id -u)" = 0 ] && echo "" || echo sudo)"

BARVINOK_VER=0.41.8
# The infrastructure Dockerfile downloads from barvinok.sourceforge.io, which now refuses the request.
BARVINOK_URL="https://downloads.sourceforge.net/project/barvinok/barvinok-$BARVINOK_VER.tar.gz"
declare -A COMMITS=(
  [timeloop]=32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa
  [timeloopfe]=5603893c0ff75183b5ffd6839aba33774fc3b6fe
  [accelergy]=6911d15686ee7efdceba7d95605102df4472ae3a
  [accelergy-table-based-plug-ins]=bad19e941043045e130ea999852331f203d8c3fe
  [accelergy-aladdin-plug-in]=5e2e1263ddcc896ba3b8ce95954d76cdeebe03ab
  [accelergy-library-plug-in]=a0ec0b8ad30615544015a40079f7378bff1b7d76
  [accelergy-cacti-plug-in]=291018b12cc9cd467168973fc47528670e1480ce
  [accelergy-neurosim-plug-in]=bf874bc3c81734f28a44cfa0558512a25d2e90f4
  [accelergy-adc-plug-in]=b6e23e327419b073118001c6fece6ce1c95550f9
  [cacti]=1ffd8dfb10303d306ecd8d215320aea07651e878
)
declare -A REPOS=(
  [timeloop]=https://github.com/NVlabs/timeloop.git
  [timeloopfe]=https://github.com/Accelergy-Project/timeloopfe.git
  [accelergy]=https://github.com/Accelergy-Project/accelergy.git
  [accelergy-table-based-plug-ins]=https://github.com/Accelergy-Project/accelergy-table-based-plug-ins.git
  [accelergy-aladdin-plug-in]=https://github.com/Accelergy-Project/accelergy-aladdin-plug-in.git
  [accelergy-library-plug-in]=https://github.com/Accelergy-Project/accelergy-library-plug-in.git
  [accelergy-cacti-plug-in]=https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git
  [accelergy-neurosim-plug-in]=https://github.com/Accelergy-Project/accelergy-neurosim-plug-in.git
  [accelergy-adc-plug-in]=https://github.com/Accelergy-Project/accelergy-adc-plug-in.git
  [cacti]=https://github.com/HewlettPackard/cacti.git
)

fetch() {
  # Clone one repository into BUILD and check out its pinned commit.
  local name=$1 dir=$BUILD/$1
  [ -d "$dir/.git" ] || git clone --quiet "${REPOS[$name]}" "$dir"
  git -C "$dir" checkout --quiet "${COMMITS[$name]}"
  # Some nested submodules are recorded with SSH URLs, which fail on machines without GitHub SSH keys.
  git -C "$dir" -c url."https://github.com/".insteadOf=git@github.com: submodule update --init --recursive --quiet
}

patch_cacti() {
  # The same edit as cacti.patch in the infrastructure repository. It drops -gstabs+ and -m64, flags that
  # current GCC releases and non-x86 compilers reject.
  sed -i -e 's/ -gstabs+//' -e 's/^CXX = g++ -m64/CXX = g++/' -e 's/^CC  = gcc -m64/CC  = gcc/' "$1/cacti.mk"
}

stage_deps() {
  $SUDO apt-get update
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git wget scons autotools-dev autoconf automake libtool cmake pkg-config \
    libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libboost-thread-dev \
    libyaml-cpp-dev \
    libncurses-dev libgpm-dev libgmp-dev libntl-dev
}

stage_barvinok() {
  mkdir -p "$BUILD" && cd "$BUILD"
  [ -f "barvinok-$BARVINOK_VER.tar.gz" ] || {
    wget -q -O "barvinok-$BARVINOK_VER.tar.gz.part" "$BARVINOK_URL" &&
    mv "barvinok-$BARVINOK_VER.tar.gz.part" "barvinok-$BARVINOK_VER.tar.gz"; }
  [ -d "barvinok-$BARVINOK_VER" ] || tar -xzf "barvinok-$BARVINOK_VER.tar.gz"
  cd "barvinok-$BARVINOK_VER"
  ./configure --prefix="$PREFIX" --enable-shared-barvinok
  make -j"$JOBS"
  $SUDO make install
  $SUDO ldconfig
}

stage_timeloop() {
  mkdir -p "$BUILD" && fetch timeloop
  cd "$BUILD/timeloop"
  [ -e src/pat ] || ln -s ../pat-public/src/pat src/pat
  scons --accelergy -j"$JOBS"
  scons --static --accelergy -j"$JOBS"
  $SUDO install -m 0755 build/timeloop-mapper build/timeloop-model build/timeloop-metrics "$PREFIX/bin/"
}

stage_python() {
  mkdir -p "$BUILD"
  for name in accelergy accelergy-table-based-plug-ins accelergy-aladdin-plug-in accelergy-library-plug-in \
              cacti accelergy-cacti-plug-in accelergy-neurosim-plug-in accelergy-adc-plug-in timeloopfe; do
    fetch "$name"
  done
  patch_cacti "$BUILD/cacti"
  make -C "$BUILD/cacti" -j"$JOBS"
  "$PYTHON" -m pip install --quiet libconf numpy pydot ruamel.yaml psutil joblib
  "$PYTHON" -m pip install --quiet "$BUILD/accelergy"
  for plugin in accelergy-table-based-plug-ins accelergy-aladdin-plug-in accelergy-library-plug-in; do
    "$PYTHON" -m pip install --quiet "$BUILD/$plugin"
  done
  patch_cacti "$BUILD/accelergy-cacti-plug-in/cacti"
  make -C "$BUILD/accelergy-cacti-plug-in"
  "$PYTHON" -m pip install --quiet "$BUILD/accelergy-cacti-plug-in"
  # NeuroSim estimates the adders, multipliers and registers that no other plug-in covers in the Eyeriss,
  # FLAT and DXE specifications. The Makefile target named make assembles NeuroSim from its submodule and
  # the drop-in sources of the plug-in, and setup.py packages the resulting binary.
  make -C "$BUILD/accelergy-neurosim-plug-in" make
  "$PYTHON" -m pip install --quiet "$BUILD/accelergy-neurosim-plug-in"
  # The ADC plug-in estimates only analog-to-digital converters, which no substrate here uses. Installing it
  # keeps the plug-in set equal to the environment that produced the reference mapper outputs.
  "$PYTHON" -m pip install --quiet "$BUILD/accelergy-adc-plug-in"
  "$PYTHON" -m pip install --quiet "$BUILD/timeloopfe"
}

stage_check() {
  command -v timeloop-mapper
  PYTHONPATH="$ROOT/src" "$PYTHON" -c \
    "from llmforge.hw.timeloop.gemm import timeloop_available; print('timeloop_available', timeloop_available())"
}

case "${1:-all}" in
  deps) stage_deps ;;
  barvinok) stage_barvinok ;;
  timeloop) stage_timeloop ;;
  python) stage_python ;;
  check) stage_check ;;
  all) stage_deps; stage_barvinok; stage_timeloop; stage_python; stage_check ;;
  *) echo "usage: $0 [deps|barvinok|timeloop|python|check|all]"; exit 2 ;;
esac
