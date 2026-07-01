set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

cpp_sources     := `git ls-files 'src/**/*.cpp' | tr '\n' ' '`
cpp_files       := `git ls-files 'src/**/*.cpp' 'src/**/*.hpp' | tr '\n' ' '`
clang_tidy      := `which clang-tidy 2>/dev/null || true`
clang_cxx       := `which clang++ 2>/dev/null || true`
gcc_install_dir := `g++ -print-search-dirs 2>/dev/null \
  | grep '^install:' | sed 's/install: //' | sed 's:/$::' \
  || true`

setup:
    uv sync
    uv pip install -e .

test:
    uv run pytest

format-python:
    uv run ruff format .
    uv run ruff check . --fix

format-cpp:
    clang-format -i {{cpp_files}}

format: format-python format-cpp

lint-python:
    uv run ruff format --check .
    uv run ruff check .

configure:
    cmake --preset dev

build-cpp: configure
    cmake --build --preset dev

_check-tidy-toolchain:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{clang_tidy}}" ] || [ ! -x "{{clang_tidy}}" ]; then
      echo "error: clang-tidy not found on PATH"
      echo "       Override with: just clang_tidy=/path/to/clang-tidy lint-cpp"
      exit 1
    fi
    CT_VER=$({{clang_tidy}} --version | grep -oP 'version \K[0-9]+')
    if [ "$CT_VER" -lt 18 ]; then
      echo "error: clang-tidy >= 18 required, found version $CT_VER (at {{clang_tidy}})"
      echo "       Override with: just clang_tidy=/path/to/clang-tidy lint-cpp"
      exit 1
    fi
    if [ -z "{{clang_cxx}}" ] || [ ! -x "{{clang_cxx}}" ]; then
      echo "error: clang++ not found on PATH"
      echo "       Override with: just clang_cxx=/path/to/clang++ lint-cpp"
      exit 1
    fi
    CXX_VER=$({{clang_cxx}} --version | grep -oP 'version \K[0-9]+')
    if [ "$CXX_VER" -lt 18 ]; then
      echo "error: clang++ >= 18 required, found version $CXX_VER (at {{clang_cxx}})"
      echo "       Override with: just clang_cxx=/path/to/clang++ lint-cpp"
      exit 1
    fi
    if [ -z "{{gcc_install_dir}}" ] || [ ! -d "{{gcc_install_dir}}" ]; then
      echo "error: GCC install directory not found (g++ not on PATH or g++ -print-search-dirs failed)"
      echo "       Override with: just gcc_install_dir=/path/to/gcc/lib/gcc/triplet/version lint-cpp"
      exit 1
    fi
    GCC_VER=$(basename "{{gcc_install_dir}}" | grep -oP '^\K[0-9]+')
    if [ "$GCC_VER" -lt 15 ]; then
      echo "error: GCC >= 15 required, found version $GCC_VER (at {{gcc_install_dir}})"
      echo "       Override with: just gcc_install_dir=/path/to/gcc/lib/gcc/triplet/version lint-cpp"
      exit 1
    fi
    echo "Toolchain OK: clang-tidy $CT_VER, clang++ $CXX_VER, GCC $GCC_VER"

lint-cpp: _check-tidy-toolchain
    cmake -B build-tidy --fresh \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_CXX_COMPILER={{clang_cxx}} \
      "-DCMAKE_CXX_FLAGS=--gcc-install-dir={{gcc_install_dir}}" \
      -DCOMPTON_ENABLE_OMP=OFF
    clang-format --dry-run --Werror {{cpp_files}}
    {{clang_tidy}} {{cpp_sources}} -p build-tidy

lint: lint-python lint-cpp

build:
    uv build

check: lint test build
