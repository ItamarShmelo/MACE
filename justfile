set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

cpp_sources := `git ls-files 'src/**/*.cpp' | tr '\n' ' '`
cpp_files   := `git ls-files 'src/**/*.cpp' 'src/**/*.hpp' | tr '\n' ' '`

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

lint-cpp: configure
    clang-format --dry-run --Werror {{cpp_files}}
    clang-tidy {{cpp_sources}} -p build

lint: lint-python lint-cpp

build:
    uv build

check: lint test build
