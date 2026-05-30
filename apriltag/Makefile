# aidall_apriltag — build / install / test convenience targets.
#
#   make install    # build and install the C++ shared library to PREFIX
#   make test       # run all tests (C++ smoke tests + Python suite)
#   make format     # format C++ (clang-format) and Python (ruff)
#   make clean      # remove build artifacts
#
# Variables: PREFIX (install prefix, default /usr/local), PYTHON, BUILD (dir).

PYTHON ?= python3
BUILD  ?= build
DIST   := dist
PREFIX ?= /usr/local

.PHONY: help install test format clean

help:
	@sed -n 's/^#   //p' $(MAKEFILE_LIST)

# Build and install the self-contained C++ shared library + headers + CMake
# package config to PREFIX. Use `sudo make install` for system prefixes, or
# `make install PREFIX=$$HOME/.local` for a user prefix.
install:
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(PREFIX)
	cmake --build $(BUILD) --parallel
	cmake --install $(BUILD)

# Run all tests: C++ tests, then the Python suite.
test:
	cmake -S . -B $(BUILD) -DAIDALL_APRILTAG_BUILD_TESTS=ON
	cmake --build $(BUILD) --target test_detector
	./$(BUILD)/test_detector
	$(PYTHON) -m pytest python/tests

# Format C++ (clang-format, .clang-format) and Python (ruff: imports + style).
format:
	@cxx=$$(find cpp python/bindings \( -name '*.cpp' -o -name '*.hpp' \) 2>/dev/null); \
	if command -v clang-format >/dev/null 2>&1; then \
	  clang-format -i $$cxx && echo "clang-format: formatted C++"; \
	else echo "clang-format not found; skipping C++ (e.g. brew install clang-format)"; fi
	@if $(PYTHON) -m ruff --version >/dev/null 2>&1; then \
	  $(PYTHON) -m ruff check --select I --fix python && \
	  $(PYTHON) -m ruff format python; \
	else echo "ruff not found; run: $(PYTHON) -m pip install ruff"; fi

clean:
	rm -rf $(DIST) $(BUILD) *.egg-info python/aidall_apriltag/*.so python/aidall_apriltag/_core*.so
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
