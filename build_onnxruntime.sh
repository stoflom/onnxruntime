#!/bin/bash

# ==============================================================================
# Script: build_onnxruntime.sh
# Purpose: Build onnxruntime (on CPU and MIGraphX) for Fedora 43/44.
# Dependencies: Core dependencies are build as sub-modules
# Usage: ./build_onnxruntime.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MAIN_DIR="$SCRIPT_DIR"
SOURCE_DIR="${MAIN_DIR}/onnxruntime"
BUILD_DIR="${SOURCE_DIR}/build"
BIN_DIR="${BUILD_DIR}/Linux/Release"
WHEEL_DIR="${BIN_DIR}/dist"

# CMake options (see ${SOURCE_DIR}/build.sh --help)
CMAKE_OPTIONS=(
	--config Release
	--build_wheel
	--parallel
	--build_shared_lib
	--enable_lto
	--use_cache
	--use_migraphx
	--migraphx_home "/usr/lib64"
	--cmake_extra_defines "CMAKE_CXX_FLAGS=-march=native -O3 -flto -Wno-unused-parameter -Wno-ignored-attributes -Wno-c++20-compat -Wno-calloc-transposed-args"
	--cmake_extra_defines "CMAKE_C_FLAGS=-march=native -O3 -flto -Wno-unused-parameter -Wno-ignored-attributes -Wno-calloc-transposed-args"
	--compile_no_warning_as_error
	--skip_tests
)

# -----------------------------------------------------------------------------
# Check Dependencies
# -----------------------------------------------------------------------------
# Assumes onnxruntime is cloned in a subdirectory; clone it if missing.
ONNXRUNTIME_REPO="https://github.com/Microsoft/onnxruntime.git"

if [ ! -d "$SOURCE_DIR" ]; then
	echo "Source directory not found at $SOURCE_DIR. Cloning $ONNXRUNTIME_REPO.."
	git clone "$ONNXRUNTIME_REPO" "$SOURCE_DIR" || {
		echo "Error: Failed to clone $ONNXRUNTIME_REPO"
		exit 1
	}
fi

cd "${SOURCE_DIR}"

# -----------------------------------------------------------------------------
# Ensure git submodules are initialized and pinned to the root repo versions
# -----------------------------------------------------------------------------
echo "Initializing/updating submodules.."
git submodule update --init --recursive --force

# Verify every submodule HEAD matches the commit recorded by the root repo.
# Status line prefixes: ' ' = OK, '-' = uninitialized, '+' = different commit,
# 'U' = merge conflict.
SUBMODULE_STATUS="$(git submodule status)"
if printf '%s\n' "$SUBMODULE_STATUS" | grep -qE "^[[:space:]]*[-+U]"; then
	echo "Error: Submodule state does not match the root repo versions:"
	printf '%s\n' "$SUBMODULE_STATUS"
	exit 1
fi
echo "Submodules in sync with root repo."

# -----------------------------------------------------------------------------
# Clean, Update and Build is done by the build.sh script, but do force clean
# -----------------------------------------------------------------------------
echo "Cleaning op build subdirectory."
rm -rf "${BUILD_DIR}"

echo "Running CMake configuration.."
./build.sh "${CMAKE_OPTIONS[@]}"

echo "Build completed successfully."

# -----------------------------------------------------------------------------
# Install under /usr/local
# -----------------------------------------------------------------------------
echo "Installing lib..."

if [ ! -d "$BIN_DIR" ]; then
	echo "Error: Install directory not found at $BIN_DIR"
	exit 1
fi

# Wait for the user to wake up
echo "Install requires root priviledges."
read -p "Press Enter to continue..."

cd "${BIN_DIR}" && sudo make install

echo "Install wheel (locally)..."

if [ ! -d "$WHEEL_DIR" ]; then
	echo "Error: Wheel directory not found at $WHEEL_DIR"
	exit 1
fi

WHEEL_FILE=("$WHEEL_DIR"/onnxruntime*.whl)
WHEEL_FILE="${WHEEL_FILE[0]}"

if [ ! -s "$WHEEL_FILE" ]; then
	echo "Error: Wheel file not found at $WHEEL_FILE"
	exit 1
fi

pip install "$WHEEL_FILE" --force-reinstall

echo "Execution providers:"
cd "${SCRIPT_DIR}" && python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"

echo "Installed successfully."
