#!/bin/bash

# ==============================================================================
# Script: build_onnxruntime.sh
# Purpose: Build onnxruntime (on CPU only).
# Dependencies: Core dependencies are build as sub-modules
# Usage: ./build_onnxruntime.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MAIN_DIR="$SCRIPT_DIR"
SOURCE_DIR="${MAIN_DIR}/onnxruntime"
BUILD_DIR="${SOURCE_DIR}/build"
INSTALL_DIR="${BUILD_DIR}/Linux/Release"
WHEEL_DIR="${INSTALL_DIR}/dist"

# CMake options (see ${SOURCE_DIR}/build.sh --help) (MIGRaphX not included,
#   not supported yet on my AMD gfx1103)
CMAKE_OPTIONS=(
	--config Release
	--build_wheel
	--parallel
	--build_shared_lib
	--enable_lto
	--use_cache
	--cmake_extra_defines "CMAKE_CXX_FLAGS=-march=native -O3 -flto -Wno-unused-parameter -Wno-ignored-attributes"
	--cmake_extra_defines "CMAKE_C_FLAGS=-march=native -O3 -flto -Wno-unused-parameter -Wno-ignored-attributes"
	--compile_no_warning_as_error
	--skip_tests
)

# -----------------------------------------------------------------------------
# Check Dependencies
# -----------------------------------------------------------------------------
# Assumes onnxruntime is cloned in a subdirectory:

if [ ! -d "$SOURCE_DIR" ]; then
	echo "Error: Source directory not found at $SOURCE_DIR"
	exit 1
fi

# -----------------------------------------------------------------------------
# Clean, Update and Build is done by the buid.sh script
# -----------------------------------------------------------------------------
cd "${SOURCE_DIR}"

echo "Running CMake configuration.."
#./build.sh   "${CMAKE_OPTIONS[@]}"

echo "Build completed successfully."

# -----------------------------------------------------------------------------
# Install under /usr/local
# -----------------------------------------------------------------------------
echo "Installing lib..."

if [ ! -d "$INSTALL_DIR" ]; then
	echo "Error: Install directory not found at $INSTALL_DIR"
	exit 1
fi

cd "${INSTALL_DIR}" && sudo make install

echo "Install wheel..."

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

echo "Installed successfully."
