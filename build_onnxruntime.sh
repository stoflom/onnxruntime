#!/bin/bash

# ==============================================================================
# Script: build_onnxruntime.sh
# Purpose: Build onnxruntime (on CPU and MIGraphX) for Fedora 43/44/45
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
#	--cmake_extra_defines "CMAKE_CXX_FLAGS=-march=native -O3 -flto -Wno-unused-parameter -Wno-ignored-attributes -Wno-c++20-compat -Wno-calloc-transposed-args #-Wno-stringop-overflow"
#	--cmake_extra_defines "CMAKE_C_FLAGS=-march=native -O3 -flto -Wno-unused-parameter -Wno-ignored-attributes -Wno-calloc-transposed-args #-Wno-stringop-overflow"
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
# Apply local patches from ../patches on a dedicated dev branch
# -----------------------------------------------------------------------------
# The onnxruntime clone normally tracks upstream main, so local fixes live as
# patches in the outer workspace repo (see ../README.md). Before building we
# check whether they are already applied; if not, we check out (or create) a
# dev branch and `git am` whatever is missing. Applied patches are left alone
# on whatever branch the checkout already sits on.
PATCH_BRANCH="dev/local-patches"
PATCH_FILES=()
for p in "$SCRIPT_DIR"/patches/*.patch; do
	[ -e "$p" ] && PATCH_FILES+=("$p")
done

# Fills PENDING_PATCHES with patches that are not yet reverse-appliable
# (i.e. not yet applied) in the current checkout.
PENDING_PATCHES=()
collect_pending_patches() {
	PENDING_PATCHES=()
	local p
	for p in "${PATCH_FILES[@]}"; do
		git apply --reverse --check "$p" >/dev/null 2>&1 || PENDING_PATCHES+=("$p")
	done
}

if [ "${#PATCH_FILES[@]}" -eq 0 ]; then
	echo "No patches found in $SCRIPT_DIR/patches - building unpatched."
else
	collect_pending_patches
	if [ "${#PENDING_PATCHES[@]}" -gt 0 ]; then
		if git show-ref --verify --quiet "refs/heads/$PATCH_BRANCH"; then
			if [ "$(git branch --show-current)" != "$PATCH_BRANCH" ]; then
				echo "Switching to branch $PATCH_BRANCH.."
				git switch "$PATCH_BRANCH"
			fi
			collect_pending_patches
		else
			echo "Creating branch $PATCH_BRANCH.."
			git switch -c "$PATCH_BRANCH"
		fi
		if [ "${#PENDING_PATCHES[@]}" -gt 0 ]; then
			echo "Applying local patches:"
			for p in "${PENDING_PATCHES[@]}"; do echo "  - ${p##*/}"; done
			if ! git am --3way "${PENDING_PATCHES[@]}"; then
				git am --abort || true
				echo "Error: git am failed for one of the patches above."
				echo "The patches in $SCRIPT_DIR/patches may need rebasing against upstream main."
				exit 1
			fi
		fi
	fi
	echo "Local patches applied (branch: $(git branch --show-current))."
fi

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
