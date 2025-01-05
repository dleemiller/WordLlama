#!/bin/bash

# Exit on error and echo each command (for debugging)
set -e
set -x

# ------------------------------------------------------------------------------
# 1. Reproducible Builds
# ------------------------------------------------------------------------------

# SOURCE_DATE_EPOCH is used to produce byte-for-byte reproducible builds
# based on the last commit timestamp.
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

# Ensure consistent hashing in Python
export PYTHONHASHSEED=0

# ------------------------------------------------------------------------------
# 2. macOS-Specific OpenMP Setup
# ------------------------------------------------------------------------------

if [[ "$(uname)" == "Darwin" ]]; then
    # Determine which macOS architecture we are building for
    # and set deployment target accordingly.
    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        # If the runner is x86_64 but we're building for arm64, enable cross-compilation
        if [[ "$(uname -m)" == "x86_64" ]]; then
            export PYTHON_CROSSENV=1
        fi
        # SciPy uses MACOSX_DEPLOYMENT_TARGET=12.0 on arm64; we match that to avoid issues
        export MACOSX_DEPLOYMENT_TARGET=12.0
    else
        # For x86_64 wheels, set an older deployment target (10.9)
        export MACOSX_DEPLOYMENT_TARGET=10.9
    fi

    # Install libomp and llvm from Homebrew (NOT Conda).
    # This ensures we use a Clang that fully supports OpenMP.
    echo "Installing libomp + llvm via Homebrew..."
    brew install libomp llvm

    # Force the use of Homebrew's LLVM Clang/Clang++ instead of system Clang
    export CC=/usr/local/opt/llvm/bin/clang
    export CXX=/usr/local/opt/llvm/bin/clang++
    
    # Prepend Homebrew LLVM to PATH so it’s picked up first
    export PATH="/usr/local/opt/llvm/bin:$PATH"

    # ------------------------------------------------------------------------------
    # 3. Compiler & Linker Flags for OpenMP
    # ------------------------------------------------------------------------------

    # CPPFLAGS: C/C++ preprocessor flags
    export CPPFLAGS="$CPPFLAGS -Xclang -fopenmp -I/usr/local/opt/libomp/include"
    # CFLAGS/CXXFLAGS: Compiler flags for C/C++
    export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
    # LDFLAGS: Linker flags
    export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
fi

# ------------------------------------------------------------------------------
# 4. Optional: CPython Free-Threaded Support
# ------------------------------------------------------------------------------

# If CIBW_FREE_THREADED_SUPPORT is set to "true", we install pre-release wheels
# from the scientific-python-nightly-wheels channel, which include free-threaded
# interpreter builds for CPython. This setting is rarely needed; remove if not used.
if [[ "$CIBW_FREE_THREADED_SUPPORT" =~ [tT]rue ]]; then
    export CIBW_BUILD_FRONTEND='pip; args: --pre --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" --only-binary :all:'
fi

# ------------------------------------------------------------------------------
# 5. Build the Wheels with cibuildwheel
# ------------------------------------------------------------------------------

python -m pip install --upgrade pip
python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse

