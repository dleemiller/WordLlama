#!/bin/bash

set -e
set -x

# Set reproducible build variables
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
export PYTHONHASHSEED=0

if [[ "$(uname)" == "Darwin" ]]; then
    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        # For arm64 builds, use Homebrew
        if [[ "$(uname -m)" == "x86_64" ]]; then
            # Cross-compilation on x86_64 for arm64
            export PYTHON_CROSSENV=1
        fi
        export MACOSX_DEPLOYMENT_TARGET=14.0  # Match SciPy recommendations for arm64
        HOMEBREW_PREFIX="/opt/homebrew"

        echo "Installing libomp + llvm via Homebrew for arm64..."
        brew install libomp llvm

        export CC="$HOMEBREW_PREFIX/opt/llvm/bin/clang"
        export CXX="$HOMEBREW_PREFIX/opt/llvm/bin/clang++"
        export PATH="$HOMEBREW_PREFIX/opt/llvm/bin:$PATH"
        export CPPFLAGS="$CPPFLAGS -Xclang -fopenmp -I$HOMEBREW_PREFIX/opt/libomp/include"
        export CFLAGS="$CFLAGS -I$HOMEBREW_PREFIX/opt/libomp/include -ffp-contract=off"
        export CXXFLAGS="$CXXFLAGS -I$HOMEBREW_PREFIX/opt/libomp/include -ffp-contract=off"
        export LDFLAGS="$LDFLAGS -L$HOMEBREW_PREFIX/opt/libomp/lib -lomp"

    else
        # For x86_64 builds, adjust deployment target and install llvm-openmp via Conda
        export MACOSX_DEPLOYMENT_TARGET=13.0  # Matches Homebrew's libomp minimum
    
        # Install llvm-openmp via Conda
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/19.1.6/download/osx-64/llvm-openmp-19.1.6-ha54dae1_0.conda"
        echo "Installing llvm-openmp via Conda for x86_64..."
        sudo conda create -n build $OPENMP_URL
        PREFIX="$CONDA_HOME/envs/build"
    
        # Use system Clang and point it to Conda's OpenMP paths
        export CC="/usr/bin/clang"
        export CXX="/usr/bin/clang++"
    
        # Locate omp.h dynamically
        OMP_INCLUDE_DIR=$(find $PREFIX -type d -name "include" | head -n 1)
        if [[ -n "$OMP_INCLUDE_DIR" && -f "$OMP_INCLUDE_DIR/omp.h" ]]; then
            echo "Found omp.h in: $OMP_INCLUDE_DIR"
        else
            echo "Error: omp.h not found in Conda environment"
            echo "Contents of $PREFIX:"
            ls -R $PREFIX  # Debug: Show the structure of the Conda environment
            exit 1
        fi
    
        # Set flags
        export CPPFLAGS="-Xpreprocessor -fopenmp -I$OMP_INCLUDE_DIR"
        export CFLAGS="-I$OMP_INCLUDE_DIR -ffp-contract=off"
        export CXXFLAGS="-I$OMP_INCLUDE_DIR -ffp-contract=off"
        export LDFLAGS="-Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -lomp"
    fi


fi

if [[ "$CIBW_FREE_THREADED_SUPPORT" =~ [tT]rue ]]; then
    # Enable free-threaded builds for CPython if specified
    export CIBW_BUILD_FRONTEND='pip; args: --pre --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple" --only-binary :all:'
fi

# Ensure pip is updated before installing dependencies
python -m pip install --upgrade pip

# Install cibuildwheel and build wheels
python -m pip install --upgrade cibuildwheel
python -m cibuildwheel --output-dir wheelhouse

