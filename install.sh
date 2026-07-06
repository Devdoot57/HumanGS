#!/bin/bash

main() {
    echo "[INFO] Installing standard dependencies..."
    if ! pip install -r requirements.txt; then
        echo "[ERROR] Failed to install standard dependencies."
        return 1
    fi

    echo "[INFO] Building PyTorch3D from source (Requires CUDA 13.0)..."
    if ! NVCC_FLAGS="-static-global-template-stub=false" FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation; then
        echo "[ERROR] PyTorch3D build failed."
        echo "[INFO] Please ensure you have CUDA 13.0 installed and nvcc is in your PATH."
        return 1
    fi

    echo "[INFO] Installing Differential Gaussian Rasterization..."
    if [ ! -d "diff-gaussian-rasterization" ]; then
        echo "[INFO] diff-gaussian-rasterization directory not found. Cloning repository..."
        if ! git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git; then
            echo "[ERROR] Failed to clone diff-gaussian-rasterization."
            return 1
        fi
    fi

    cd diff-gaussian-rasterization || return 1    
    git submodule update --init --recursive

    echo "[INFO] Adding missing C++ headers for modern compilers..."
    # Automatically injects <cstdint> at the top of the header file to prevent uint32_t errors in C++20
    sed -i '1i #include <cstdint>' cuda_rasterizer/rasterizer_impl.h

    if ! pip install . --no-build-isolation; then
        echo "[ERROR] Failed to install diff-gaussian-rasterization."
        cd ..
        return 1
    fi

    cd ..

    echo "[INFO] Environment setup complete."
}

main