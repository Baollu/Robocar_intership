#!/bin/bash
# Automatic setup depending on the platform (PC x86_64 or Jetson Nano aarch64)

set -e

ARCH=$(uname -m)

echo "Detected platform: $ARCH"

if [ "$ARCH" = "aarch64" ]; then
    echo "=== Jetson Nano — installing dependencies ==="

    # Use the exact Python that will run the project
    PYTHON=$(which python3)
    PIP="$PYTHON -m pip"
    PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
    echo "Python binary : $PYTHON"
    echo "Python version: $($PYTHON --version)"
    echo "pip command   : $PIP"

    if ! $PYTHON -c "import torch" 2>/dev/null; then
        echo "Downloading PyTorch for ARM (Python $PY_VER)..."

        if [ "$PY_VER" = "36" ]; then
            # JetPack 4.6, Python 3.6 — installs torch 1.10.0
            TORCH_WHEEL="torch-1.8.0-cp36-cp36m-linux_aarch64.whl"
            TORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
            TORCHVISION_TAG="v0.11.1"
        elif [ "$PY_VER" = "38" ]; then
            # JetPack 4.6.x, Python 3.8 — installs torch 1.11.0
            TORCH_WHEEL="torch-1.11.0-cp38-cp38-linux_aarch64.whl"
            TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp38-cp38-linux_aarch64.whl"
            TORCHVISION_TAG="v0.12.0"
        else
            echo "ERROR: unsupported Python version ($PY_VER). Expected 36 or 38."
            exit 1
        fi

        wget "$TORCH_URL" -O "$TORCH_WHEEL"
        $PIP install "$TORCH_WHEEL"
        rm "$TORCH_WHEEL"
    else
        echo "PyTorch already installed, skipping."
    fi

    # torchvision has no ARM wheel on PyPI — build from source
    if ! $PYTHON -c "import torchvision" 2>/dev/null; then
        echo "Building torchvision $TORCHVISION_TAG from source (this takes ~10 min)..."
        sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev \
            libavcodec-dev libavformat-dev libswscale-dev
        git clone --depth 1 --branch "$TORCHVISION_TAG" \
            https://github.com/pytorch/vision torchvision_build
        cd torchvision_build
        $PYTHON setup.py install --user
        cd ..
        rm -rf torchvision_build
    else
        echo "torchvision already installed, skipping."
    fi

    $PIP install -r requirements_jetson.txt

else
    echo "=== PC (x86_64) — installing dependencies ==="
    pip install -r requirements_pc.txt
fi

echo ""
echo "Installation complete. Run the project with:"
echo "  python3 follow_the_line/mask_inference.py"
