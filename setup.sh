#!/bin/bash
# Automatic setup depending on the platform (PC x86_64 or Jetson Nano aarch64)

set -e

ARCH=$(uname -m)

echo "Detected platform: $ARCH"

if [ "$ARCH" = "aarch64" ]; then
    echo "=== Jetson Nano — installing dependencies ==="

    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
    echo "Python version: $(python3 --version)"

    if ! python3 -c "import torch" 2>/dev/null; then
        echo "Downloading PyTorch for ARM (Python $PY_VER)..."

        if [ "$PY_VER" = "36" ]; then
            # JetPack 4.6, Python 3.6
            TORCH_WHEEL="torch-1.8.0-cp36-cp36m-linux_aarch64.whl"
            TORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
            TORCHVISION_VER="0.9.0"
        elif [ "$PY_VER" = "38" ]; then
            # JetPack 4.6.x, Python 3.8
            TORCH_WHEEL="torch-1.11.0-cp38-cp38-linux_aarch64.whl"
            TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp38-cp38-linux_aarch64.whl"
            TORCHVISION_VER="0.12.0"
        else
            echo "ERROR: unsupported Python version ($PY_VER). Expected 36 or 38."
            exit 1
        fi

        wget -q "$TORCH_URL" -O "$TORCH_WHEEL"
        pip3 install "$TORCH_WHEEL"
        rm "$TORCH_WHEEL"
    else
        echo "PyTorch already installed, skipping."
    fi

    if ! python3 -c "import torchvision" 2>/dev/null; then
        pip3 install "torchvision==$TORCHVISION_VER"
    fi

    pip3 install -r requirements_jetson.txt

else
    echo "=== PC (x86_64) — installing dependencies ==="
    pip install -r requirements_pc.txt
fi

echo ""
echo "Installation complete. Run the project with:"
echo "  python3 follow_the_line/mask_inference.py"
