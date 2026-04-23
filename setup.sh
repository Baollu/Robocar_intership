#!/bin/bash
# Setup automatique selon la plateforme (PC x86_64 ou Jetson Nano aarch64)

set -e

ARCH=$(uname -m)

echo "Plateforme détectée : $ARCH"

if [ "$ARCH" = "aarch64" ]; then
    echo "=== Jetson Nano — installation des dépendances ==="

    # Torch ARM — wheel officiel NVIDIA (JetPack 4.6, Python 3.6)
    TORCH_WHEEL="torch-1.8.0-cp36-cp36m-linux_aarch64.whl"
    TORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"

    if ! python -c "import torch" 2>/dev/null; then
        echo "Téléchargement de PyTorch ARM..."
        wget -q "$TORCH_URL" -O "$TORCH_WHEEL"
        pip install "$TORCH_WHEEL"
        rm "$TORCH_WHEEL"
    else
        echo "PyTorch déjà installé, on passe."
    fi

    # torchvision compatible
    if ! python -c "import torchvision" 2>/dev/null; then
        pip install torchvision==0.9.0
    fi

    pip install -r requirements_jetson.txt

else
    echo "=== PC (x86_64) — installation des dépendances ==="
    pip install -r requirements_pc.txt
fi

echo ""
echo "Installation terminée. Lance le projet avec :"
echo "  python follow_the_line/mask_inference.py"
