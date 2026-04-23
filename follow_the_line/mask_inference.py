import sys
import os
import json
import platform

import cv2
import torch
import numpy as np

try:
    import depthai as dai
    _DEPTHAI_AVAILABLE = True
except ImportError:
    _DEPTHAI_AVAILABLE = False

MASK_GEN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mask_generator'))
sys.path.insert(0, MASK_GEN_DIR)

from model import SegNet  # noqa: E402

# ── Résolution attendue par le modèle ─────────────────────────────────────
IMAGE_WIDTH  = 256
IMAGE_HEIGHT = 128

# ── Détection automatique de la plateforme ────────────────────────────────
# Caméra par défaut selon la plateforme :
#   aarch64 (Jetson Nano) → "oak" si DepthAI disponible, sinon "csi"
#   x86_64  (PC)          → "oak" si DepthAI disponible, sinon "usb"
# Surcharge : CAMERA_TYPE=usb python mask_inference.py
_IS_JETSON = platform.machine() == "aarch64"

if _DEPTHAI_AVAILABLE:
    _default_camera = "oak"
elif _IS_JETSON:
    _default_camera = "csi"
else:
    _default_camera = "usb"

CAMERA_TYPE = os.environ.get("CAMERA_TYPE", _default_camera)

print(f"Plateforme : {'Jetson Nano (aarch64)' if _IS_JETSON else 'PC (x86_64)'} — caméra : {CAMERA_TYPE}")

# ── GStreamer pipeline pour la caméra CSI de la Jetson Nano ───────────────
CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)


def _find_latest_checkpoint(weights_dir: str) -> str:
    checkpoints = sorted(f for f in os.listdir(weights_dir) if f.endswith(".pth.tar"))
    if not checkpoints:
        raise FileNotFoundError(f"Aucun checkpoint trouvé dans {weights_dir}. Lance train.py d'abord.")
    return os.path.join(weights_dir, checkpoints[-1])


def load_model() -> SegNet:
    json_path = os.path.join(MASK_GEN_DIR, "model.json")
    with open(json_path) as f:
        cfg = json.load(f)

    model = SegNet(in_chn=cfg["in_chn"], out_chn=cfg["out_chn"], BN_momentum=cfg["bn_momentum"])

    weights_dir = os.path.join(MASK_GEN_DIR, "weights")
    ckpt_path   = _find_latest_checkpoint(weights_dir)
    print(f"Chargement du checkpoint : {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if torch.cuda.is_available():
        try:
            model = model.cuda()
            dummy = torch.zeros(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).cuda()
            with torch.no_grad():
                model(dummy)
            torch.cuda.synchronize()  # force l'exécution GPU pour détecter les erreurs maintenant
            print("GPU détecté — inférence sur CUDA")
        except RuntimeError:
            print("GPU incompatible avec cette version de PyTorch — fallback CPU")
            model = model.cpu()
    else:
        print("CUDA non disponible — inférence sur CPU")

    return model


def open_camera(camera_type: str) -> cv2.VideoCapture:
    """Ouvre une caméra CSI ou USB (pas OAK-D — géré séparément)."""
    if camera_type == "csi":
        cap = cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la caméra '{camera_type}'.\n"
            "  CSI : vérifie que nvarguscamerasrc est disponible (JetPack installé).\n"
            "  USB : vérifie que /dev/video0 existe.\n"
            "  Surcharge possible : CAMERA_TYPE=usb python mask_inference.py"
        )
    return cap


def _build_oak_pipeline(width: int = 640, height: int = 480) -> "dai.Pipeline":
    pipeline     = dai.Pipeline()
    cam          = pipeline.create(dai.node.ColorCamera)
    xout         = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.setPreviewSize(width, height)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.preview.link(xout.input)
    return pipeline


def preprocess(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor  = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def infer_mask(model: SegNet, tensor: torch.Tensor) -> np.ndarray:
    """Retourne un masque binaire numpy uint8 (0 = non-route, 255 = route)."""
    with torch.no_grad():
        output = model(tensor)
    mask_idx = torch.argmax(output, dim=1)[0]
    return (mask_idx.cpu().numpy() * 255).astype(np.uint8)


def _run_oak(model: torch.nn.Module, device: torch.device, display: bool) -> None:
    pipeline = _build_oak_pipeline()
    print("Appuie sur 'q' pour quitter.")
    with dai.Device(pipeline) as dev:
        q = dev.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        while True:
            frame = q.get().getCvFrame()          # BGR numpy array
            tensor = preprocess(frame, device)
            mask   = infer_mask(model, tensor)

            if display:
                preview = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
                cv2.imshow("Camera", preview)
                cv2.imshow("Mask",   mask)

            if cv2.waitKey(1) == ord("q"):
                break
    cv2.destroyAllWindows()


def _run_cv2(model: torch.nn.Module, device: torch.device, display: bool) -> None:
    cap = open_camera(CAMERA_TYPE)
    print("Appuie sur 'q' pour quitter.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : impossible de lire la caméra.")
            break

        tensor = preprocess(frame, device)
        mask   = infer_mask(model, tensor)

        if display:
            preview = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            cv2.imshow("Camera", preview)
            cv2.imshow("Mask",   mask)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_live(display: bool = True) -> None:
    model  = load_model()
    device = next(model.parameters()).device

    if CAMERA_TYPE == "oak":
        if not _DEPTHAI_AVAILABLE:
            raise RuntimeError("DepthAI non installé. Lance : pip install depthai")
        _run_oak(model, device, display)
    else:
        _run_cv2(model, device, display)


if __name__ == "__main__":
    run_live(display=True)
