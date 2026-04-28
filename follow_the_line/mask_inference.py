# -*- coding: utf-8 -*-
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

from model import SegNet

IMAGE_WIDTH  = 256
IMAGE_HEIGHT = 128
_IS_JETSON = platform.machine() == "aarch64"

if _DEPTHAI_AVAILABLE:
    _default_camera = "oak"
elif _IS_JETSON:
    _default_camera = "csi"
else:
    _default_camera = "usb"

CAMERA_TYPE = os.environ.get("CAMERA_TYPE", _default_camera)

print(f"Platform: {'Jetson Nano (aarch64)' if _IS_JETSON else 'PC (x86_64)'} - camera: {CAMERA_TYPE}")

# ── GStreamer pipeline for the Jetson Nano CSI camera ────────────────────
CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)


def _find_latest_checkpoint(weights_dir: str) -> str:
    checkpoints = sorted(f for f in os.listdir(weights_dir) if f.endswith(".pth.tar"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {weights_dir}. Run train.py first.")
    return os.path.join(weights_dir, checkpoints[-1])


def load_model() -> SegNet:
    json_path = os.path.join(MASK_GEN_DIR, "model.json")
    with open(json_path) as f:
        cfg = json.load(f)

    model = SegNet(in_chn=cfg["in_chn"], out_chn=cfg["out_chn"], BN_momentum=cfg["bn_momentum"])

    weights_dir = os.path.join(MASK_GEN_DIR, "weights")
    ckpt_path   = _find_latest_checkpoint(weights_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if torch.cuda.is_available():
        try:
            model = model.cuda()
            dummy = torch.zeros(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).cuda()
            with torch.no_grad():
                model(dummy)
            torch.cuda.synchronize()  # force GPU execution to catch errors early
            print("GPU detected — running inference on CUDA")
        except RuntimeError:
            print("GPU incompatible with this PyTorch version — falling back to CPU")
            model = model.cpu()
    else:
        print("CUDA not available — running inference on CPU")

    return model


def open_camera(camera_type: str) -> cv2.VideoCapture:
    """Open a CSI or USB camera (OAK-D is handled separately)."""
    if camera_type == "csi":
        cap = cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera '{camera_type}'.\n"
            "  CSI: check that nvarguscamerasrc is available (JetPack installed).\n"
            "  USB: check that /dev/video0 exists.\n"
            "  Override: CAMERA_TYPE=usb python mask_inference.py"
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
    """Return a binary numpy uint8 mask (0 = non-road, 255 = road)."""
    with torch.no_grad():
        output = model(tensor)
    mask_idx = torch.argmax(output, dim=1)[0]
    return (mask_idx.cpu().numpy() * 255).astype(np.uint8)


def apply_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a green overlay on road pixels (mask=255) over the resized frame."""
    preview = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    overlay = preview.copy()
    overlay[mask == 255] = (0, 200, 0)
    return cv2.addWeighted(overlay, alpha, preview, 1 - alpha, 0)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _save_frame(frame: np.ndarray, mask: np.ndarray, index: int) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"frame_{index:06d}.jpg")
    cv2.imwrite(path, apply_overlay(frame, mask))


def _run_oak(model: torch.nn.Module, device: torch.device, save: bool) -> None:
    pipeline = _build_oak_pipeline()
    print("Press Ctrl+C to quit.")
    i = 0
    with dai.Device(pipeline) as dev:
        q = dev.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        while True:
            frame  = q.get().getCvFrame()
            tensor = preprocess(frame, device)
            mask   = infer_mask(model, tensor)

            if save:
                _save_frame(frame, mask, i)
                if i % 30 == 0:
                    print(f"Saved frame {i} -> {OUTPUT_DIR}")
            i += 1


def _run_cv2(model: torch.nn.Module, device: torch.device, save: bool) -> None:
    cap = open_camera(CAMERA_TYPE)
    print("Press Ctrl+C to quit.")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: cannot read from camera.")
            break

        tensor = preprocess(frame, device)
        mask   = infer_mask(model, tensor)

        if save:
            _save_frame(frame, mask, i)
            if i % 30 == 0:
                print(f"Saved frame {i} -> {OUTPUT_DIR}")
        i += 1

    cap.release()


def run_live(save: bool = True) -> None:
    model  = load_model()
    device = next(model.parameters()).device

    if CAMERA_TYPE == "oak":
        if not _DEPTHAI_AVAILABLE:
            raise RuntimeError("DepthAI not installed. Run: pip install depthai")
        _run_oak(model, device, save)
    else:
        _run_cv2(model, device, save)


if __name__ == "__main__":
    run_live(save=True)
