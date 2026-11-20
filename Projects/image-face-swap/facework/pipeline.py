"""Reusable, file- and HTTP-independent frame processing for images and video."""
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from .geometry import align_face, blend_face, quality_report
from .model import SAEHDPredictor


def choose_device(requested="auto"):
    available = ["cpu"]
    if torch.backends.mps.is_available():
        available.append("mps")
    if torch.cuda.is_available():
        available.append("cuda")
    if requested == "auto":
        return "cuda" if "cuda" in available else "mps" if "mps" in available else "cpu"
    if requested not in available:
        raise ValueError(f"设备 {requested} 当前不可用，可用设备：{', '.join(available)}")
    return requested


@dataclass(frozen=True)
class BlendOptions:
    feather: int = 8
    erode: int = 4
    color_match: bool = True

    def __post_init__(self):
        if not 0 <= self.feather <= 30 or not 0 <= self.erode <= 20:
            raise ValueError("羽化需为 0～30，边缘收缩需为 0～20")


@dataclass
class FrameResult:
    image: np.ndarray
    mask: np.ndarray
    elapsed_ms: int
    device: str
    warnings: list


class FacePipeline:
    def __init__(self, asset_dir, device="auto"):
        self.device = choose_device(device)
        self.asset_dir = Path(asset_dir)
        self._detector = None
        self._predictor = None
        self._model_key = None

    def detect(self, frame_bgr):
        if self._detector is None:
            from .detection import FaceDetector
            self._detector = FaceDetector(self.asset_dir, self.device)
        return self._detector.detect(frame_bgr)

    def invalidate_model(self):
        """Explicit re-import must reload weights even when the path is unchanged."""
        self._predictor = None
        self._model_key = None

    def process_frame(self, frame_bgr, model, *, face=None, face_index=0, options=None):
        """Process one frame; a video tracker may pass its selected face explicitly.

        No storage, web requests, frame extraction, audio or tracking lives here.
        The caller owns identity selection and temporal consistency across frames.
        """
        start = time.perf_counter()
        options = options or BlendOptions()
        if face is None:
            faces = self.detect(frame_bgr)
            if not 0 <= face_index < len(faces):
                raise ValueError("没有可处理的人脸，或选择的人脸不存在")
            face = faces[face_index]
        key = (model["directory"], model["prefix"])
        if key != self._model_key:
            self.invalidate_model()
            if self.device == "mps":
                torch.mps.empty_cache()
            self._predictor = SAEHDPredictor(model, self.device)
            self._model_key = key
        aligned, affine = align_face(frame_bgr, np.asarray(face["landmarks"], np.float32), model["resolution"], model["face_type"])
        swapped, mask = self._predictor.predict(aligned)
        merged, full_mask = blend_face(frame_bgr, swapped, mask, affine,
                                       feather=options.feather, erode=options.erode, color_match=options.color_match)
        warnings = list(quality_report(frame_bgr, face["box"], np.asarray(face["landmarks"]))["warnings"])
        if np.count_nonzero(full_mask > 20) < 100:
            warnings.append("有效融合区域很小，请检查模型训练状态或减小边缘收缩")
        return FrameResult(merged, full_mask, round((time.perf_counter() - start) * 1000), self.device, warnings)
