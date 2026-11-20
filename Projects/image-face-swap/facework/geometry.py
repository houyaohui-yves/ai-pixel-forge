"""Face alignment, dataset diagnostics and compositing without a UI dependency.

Derived from DeepFaceLab (GPLv3), specifically facelib/LandmarksProcessor.py
and core/mathlib/umeyama.py. The 33-point template, similarity fit, padding,
and whole-face offset preserve DFL's alignment conventions. Validation,
quality heuristics and premultiplied-alpha compositing are adapted here.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


LANDMARK_TEMPLATE = np.array([
    [0.000213256, 0.106454], [0.0752622, 0.038915],
    [0.18113, 0.0187482], [0.29077, 0.0344891],
    [0.393397, 0.0773906], [0.586856, 0.0773906],
    [0.689483, 0.0344891], [0.799124, 0.0187482],
    [0.904991, 0.038915], [0.98004, 0.106454],
    [0.490127, 0.203352], [0.490127, 0.307009],
    [0.490127, 0.409805], [0.490127, 0.515625],
    [0.36688, 0.587326], [0.426036, 0.609345],
    [0.490127, 0.628106], [0.554217, 0.609345],
    [0.613373, 0.587326], [0.121737, 0.216423],
    [0.187122, 0.178758], [0.265825, 0.179852],
    [0.334606, 0.231733], [0.260918, 0.245099],
    [0.182743, 0.244077], [0.645647, 0.231733],
    [0.714428, 0.179852], [0.793132, 0.178758],
    [0.858516, 0.216423], [0.79751, 0.244077],
    [0.719335, 0.245099], [0.254149, 0.780233],
    [0.726104, 0.780233],
], dtype=np.float32)

_FACE_TYPES = {
    "h": ("h", 0.0), "half_face": ("h", 0.0),
    "mf": ("mf", 0.0675), "midfull_face": ("mf", 0.0675),
    "f": ("f", 0.2109375), "full_face": ("f", 0.2109375),
    "wf": ("wf", 0.40), "whole_face": ("wf", 0.40),
}


def _image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("图片必须是 uint8 BGR 三通道数组。")
    if min(image.shape[:2]) < 1:
        raise ValueError("图片尺寸不能为空。")
    return image


def _landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError("需要 68 个有效的人脸关键点。")
    selected = np.concatenate([points[17:49], points[54:55]])
    if np.linalg.matrix_rank(selected - selected.mean(axis=0)) < 2:
        raise ValueError("人脸关键点退化，无法可靠对齐。")
    return points


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DFL/Umeyama similarity transform, including proper rotation handling."""
    num, dim = src.shape
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_delta, dst_delta = src - src_mean, dst - dst_mean
    covariance = dst_delta.T @ src_delta / num
    signs = np.ones(dim, dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        signs[-1] = -1
    transform = np.eye(dim + 1, dtype=np.float64)
    u, singular, v = np.linalg.svd(covariance)
    rank = np.linalg.matrix_rank(covariance)
    if rank == 0:
        raise ValueError("关键点无法构成有效变换。")
    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(v) > 0:
            rotation = u @ v
        else:
            previous = signs[-1]
            signs[-1] = -1
            rotation = u @ np.diag(signs) @ v
            signs[-1] = previous
    else:
        rotation = u @ np.diag(signs) @ v
    scale = np.dot(singular, signs) / src_delta.var(axis=0).sum()
    transform[:dim, :dim] = rotation * scale
    transform[:dim, dim] = dst_mean - scale * rotation @ src_mean
    return transform


def alignment_matrix(landmarks: np.ndarray, size: int, face_type: str = "f") -> np.ndarray:
    """Return the original-image → aligned-face affine matrix used by DFL."""
    points = _landmarks(landmarks)
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or not 16 <= size <= 4096:
        raise ValueError("对齐尺寸必须是 16～4096 之间的整数。")
    if not isinstance(face_type, str) or face_type.lower() not in _FACE_TYPES:
        raise ValueError("当前仅支持 h、mf、f、wf 人脸类型。")
    kind, padding = _FACE_TYPES[face_type.lower()]
    selected = np.concatenate([points[17:49], points[54:55]])
    normalized = _umeyama(selected, LANDMARK_TEMPLATE)[:2]
    corners = cv2.transform(
        np.array([[(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]], dtype=np.float32),
        cv2.invertAffineTransform(normalized),
    )[0]
    center = corners[4].copy()
    diagonal_a = (corners[2] - corners[0]).astype(np.float32)
    diagonal_b = (corners[1] - corners[3]).astype(np.float32)
    length = np.linalg.norm(diagonal_a)
    if length < 1e-6 or not np.isfinite(length):
        raise ValueError("人脸区域过小，无法对齐。")
    diagonal_a /= length
    diagonal_b /= np.linalg.norm(diagonal_b)
    extent = length * (padding * math.sqrt(2.0) + 0.5)
    if kind == "wf":
        center += (corners[0] - corners[3]) * 0.07
    source = np.asarray([
        center - diagonal_a * extent,
        center + diagonal_b * extent,
        center + diagonal_a * extent,
    ], dtype=np.float32)
    target = np.float32([(0, 0), (size, 0), (size, size)])
    affine = cv2.getAffineTransform(source, target)
    if not np.isfinite(affine).all():
        raise ValueError("对齐变换无效。")
    return affine


def align_face(image_bgr, landmarks, size: int, face_type: str = "f"):
    """Produce a DFL-aligned crop and its reusable source-to-crop transform."""
    image = _image(image_bgr)
    affine = alignment_matrix(landmarks, size, face_type)
    aligned = cv2.warpAffine(
        image, affine, (size, size), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return aligned, affine


def quality_report(image_bgr, box, landmarks) -> dict:
    """Heuristics for manual dataset curation, never an identity/beauty score.

    brightness is a mean gray level in [0,255]; sharpness is Laplacian
    variance on the detected crop. yaw is a normalized nose/eye offset in
    [-1,1], NOT an angle in degrees and NOT a calibrated pose estimate.
    """
    image = _image(image_bgr)
    points = _landmarks(landmarks)
    bounds = np.asarray(box, dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError("人脸框必须包含四个有效坐标。")
    height, width = image.shape[:2]
    left, top = np.floor(bounds[:2]).astype(int)
    right, bottom = np.ceil(bounds[2:]).astype(int)
    if right <= left or bottom <= top:
        raise ValueError("人脸框尺寸无效。")
    crop = image[max(0, top):min(height, bottom), max(0, left):min(width, right)]
    if crop.size == 0:
        raise ValueError("人脸框位于图片范围之外。")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    eye_left, eye_right = points[36:42].mean(axis=0), points[42:48].mean(axis=0)
    eye_vector = eye_right - eye_left
    eye_distance = float(np.linalg.norm(eye_vector))
    yaw = 0.0 if eye_distance < 1e-6 else float(np.clip(
        2 * np.dot(points[30] - (eye_left + eye_right) / 2, eye_vector) / eye_distance**2,
        -1, 1,
    ))
    warnings = []
    if min(crop.shape[:2]) < 96:
        warnings.append("人脸区域较小，建议补充更清晰的大图。")
    if sharpness < 60:
        warnings.append("可能存在模糊，请人工检查眼睛和轮廓。")
    if brightness < 45:
        warnings.append("面部偏暗，建议补充光线更均匀的素材。")
    elif brightness > 215:
        warnings.append("面部偏亮，请检查高光是否丢失细节。")
    if abs(yaw) > 0.45:
        warnings.append("左右偏转较大；可用于补充角度，需检查关键点。")
    if left < 0 or top < 0 or right > width or bottom > height:
        warnings.append("人脸贴近图片边界，请检查是否裁掉了轮廓。")
    return {
        "sharpness": round(sharpness, 2), "brightness": round(brightness, 2),
        "yaw": round(yaw, 3), "warnings": warnings,
    }


def _color_match(source, target, mask):
    valid = mask > 0.2
    if np.count_nonzero(valid) < 16:
        return source
    src_lab = cv2.cvtColor(source.astype(np.float32), cv2.COLOR_BGR2LAB)
    dst_lab = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2LAB)
    weight = mask[valid].astype(np.float64)
    weight /= weight.sum()
    src_values, dst_values = src_lab[valid], dst_lab[valid]
    src_mean = np.sum(src_values * weight[:, None], axis=0)
    dst_mean = np.sum(dst_values * weight[:, None], axis=0)
    src_std = np.sqrt(np.sum((src_values - src_mean)**2 * weight[:, None], axis=0))
    dst_std = np.sqrt(np.sum((dst_values - dst_mean)**2 * weight[:, None], axis=0))
    ratio = np.clip(dst_std / np.maximum(src_std, 1e-4), 0.5, 2.0)
    matched = (src_lab - src_mean) * ratio + dst_mean
    matched[:, :, 0] = np.clip(matched[:, :, 0], 0, 100)
    matched[:, :, 1:] = np.clip(matched[:, :, 1:], -127, 127)
    return np.clip(cv2.cvtColor(matched.astype(np.float32), cv2.COLOR_LAB2BGR), 0, 1)


def blend_face(original_bgr, swapped_float_bgr, mask_float, affine,
               feather: float = 8, erode: float = 0, color_match: bool = True):
    """Composite one selected face, returning an image and full-size alpha.

    Mask settings are pixel radii in the aligned crop. Premultiplied-color
    warping prevents black crop borders from leaking into partially covered
    pixels. Pixels outside the returned mask stay byte-for-byte unchanged.
    """
    original = _image(original_bgr)
    swapped = np.asarray(swapped_float_bgr, dtype=np.float32)
    mask = np.asarray(mask_float, dtype=np.float32)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if swapped.ndim != 3 or swapped.shape[2] != 3 or swapped.shape[:2] != mask.shape:
        raise ValueError("预测人脸和遮罩的尺寸不匹配。")
    if min(mask.shape, default=0) < 1 or not np.isfinite(swapped).all() or not np.isfinite(mask).all():
        raise ValueError("预测人脸或遮罩包含无效值。")
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (2, 3) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:, :2])) < 1e-12:
        raise ValueError("图片对齐变换无效。")
    if not np.isfinite(feather) or not 0 <= feather <= 30 or not np.isfinite(erode) or not 0 <= erode <= 20:
        raise ValueError("羽化需在 0～30，收缩需在 0～20 之间。")
    swapped = np.clip(swapped, 0, 1)
    mask = np.clip(mask.copy(), 0, 1)
    height, width = original.shape[:2]
    crop_height, crop_width = mask.shape
    empty = np.zeros((height, width), dtype=np.uint8)
    if not np.any(mask > 0):
        return original.copy(), empty
    # Define zero beyond the aligned crop before erosion/blur.
    radius = int(round(erode))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    blur_radius = int(math.ceil(feather))
    if blur_radius:
        mask[[0, -1], :] = 0
        mask[:, [0, -1]] = 0
        mask = cv2.GaussianBlur(mask, (blur_radius * 2 + 1, blur_radius * 2 + 1),
                                max(float(feather) / 3, 0.1), borderType=cv2.BORDER_CONSTANT)
    if color_match:
        target = cv2.warpAffine(original.astype(np.float32) / 255, affine,
                                (crop_width, crop_height), flags=cv2.INTER_LINEAR)
        coverage = cv2.warpAffine(np.ones((height, width), dtype=np.float32), affine,
                                  (crop_width, crop_height), flags=cv2.INTER_LINEAR)
        swapped = _color_match(swapped, target, mask * (coverage > 0.999))
    inverse = cv2.invertAffineTransform(affine)
    warp_args = dict(dsize=(width, height), flags=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = np.clip(cv2.warpAffine(mask, inverse, **warp_args), 0, 1)
    mask_u8 = np.rint(alpha * 255).astype(np.uint8)
    active = mask_u8 > 0
    if not np.any(active):
        return original.copy(), empty
    premultiplied = cv2.warpAffine(swapped * mask[:, :, None], inverse, **warp_args)
    composite = np.rint(np.clip(
        original.astype(np.float32) * (1 - alpha[:, :, None]) + premultiplied * 255, 0, 255,
    )).astype(np.uint8)
    result = original.copy()
    result[active] = composite[active]
    return result, mask_u8
