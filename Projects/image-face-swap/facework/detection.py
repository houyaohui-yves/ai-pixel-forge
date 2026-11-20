"""DeepFaceLab S3FD + 2DFAN inference, ported to PyTorch.

Derived from DeepFaceLab/facelib/{S3FDExtractor,FANExtractor}.py,
licensed under GNU GPL v3. FAN in DeepFaceLab credits
https://github.com/1adrianb/face-alignment. See LICENSE and THIRD_PARTY.md.

The pretrained weights remain in their original DFL format. This module does
not import TensorFlow, invoke DeepFaceLab, or download models. BGR images are
converted to RGB for both upstream detector networks; tensors use NCHW.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .weights import TensorWeights


def _nms(boxes: np.ndarray, threshold: float = 0.3) -> list[int]:
    """DFL's inclusive-coordinate non-maximum suppression."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2, scores = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        others = order[1:]
        width = np.maximum(0, np.minimum(x2[index], x2[others]) - np.maximum(x1[index], x1[others]) + 1)
        height = np.maximum(0, np.minimum(y2[index], y2[others]) - np.maximum(y1[index], y1[others]) + 1)
        intersection = width * height
        union = areas[index] + areas[others] - intersection
        iou = intersection / np.maximum(union, 1e-12)
        order = others[iou <= threshold]
    return keep


def _fan_transform(points: np.ndarray, center: np.ndarray, scale: float, resolution: int) -> np.ndarray:
    """Inverse FAN crop transform, matching upstream coordinate convention."""
    return np.asarray(points, dtype=np.float64) * (200.0 * scale / resolution) + center - 100.0 * scale


def _fan_crop(rgb: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    # Keep upstream's one-based border convention. Replacing this with a simple
    # rectangle resize introduces an approximately one-pixel landmark shift.
    upper_left = _fan_transform(np.array([1, 1]), center, scale, 256).astype(np.int64)
    bottom_right = _fan_transform(np.array([256, 256]), center, scale, 256).astype(np.int64)
    width, height = (bottom_right - upper_left).tolist()
    if width <= 0 or height <= 0 or max(width, height) > 16384 or width * height > 64_000_000:
        raise ValueError("人脸裁剪范围无效或过大，请使用清晰的近景照片。")
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    source_height, source_width = rgb.shape[:2]
    new_x = np.array([max(1, -upper_left[0] + 1), min(bottom_right[0], source_width) - upper_left[0]], dtype=np.int64)
    new_y = np.array([max(1, -upper_left[1] + 1), min(bottom_right[1], source_height) - upper_left[1]], dtype=np.int64)
    old_x = np.array([max(1, upper_left[0] + 1), min(bottom_right[0], source_width)], dtype=np.int64)
    old_y = np.array([max(1, upper_left[1] + 1), min(bottom_right[1], source_height)], dtype=np.int64)
    if old_x[1] > old_x[0] and old_y[1] > old_y[0]:
        crop[new_y[0] - 1:new_y[1], new_x[0] - 1:new_x[1]] = rgb[old_y[0] - 1:old_y[1], old_x[0] - 1:old_x[1]]
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)


def _landmarks_from_heatmaps(heatmaps: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    channels, height, width = heatmaps.shape
    if channels != 68 or not np.isfinite(heatmaps).all():
        raise ValueError("关键点网络输出无效。")
    indices = heatmaps.reshape(channels, -1).argmax(axis=1)
    points = np.column_stack((indices % width, indices // width)).astype(np.float64)
    for index in range(channels):
        x, y = points[index].astype(int)
        if 0 < x < width - 1 and 0 < y < height - 1:
            gradient = np.array([
                heatmaps[index, y, x + 1] - heatmaps[index, y, x - 1],
                heatmaps[index, y + 1, x] - heatmaps[index, y - 1, x],
            ])
            points[index] += np.sign(gradient) * 0.25
    return _fan_transform(points + 0.5, center, scale, width).astype(np.float32)


class FaceDetector:
    """Cached DFL networks for reusable image/frame face detection.

    ``detect`` returns at most eight faces, largest first. Landmark confidence
    is not exposed as an identity or image-quality probability; ``score`` is
    the S3FD face-detection score only. DFL's optional second-pass and jittered
    landmark averaging are intentionally not included in this first version.
    """

    def __init__(self, asset_dir: Path, device: str = "cpu"):
        self.device = str(device)
        asset_dir = Path(asset_dir)
        self.s3fd = TensorWeights(asset_dir / "S3FD.npy", device)
        self.fan = TensorWeights(asset_dir / "2DFAN.npy", device)

    def _s3fd_forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        weights = self.s3fd
        x = image - torch.tensor([104.0, 117.0, 123.0], device=image.device).reshape(1, 3, 1, 1)
        features = []
        for stage, count in ((1, 2), (2, 2), (3, 3), (4, 3), (5, 3)):
            for index in range(1, count + 1):
                x = F.relu(weights.conv(x, f"conv{stage}_{index}"))
            if stage >= 3:
                features.append(x)
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(weights.conv(x, "fc6", padding=3))
        x = F.relu(weights.conv(x, "fc7"))
        features.append(x)
        x = F.relu(weights.conv(x, "conv6_1"))
        x = F.relu(weights.conv(x, "conv6_2", stride=2))
        features.append(x)
        x = F.relu(weights.conv(x, "conv7_1"))
        x = F.relu(weights.conv(x, "conv7_2", stride=2))
        features.append(x)
        prefixes = ("conv3_3_norm", "conv4_3_norm", "conv5_3_norm", "fc7", "conv6_2", "conv7_2")
        output = []
        for index, (feature, prefix) in enumerate(zip(features, prefixes)):
            if index < 3:
                factor = weights.tensor(prefix + "/weight:0").reshape(1, -1, 1, 1)
                feature = feature / (torch.sqrt(torch.sum(feature.square(), dim=1, keepdim=True)) + 1e-10) * factor
            classification = weights.conv(feature, prefix + "_mbox_conf")
            regression = weights.conv(feature, prefix + "_mbox_loc")
            if index == 0:
                background = classification[:, :3].amax(dim=1, keepdim=True)
                classification = torch.cat((background, classification[:, 3:4]), dim=1)
            output.extend((F.softmax(classification, dim=1), regression))
        return output

    def _s3fd_boxes(self, rgb: np.ndarray) -> list[tuple[list[int], float]]:
        height, width = rgb.shape[:2]
        longest = max(width, height)
        target = max(64.0, 640.0 if longest >= 1280 else longest / 2.0)
        scale = longest / target
        resized_width, resized_height = max(1, int(width / scale)), max(1, int(height / scale))
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        # Very narrow images need padding to survive five pooling stages. The
        # valid image extent stays explicit so padding cannot become a face.
        if min(resized_height, resized_width) < 64:
            resized = cv2.copyMakeBorder(resized, 0, max(0, 64 - resized_height), 0, max(0, 64 - resized_width), cv2.BORDER_CONSTANT)
        tensor = torch.from_numpy(resized.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        output = self._s3fd_forward(tensor)
        candidates = []
        for index in range(6):
            classification = output[index * 2][0, 1].cpu().numpy()
            regression = output[index * 2 + 1][0].permute(1, 2, 0).cpu().numpy()
            yy, xx = np.where(classification > 0.05)
            if len(xx) == 0:
                continue
            stride = 2 ** (index + 2)
            sizes = float(stride * 4)
            locations = regression[yy, xx]
            centers = np.column_stack((xx * stride + stride / 2, yy * stride + stride / 2)) + locations[:, :2] * 0.1 * sizes
            dimensions = sizes * np.exp(np.clip(locations[:, 2:] * 0.2, -50, 50))
            boxes = np.column_stack((centers - dimensions / 2, centers + dimensions / 2, classification[yy, xx]))
            candidates.append(boxes[np.isfinite(boxes).all(axis=1)])
        if not candidates:
            return []
        boxes = np.concatenate(candidates)
        result = []
        for index in _nms(boxes):
            box = boxes[index]
            if box[4] < 0.5:
                continue
            # Upstream truncates predicted coordinates before rescaling.
            left, top, right, bottom = box[:4].astype(np.int64) * scale
            if min(right - left, bottom - top) < 40:
                continue
            if not (0 <= (left + right) / 2 < width and 0 <= (top + bottom) / 2 < height):
                continue
            bottom += (bottom - top) * 0.1
            rectangle = [int(value) for value in (left, top, right, bottom)]
            result.append((rectangle, float(box[4])))
        result.sort(key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), reverse=True)
        return result[:8]

    def _fan_block(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        weights = self.fan
        out1 = weights.conv(F.relu(weights.bn(x, prefix + "/bn1")), prefix + "/conv1")
        out2 = weights.conv(F.relu(weights.bn(out1, prefix + "/bn2")), prefix + "/conv2")
        out3 = weights.conv(F.relu(weights.bn(out2, prefix + "/bn3")), prefix + "/conv3")
        combined = torch.cat((out1, out2, out3), dim=1)
        if combined.shape[1] != x.shape[1]:
            skip = weights.conv(F.relu(weights.bn(x, prefix + "/down_bn1")), prefix + "/down_conv1", padding=0)
        else:
            skip = x
        return combined + skip

    def _hourglass(self, x: torch.Tensor, prefix: str, depth: int) -> torch.Tensor:
        upper = self._fan_block(x, prefix + "/b1")
        lower = self._fan_block(F.avg_pool2d(x, 2, 2), prefix + "/b2")
        if depth > 1:
            lower = self._hourglass(lower, prefix + "/b2_plus", depth - 1)
        else:
            lower = self._fan_block(lower, prefix + "/b2_plus")
        lower = self._fan_block(lower, prefix + "/b3")
        return upper + F.interpolate(lower, scale_factor=2, mode="nearest")

    def _fan_forward(self, image: torch.Tensor) -> torch.Tensor:
        weights = self.fan
        x = weights.conv(image, "conv1", stride=2)
        x = F.relu(weights.bn(x, "bn1"))
        x = self._fan_block(x, "conv2")
        x = F.avg_pool2d(x, 2, 2)
        x = self._fan_block(x, "conv3")
        previous = self._fan_block(x, "conv4")
        prediction = None
        for index in range(4):
            x = self._hourglass(previous, f"m_{index}", 4)
            x = self._fan_block(x, f"top_m_{index}")
            x = weights.conv(x, f"conv_last_{index}", padding=0)
            x = F.relu(weights.bn(x, f"bn_end_{index}"))
            prediction = weights.conv(x, f"l_{index}", padding=0)
            if index < 3:
                previous = previous + weights.conv(x, f"bl_{index}", padding=0) + weights.conv(prediction, f"al_{index}", padding=0)
        return prediction

    @torch.inference_mode()
    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
            raise ValueError("人脸检测需要三通道 uint8 BGR 图片。")
        if min(image_bgr.shape[:2]) < 2:
            raise ValueError("图片尺寸过小。")
        rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        faces = []
        for box, score in self._s3fd_boxes(rgb):
            left, top, right, bottom = box
            scale = (right - left + bottom - top) / 195.0
            center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
            crop = _fan_crop(rgb, center, scale)
            tensor = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            heatmaps = self._fan_forward(tensor)[0].cpu().numpy()
            landmarks = _landmarks_from_heatmaps(heatmaps, center, scale)
            faces.append({"box": box, "landmarks": landmarks, "score": score})
        return faces
