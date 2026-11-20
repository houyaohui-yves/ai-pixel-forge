"""Local-only storage, source-image validation and DFL-compatible face exports."""
import base64
import hashlib
import io
import json
import pickle
import re
import struct
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .geometry import align_face, quality_report

MAX_IMAGE_BYTES = 30 * 1024**2
# Accept full-resolution camera exports, including 6192×4128 and 61 MP images.
# This bounds decoded memory; detector/model crops have their own input sizes.
MAX_PIXELS = 64_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def decode_image(encoded):
    if not isinstance(encoded, str) or len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 1024:
        raise ValueError(f"图片超过 {MAX_IMAGE_BYTES // 1024**2} MiB 限制")
    if encoded.startswith("data:"):
        if "," not in encoded:
            raise ValueError("图片数据 URL 缺少内容")
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"图片超过 {MAX_IMAGE_BYTES // 1024**2} MiB 限制")
        with Image.open(io.BytesIO(raw)) as source:
            width, height = source.size
            if min(width, height) < 32:
                raise ValueError(f"图片尺寸为 {width}×{height} 像素，宽和高都需至少 32 像素，请使用更大尺寸的原图")
            pixels = width * height
            if pixels > MAX_PIXELS:
                raise ValueError(
                    f"图片尺寸为 {width}×{height} 像素（约 {pixels / 10_000:.1f} 万像素），"
                    f"超过当前 {MAX_PIXELS / 10_000:g} 万像素的导入上限"
                )
            image = ImageOps.exif_transpose(source).convert("RGB")
            return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    except Image.DecompressionBombError as exc:
        raise ValueError(f"图片像素数过大，超过当前 {MAX_PIXELS / 10_000:g} 万像素的导入上限") from exc
    except (OSError, base64.binascii.Error) as exc:
        raise ValueError("无法读取图片，请使用正常的 JPG、PNG 或 WebP 文件") from exc


def save_png(path, bgr):
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("图片编码失败")
    Path(path).write_bytes(encoded.tobytes())


def dfl_jpeg(aligned, metadata):
    """DFLJPG APP15 serialization (DeepFaceLab/DFLIMG/DFLJPG.py, GPLv3).

    Metadata uses lists / builtin types so it is readable across NumPy versions.
    """
    ok, encoded = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 98])
    if not ok:
        raise ValueError("人脸 JPEG 编码失败")
    payload = pickle.dumps(metadata, protocol=4)
    if len(payload) > 65533:
        raise ValueError("人脸元数据超过 JPEG APP15 限制")
    image = encoded.tobytes()
    return image[:2] + b"\xff\xef" + struct.pack(">H", len(payload) + 2) + payload + image[2:]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        for name in ("images", "results", "identities", "models"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_id(value):
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError("无效记录编号")
        return value

    def identity_path(self, identity_id):
        path = (self.root / "identities" / self.validate_id(identity_id)).resolve()
        if not path.is_relative_to(self.root / "identities"):
            raise ValueError("无效素材库路径")
        if not (path / "identity.json").is_file():
            raise ValueError("人物素材库不存在")
        return path

    def identities(self):
        return [self.identity(p.parent.name, include_items=False) for p in sorted((self.root / "identities").glob("*/identity.json"))]

    def create_identity(self, name):
        name = str(name).strip()
        if not name or len(name) > 80:
            raise ValueError("人物名称需为 1～80 个字符")
        identity_id = uuid.uuid4().hex
        directory = self.root / "identities" / identity_id
        directory.mkdir()
        write_json(directory / "identity.json", {"id": identity_id, "name": name})
        return {"id": identity_id, "name": name, "count": 0}

    def identity(self, identity_id, include_items=True):
        directory = self.identity_path(identity_id)
        info = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
        items = []
        for path in sorted(directory.glob("item-*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            self.validate_id(item.get("id"))
            if path.name != "item-" + item["id"] + ".json":
                raise ValueError("素材记录与文件名不匹配")
            items.append(item)
        info["count"] = len(items)
        if include_items:
            info["items"] = items
            bins = {"left": 0, "front": 0, "right": 0}
            for item in items:
                yaw = item["quality"]["yaw"]
                bins["left" if yaw < -0.15 else "right" if yaw > 0.15 else "front"] += 1
            info["coverage"] = bins
        return info

    def add_face(self, identity_id, image, face, filename):
        directory = self.identity_path(identity_id)
        landmarks = np.asarray(face["landmarks"], np.float32)
        crop, affine = align_face(image, landmarks, 256, "f")
        digest = hashlib.sha256(crop.tobytes()).hexdigest()
        for item in self.identity(identity_id)["items"]:
            if item["sha256"] == digest:
                raise ValueError("该人脸图片已经在此素材库中，未重复保存")
        item_id = uuid.uuid4().hex
        safe_filename = Path(str(filename).replace("\\", "/")).name[:160] or "photo"
        transformed = cv2.transform(landmarks[None, ...], affine)[0]
        metadata = {"face_type": "full_face", "landmarks": transformed.tolist(),
                    "source_filename": item_id + "-source.png", "source_rect": list(face["box"]),
                    "source_landmarks": landmarks.tolist(), "image_to_face_mat": affine.tolist()}
        (directory / (item_id + ".jpg")).write_bytes(dfl_jpeg(crop, metadata))
        save_png(directory / (item_id + "-source.png"), image)
        item = {"id": item_id, "url": f"/data/identities/{identity_id}/{item_id}.jpg",
                "filename": safe_filename, "quality": quality_report(image, face["box"], landmarks),
                "sha256": digest, "metadata": metadata}
        write_json(directory / ("item-" + item_id + ".json"), item)
        return item

    def delete_item(self, identity_id, item_id):
        directory = self.identity_path(identity_id)
        self.validate_id(item_id)
        record = directory / ("item-" + item_id + ".json")
        if not record.is_file():
            raise ValueError("素材不存在")
        for name in ("item-" + item_id + ".json", item_id + ".jpg", item_id + "-source.png"):
            (directory / name).unlink(missing_ok=True)

    def export(self, identity_id):
        directory = self.identity_path(identity_id)
        info = self.identity(identity_id)
        if not info["items"]:
            raise ValueError("素材库为空")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(info, ensure_ascii=False, indent=2))
            for item in info["items"]:
                name = self.validate_id(item["id"])
                archive.write(directory / (name + ".jpg"), "aligned/" + name + ".jpg")
                archive.write(directory / (name + "-source.png"), "sources/" + name + "-source.png")
        return output.getvalue()
