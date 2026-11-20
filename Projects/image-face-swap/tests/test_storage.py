"""Image orientation, DFL export interoperability and local data boundaries."""

import base64
import io
import json
import pickle
import struct
import warnings
import zipfile
import zlib
from pathlib import PurePosixPath

import cv2
import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from facework.geometry import LANDMARK_TEMPLATE
from facework.storage import Store, decode_image, dfl_jpeg


def synthetic_face():
    image = np.random.default_rng(12).integers(0, 256, (300, 300, 3), dtype=np.uint8)
    landmarks = np.zeros((68, 2), dtype=np.float64)
    landmarks[17:49] = LANDMARK_TEMPLATE[:32] * 100 + (100, 100)
    landmarks[54] = LANDMARK_TEMPLATE[32] * 100 + (100, 100)
    return image, {"landmarks": landmarks.tolist(), "box": [90, 90, 210, 220]}


def test_face_export_is_decodable_dfl_jpeg_with_original_geometry(tmp_path):
    store = Store(tmp_path)
    identity = store.create_identity("测试人物")
    source, face = synthetic_face()
    item = store.add_face(identity["id"], source, face, "photo.png")
    archive = zipfile.ZipFile(io.BytesIO(store.export(identity["id"])))
    jpeg = archive.read("aligned/" + item["id"] + ".jpg")
    # Pillow independently parses JPEG APP markers; DFL reads APP15 as pickle.
    with Image.open(io.BytesIO(jpeg)) as image:
        assert image.size == (256, 256)
        app15 = [payload for marker, payload in image.applist if marker == "APP15"]
        assert len(app15) == 1
        metadata = pickle.loads(app15[0])  # Only our own generated builtin-only fixture.
        image.load()
    assert metadata["face_type"] == "full_face"
    assert metadata["source_rect"] == face["box"]
    transform = np.asarray(metadata["image_to_face_mat"])
    source_landmarks = np.asarray(metadata["source_landmarks"], np.float32)
    np.testing.assert_allclose(
        cv2.transform(source_landmarks[None], transform)[0], metadata["landmarks"], atol=1e-5,
    )
    # The source_filename stored by DFL can be resolved within exported sources/.
    original_png = archive.read("sources/" + metadata["source_filename"])
    decoded = cv2.imdecode(np.frombuffer(original_png, np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, source)
    assert json.loads(archive.read("manifest.json"))["count"] == 1


def test_duplicate_face_is_rejected_without_extra_files(tmp_path):
    store = Store(tmp_path)
    identity = store.create_identity("人物")
    source, face = synthetic_face()
    store.add_face(identity["id"], source, face, "first.png")
    directory = store.identity_path(identity["id"])
    filenames_before = {p.name for p in directory.iterdir()}
    with pytest.raises(ValueError, match="重复"):
        store.add_face(identity["id"], source, face, "different-name.png")
    assert store.identity(identity["id"])["count"] == 1
    assert {p.name for p in directory.iterdir()} == filenames_before


def test_uploaded_filename_cannot_escape_metadata_or_export_paths(tmp_path):
    store = Store(tmp_path)
    identity = store.create_identity("../../人物")
    source, face = synthetic_face()
    item = store.add_face(identity["id"], source, face, r"C:\private\..\portrait.png")
    assert item["filename"] == "portrait.png"
    metadata_path = PurePosixPath(item["metadata"]["source_filename"])
    assert not metadata_path.is_absolute()
    assert len(metadata_path.parts) == 1
    with zipfile.ZipFile(io.BytesIO(store.export(identity["id"]))) as archive:
        for name in archive.namelist():
            path = PurePosixPath(name)
            assert not path.is_absolute()
            assert ".." not in path.parts


def test_forged_item_id_cannot_read_files_outside_identity(tmp_path):
    store = Store(tmp_path)
    identity = store.create_identity("人物")
    source, face = synthetic_face()
    item = store.add_face(identity["id"], source, face, "source.png")
    directory = store.identity_path(identity["id"])
    record = directory / ("item-" + item["id"] + ".json")
    modified = json.loads(record.read_text())
    modified["id"] = "../../outside"
    record.write_text(json.dumps(modified))
    # If an unchecked path is used, these bytes would leak into the export ZIP.
    (tmp_path / "outside.jpg").write_bytes(b"private-test-sentinel")
    (tmp_path / "outside-source.png").write_bytes(b"private-test-sentinel")
    with pytest.raises(ValueError):
        store.export(identity["id"])


def test_exif_orientation_is_applied_before_analysis():
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    rgb[:, :40] = (255, 0, 0)
    rgb[:, 40:] = (0, 0, 255)
    image = Image.fromarray(rgb)
    exif = image.getexif()
    exif[274] = 6  # Rotate 90 degrees clockwise on display.
    stream = io.BytesIO()
    image.save(stream, "JPEG", quality=100, subsampling=0, exif=exif)
    decoded = decode_image(base64.b64encode(stream.getvalue()).decode())
    assert decoded.shape == (80, 40, 3)
    np.testing.assert_allclose(decoded[10, 20], (0, 0, 255), atol=2)
    np.testing.assert_allclose(decoded[70, 20], (255, 0, 0), atol=2)


def test_camera_photo_at_6192_by_4128_and_300_dpi_keeps_original_pixels():
    # Reproduce the reported camera dimensions with a real, lossless image.
    # DPI is metadata: it must not affect pixel dimensions or trigger resampling.
    stream = io.BytesIO()
    with Image.new("RGB", (6192, 4128), (17, 39, 211)) as image:
        image.putpixel((6191, 4127), (250, 2, 9))
        image.save(stream, "PNG", dpi=(300, 300))
    decoded = decode_image(base64.b64encode(stream.getvalue()).decode())
    assert decoded.shape == (4128, 6192, 3)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded[0, 0], (211, 39, 17))
    np.testing.assert_array_equal(decoded[-1, -1], (9, 2, 250))


def png_header_only(width, height):
    """A parseable header without a pixel payload, so large-size tests stay small."""
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
    raw = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", b"") + chunk(b"IEND", b""))
    return base64.b64encode(raw).decode()


def test_64_megapixel_boundary_is_allowed_to_reach_pixel_decoding(monkeypatch):
    class PixelDecodingReached(Exception):
        pass

    def stop_before_allocation(_image, *args, **kwargs):
        raise PixelDecodingReached

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", stop_before_allocation)
    with pytest.raises(PixelDecodingReached):
        decode_image(png_header_only(8000, 8000))


@pytest.mark.parametrize("size", [(8001, 8000), (16000, 16000)])
def test_oversized_photo_is_rejected_before_pixel_decoding(monkeypatch, size):
    def must_not_decode(_image, *args, **kwargs):
        pytest.fail("Oversized input reached pixel decoding")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", must_not_decode)
    # Pillow may warn before the application's clearer size error for inputs
    # between its warning and error thresholds. Both paths must reject early.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with pytest.raises(ValueError, match="6400 万像素"):
            decode_image(png_header_only(*size))


def test_too_narrow_photo_reports_dimensions_instead_of_upper_pixel_limit():
    with pytest.raises(ValueError, match="31×64.*至少 32"):
        decode_image(png_header_only(31, 64))


def test_malformed_data_uri_has_user_facing_validation_error():
    with pytest.raises(ValueError):
        decode_image("data:image/png;base64")


def test_dfl_metadata_size_limit_is_checked():
    with pytest.raises(ValueError, match="APP15"):
        dfl_jpeg(np.zeros((64, 64, 3), dtype=np.uint8), {"name": "x" * 70_000})


def test_deleting_face_removes_original_and_alignment(tmp_path):
    store = Store(tmp_path)
    identity = store.create_identity("人物")
    source, face = synthetic_face()
    item = store.add_face(identity["id"], source, face, "source.png")
    store.delete_item(identity["id"], item["id"])
    assert store.identity(identity["id"])["count"] == 0
    directory = store.identity_path(identity["id"])
    assert {p.name for p in directory.iterdir()} == {"identity.json"}
