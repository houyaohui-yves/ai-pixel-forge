"""Alignment compatibility and compositing invariants, with synthetic faces."""

import cv2
import numpy as np
import pytest

from facework.geometry import (
    LANDMARK_TEMPLATE, align_face, alignment_matrix, blend_face, quality_report,
)


def landmarks_for_template(scale=100.0, translation=(100.0, 100.0)):
    points = np.zeros((68, 2), dtype=np.float64)
    points[17:49] = LANDMARK_TEMPLATE[:32] * scale + translation
    points[54] = LANDMARK_TEMPLATE[32] * scale + translation
    return points


@pytest.mark.parametrize("kind,padding,offset", [
    ("h", 0, 0), ("mf", 0.0675, 0), ("f", 0.2109375, 0), ("wf", 0.4, -7),
])
def test_template_alignment_matches_dfl_crop_geometry(kind, padding, offset):
    points = landmarks_for_template()
    matrix = alignment_matrix(points, 224, kind)
    # For a 100 px template square, DFL expands each diagonal by this amount.
    half_side = 100 * (padding * np.sqrt(2) + 0.5)
    corners = np.float32([
        [150 - half_side, 150 + offset - half_side],
        [150 + half_side, 150 + offset - half_side],
        [150 + half_side, 150 + offset + half_side],
    ])
    mapped = cv2.transform(corners[None], matrix)[0]
    np.testing.assert_allclose(mapped, [(0, 0), (224, 0), (224, 224)], atol=1e-4)


def test_alignment_removes_translation_rotation_and_scale():
    points = landmarks_for_template()
    angle = 0.4
    transform = np.array([[np.cos(angle), -np.sin(angle), 30],
                          [np.sin(angle), np.cos(angle), -20]], dtype=np.float64)
    transform[:, :2] *= 1.7
    changed = cv2.transform(points[None], transform)[0]
    first = cv2.transform(points[None], alignment_matrix(points, 128))[0]
    second = cv2.transform(changed[None], alignment_matrix(changed, 128))[0]
    np.testing.assert_allclose(first, second, atol=1e-4)


def test_align_face_returns_crop_and_reusable_matrix():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[125:175, 125:175] = 180
    aligned, matrix = align_face(image, landmarks_for_template(), 96)
    assert aligned.shape == (96, 96, 3)
    assert aligned.dtype == np.uint8
    np.testing.assert_array_equal(aligned, cv2.warpAffine(image, matrix, (96, 96), flags=cv2.INTER_CUBIC))


@pytest.mark.parametrize("points", [np.zeros((68, 2)), np.zeros((5, 2)), np.full((68, 2), np.nan)])
def test_invalid_landmarks_are_rejected(points):
    with pytest.raises(ValueError):
        alignment_matrix(points, 224)


def test_unsupported_face_type_is_explicit():
    with pytest.raises(ValueError, match="人脸类型"):
        alignment_matrix(landmarks_for_template(), 224, "head")


def test_empty_mask_preserves_every_pixel():
    image = np.random.default_rng(1).integers(0, 256, (100, 140, 3), dtype=np.uint8)
    result, mask = blend_face(image, np.ones((32, 32, 3)), np.zeros((32, 32)),
                              np.array([[1, 0, -20], [0, 1, -30]]))
    np.testing.assert_array_equal(result, image)
    assert not mask.any()


def test_outside_mask_is_byte_identical_and_second_face_is_untouched():
    image = np.random.default_rng(2).integers(0, 256, (140, 180, 3), dtype=np.uint8)
    alpha = np.zeros((48, 48), np.float32)
    cv2.circle(alpha, (24, 24), 14, 1, -1)
    result, mask = blend_face(image, np.ones((48, 48, 3), np.float32), alpha,
                              np.array([[1, 0, -25], [0, 1, -35]]),
                              feather=5, erode=2, color_match=False)
    assert mask.any()
    assert np.any(result[mask > 0] != image[mask > 0])
    np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
    np.testing.assert_array_equal(result[:, 110:], image[:, 110:])


def test_subpixel_crop_edges_do_not_add_black_halo():
    image = np.full((50, 50, 3), 200, np.uint8)
    face = np.full((20, 20, 3), 200 / 255, np.float32)
    result, mask = blend_face(image, face, np.ones((20, 20), np.float32),
                              np.array([[1, 0, -10.4], [0, 1, -10.2]]),
                              feather=0, erode=0, color_match=False)
    assert mask.any()
    np.testing.assert_array_equal(result, image)


def test_color_match_ignores_unmasked_aligned_background():
    image = np.full((40, 40, 3), (90, 140, 190), np.uint8)
    alpha = np.zeros((20, 20), np.float32)
    alpha[5:15, 5:15] = 1
    face_a = np.zeros((20, 20, 3), np.float32)
    face_a[5:15, 5:15] = (0.3, 0.4, 0.5)
    face_b = face_a.copy()
    face_b[alpha == 0] = 1
    affine = np.array([[1, 0, -10], [0, 1, -10]])
    a, _ = blend_face(image, face_a, alpha, affine, feather=0, color_match=True)
    b, _ = blend_face(image, face_b, alpha, affine, feather=0, color_match=True)
    np.testing.assert_array_equal(a, b)


def test_diagnostics_are_finite_and_explain_low_quality():
    image = np.full((300, 300, 3), 20, dtype=np.uint8)
    report = quality_report(image, (100, 100, 200, 200), landmarks_for_template())
    assert report["sharpness"] == 0
    assert report["brightness"] == 20
    assert -1 <= report["yaw"] <= 1
    assert any("模糊" in warning for warning in report["warnings"])
    assert any("偏暗" in warning for warning in report["warnings"])


def test_blending_rejects_nonfinite_prediction():
    with pytest.raises(ValueError, match="无效"):
        blend_face(np.zeros((40, 40, 3), np.uint8), np.full((20, 20, 3), np.nan),
                   np.ones((20, 20)), np.array([[1, 0, 0], [0, 1, 0]]))
