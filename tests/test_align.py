"""The alignment math: the fitted transform must be a similarity that maps
the detected landmarks onto the template. Run with `python -m unittest discover tests`."""

import unittest

import numpy as np

from facepipe.align import ARCFACE_TEMPLATE_112, ArcFaceAligner, similarity_transform


def apply(matrix, points):
    return points @ matrix[:, :2].T + matrix[:, 2]


class SimilarityTransformTest(unittest.TestCase):
    def test_template_maps_to_itself_with_identity(self):
        m = similarity_transform(ARCFACE_TEMPLATE_112, ARCFACE_TEMPLATE_112)
        np.testing.assert_allclose(m, [[1, 0, 0], [0, 1, 0]], atol=1e-6)

    def test_recovers_a_known_rotation_scale_translation(self):
        theta, scale, shift = np.radians(25.0), 1.8, np.array([40.0, -15.0])
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        template = ARCFACE_TEMPLATE_112.astype(np.float64)  # build the expected values without float32 rounding
        moved = scale * template @ rotation.T + shift
        m = similarity_transform(template, moved)
        np.testing.assert_allclose(m[:, :2], scale * rotation, atol=1e-6)
        np.testing.assert_allclose(m[:, 2], shift, atol=1e-6)

    def test_fit_is_a_similarity_not_a_general_affine(self):
        # Noisy landmarks: the best fit must still be scale x rotation (orthogonal
        # columns of equal length, positive determinant), never a shear or a flip.
        rng = np.random.default_rng(0)
        noisy = ARCFACE_TEMPLATE_112 * 2.5 + 30 + rng.normal(0, 3, ARCFACE_TEMPLATE_112.shape)
        a = similarity_transform(noisy, ARCFACE_TEMPLATE_112)[:, :2]
        np.testing.assert_allclose(a.T @ a, np.eye(2) * (a[:, 0] @ a[:, 0]), atol=1e-9)
        self.assertGreater(np.linalg.det(a), 0)


class ArcFaceAlignerTest(unittest.TestCase):
    def test_crop_shape_and_landmarks_land_on_template(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # A face twice the template's size, rotated 10 degrees, somewhere in the frame.
        theta = np.radians(10.0)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        landmarks = (2.0 * ARCFACE_TEMPLATE_112 @ rotation.T + [200.0, 120.0]).astype(np.float32)
        crop = ArcFaceAligner(112).align(frame, landmarks)
        self.assertEqual(crop.shape, (112, 112, 3))
        self.assertEqual(crop.dtype, np.uint8)
        m = similarity_transform(landmarks, ARCFACE_TEMPLATE_112)
        np.testing.assert_allclose(apply(m, landmarks), ARCFACE_TEMPLATE_112, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
