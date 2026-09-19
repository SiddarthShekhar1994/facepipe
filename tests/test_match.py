"""The matching math: cosine on unit vectors, the threshold gate, best-row
scoring per person, ordering and top_k. Synthetic vectors, no models."""

import unittest

import numpy as np

from facepipe.matcher import CosineMatcher, build_gallery
from facepipe.types import Identity


def unit(*v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


class CosineMatcherTest(unittest.TestCase):
    def setUp(self):
        # alice enrolled twice (two directions), bob once, in a 4-d toy space.
        self.alice_a, self.alice_b, self.bob = unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0)
        self.names = ["alice", "alice", "bob"]
        self.gallery = np.stack([self.alice_a, self.alice_b, self.bob])

    def test_identical_scores_one_and_orthogonal_is_unknown(self):
        matches = CosineMatcher(threshold=0.5, top_k=3).match(self.alice_a, self.names, self.gallery)
        self.assertEqual([m.name for m in matches], ["alice"])
        self.assertAlmostEqual(matches[0].similarity, 1.0, places=6)
        self.assertEqual(CosineMatcher(0.5, 3).match(unit(0, 0, 0, 1), self.names, self.gallery), [])

    def test_threshold_gate(self):
        query = unit(1, 1, 0, 0)  # cosine 0.7071 with alice_a and alice_b, 0 with bob
        self.assertEqual(len(CosineMatcher(0.70, 3).match(query, self.names, self.gallery)), 1)
        self.assertEqual(CosineMatcher(0.71, 3).match(query, self.names, self.gallery), [])

    def test_person_is_scored_by_their_best_row(self):
        query = unit(0.1, 1, 0, 0)  # close to alice_b, far from alice_a
        matches = CosineMatcher(0.5, 3).match(query, self.names, self.gallery)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].name, "alice")
        self.assertAlmostEqual(matches[0].similarity, float(query @ self.alice_b), places=6)

    def test_ordered_best_first_and_truncated_to_top_k(self):
        query = unit(0.5, 0, 1, 0)  # bob 0.894, alice 0.447
        matches = CosineMatcher(0.4, 3).match(query, self.names, self.gallery)
        self.assertEqual([m.name for m in matches], ["bob", "alice"])
        self.assertEqual([m.name for m in CosineMatcher(0.4, 1).match(query, self.names, self.gallery)], ["bob"])

    def test_empty_gallery_is_unknown(self):
        names, gallery = build_gallery([])
        self.assertEqual(CosineMatcher(0.4, 3).match(self.alice_a, names, gallery), [])


class BuildGalleryTest(unittest.TestCase):
    def test_one_name_per_row_in_store_order(self):
        identities = [
            Identity("alice", np.stack([unit(1, 0), unit(0, 1)]), ["a1.jpg", "a2.jpg"]),
            Identity("bob", np.stack([unit(1, 1)]), ["b1.jpg"]),
        ]
        names, gallery = build_gallery(identities)
        self.assertEqual(names, ["alice", "alice", "bob"])
        self.assertEqual(gallery.shape, (3, 2))
        self.assertEqual(gallery.dtype, np.float32)
        np.testing.assert_allclose(gallery[2], unit(1, 1))


if __name__ == "__main__":
    unittest.main()
