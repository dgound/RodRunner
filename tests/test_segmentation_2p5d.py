import unittest

import numpy as np

from pipeline.segmentation_2p5d import (
    extrude_labels_through_z,
    filter_and_renumber_2d,
    segment_2p5d,
    z_max_projection,
)


class FakeModel:
    def __init__(self, labels):
        self.labels = labels
        self.image = None
        self.kwargs = None

    def eval(self, image, **kwargs):
        self.image = image.copy()
        self.kwargs = kwargs
        return self.labels.copy(), None, None


class Segmentation2p5dTests(unittest.TestCase):
    def test_z_max_projection(self):
        stack = np.array(
            [
                [[1, 8], [3, 4]],
                [[5, 2], [7, 0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            z_max_projection(stack),
            np.array([[5, 8], [7, 4]], dtype=np.float32),
        )

    def test_z_max_projection_rejects_non_3d_input(self):
        with self.assertRaisesRegex(ValueError, "requires a 3D"):
            z_max_projection(np.zeros((5, 5), dtype=np.float32))

    def test_filter_removes_oversized_and_renumbers_gaps(self):
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[0:2, 0:2] = 2
        labels[3:7, 3:7] = 7
        clean, stats = filter_and_renumber_2d(labels, max_diameter=4.0)
        np.testing.assert_array_equal(np.unique(clean), np.array([0, 1]))
        self.assertTrue(np.all(clean[0:2, 0:2] == 1))
        self.assertTrue(np.all(clean[3:7, 3:7] == 0))
        self.assertEqual(stats["labels_before_oversized_filter"], 2)
        self.assertEqual(stats["oversized_labels_removed"], 1)
        self.assertEqual(stats["final_labels"], 1)

    def test_extrusion_preserves_xy_ownership_on_every_plane(self):
        labels = np.array([[0, 1], [2, 2]], dtype=np.uint16)
        volume = extrude_labels_through_z(labels, 3)
        self.assertEqual(volume.shape, (3, 2, 2))
        for plane in volume:
            np.testing.assert_array_equal(plane, labels)

    def test_segment_2p5d_calls_cellpose_once_and_returns_3d_contract(self):
        stack = np.zeros((4, 6, 7), dtype=np.float32)
        stack[2, 1:3, 2:4] = 10
        raw_labels = np.zeros((6, 7), dtype=np.int32)
        raw_labels[1:3, 2:4] = 5
        model = FakeModel(raw_labels)

        projection, labels_2d, labels_3d, summary = segment_2p5d(
            model,
            stack,
            diameter=20,
            min_size=15,
            max_diameter=40,
        )

        np.testing.assert_array_equal(model.image, stack.max(axis=0))
        self.assertEqual(model.kwargs["diameter"], 20.0)
        self.assertFalse(model.kwargs["do_3D"])
        self.assertEqual(labels_2d.shape, (6, 7))
        self.assertEqual(labels_3d.shape, stack.shape)
        self.assertEqual(labels_3d.max(), 1)
        self.assertEqual(summary["segmentation_mode"], "2p5d")
        self.assertEqual(summary["z_policy"], "full_stack_extrusion")
        self.assertEqual(summary["final_labels"], 1)
        np.testing.assert_array_equal(projection, stack.max(axis=0))


if __name__ == "__main__":
    unittest.main()
