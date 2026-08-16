import unittest

import numpy as np

from pipeline.segmentation_3d import (
    filter_and_renumber_3d,
    resolve_anisotropy,
    segment_3d,
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


class Segmentation3dTests(unittest.TestCase):
    def test_anisotropy_is_z_spacing_over_xy_spacing(self):
        self.assertAlmostEqual(resolve_anisotropy(81.25, 180.0), 180 / 81.25)
        self.assertEqual(resolve_anisotropy(100, 200, override=1.7), 1.7)
        for bad in (0, -1, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_anisotropy(100, 200, override=bad)

    def test_filter_uses_largest_xy_cross_section_and_renumbers(self):
        labels = np.zeros((3, 9, 9), dtype=np.int32)
        labels[:, 0:2, 0:2] = 3
        labels[1, 3:8, 3:8] = 9
        clean, stats = filter_and_renumber_3d(labels, max_diameter=5.0)
        np.testing.assert_array_equal(np.unique(clean), np.array([0, 1]))
        self.assertTrue(np.all(clean[:, 0:2, 0:2] == 1))
        self.assertTrue(np.all(clean[1, 3:8, 3:8] == 0))
        self.assertEqual(stats["labels_before_oversized_filter"], 2)
        self.assertEqual(stats["oversized_labels_removed"], 1)
        self.assertEqual(stats["final_labels"], 1)

    def test_segment_3d_calls_cellpose_once_with_physical_anisotropy(self):
        stack = np.zeros((4, 8, 9), dtype=np.float32)
        raw_labels = np.zeros_like(stack, dtype=np.int32)
        raw_labels[1:3, 2:5, 3:6] = 7
        model = FakeModel(raw_labels)

        raw, final, summary = segment_3d(
            model,
            stack,
            xy_pixel_nm=100.0,
            z_step_nm=180.0,
            diameter=20,
            min_size=15,
            max_diameter=40,
        )

        np.testing.assert_array_equal(model.image, stack)
        np.testing.assert_array_equal(raw, raw_labels)
        self.assertEqual(model.kwargs["z_axis"], 0)
        self.assertTrue(model.kwargs["do_3D"])
        self.assertAlmostEqual(model.kwargs["anisotropy"], 1.8)
        self.assertEqual(model.kwargs["diameter"], 20.0)
        self.assertEqual(model.kwargs["min_size"], 15)
        self.assertEqual(final.shape, stack.shape)
        self.assertEqual(final.dtype, np.uint16)
        self.assertEqual(final.max(), 1)
        self.assertEqual(summary["segmentation_mode"], "3d")
        self.assertEqual(summary["z_policy"], "direct_volumetric_instances")
        self.assertAlmostEqual(summary["anisotropy_z_over_xy"], 1.8)

    def test_segment_3d_rejects_wrong_input_or_output_shape(self):
        with self.assertRaisesRegex(ValueError, "requires a .* stack"):
            segment_3d(
                FakeModel(np.zeros((5, 5), dtype=np.int32)),
                np.zeros((5, 5), dtype=np.float32),
                xy_pixel_nm=100,
                z_step_nm=100,
            )
        with self.assertRaisesRegex(ValueError, "returned shape"):
            segment_3d(
                FakeModel(np.zeros((2, 5, 5), dtype=np.int32)),
                np.zeros((3, 5, 5), dtype=np.float32),
                xy_pixel_nm=100,
                z_step_nm=100,
            )


if __name__ == "__main__":
    unittest.main()
