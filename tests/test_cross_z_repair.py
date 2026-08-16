import unittest

import numpy as np

from pipeline.mask_stitching import (
    broad_overlap_candidates,
    filter_oversized_masks,
    stitch_3d,
)
from pipeline.repair_cross_z_masks import repair_cross_z


class CrossZRepairTest(unittest.TestCase):
    def setUp(self):
        self.masks = np.zeros((2, 16, 20), dtype=np.uint16)
        self.masks[0, 3:13, 2:8] = 1
        self.masks[0, 3:13, 10:16] = 2
        # Cellpose merged both predecessor masks in the next plane.
        self.masks[1, 3:13, 2:16] = 1

        self.image = np.zeros(self.masks.shape, dtype=np.float32)
        yy, xx = np.indices(self.masks.shape[1:])
        self.image[1] = (
            np.exp(-((yy - 8) ** 2 + (xx - 5) ** 2) / 8)
            + np.exp(-((yy - 8) ** 2 + (xx - 13) ** 2) / 8)
        )

    def test_broad_overlap_finds_merged_current_mask(self):
        stitched = stitch_3d(self.masks, stitch_threshold=0.5)
        candidates = broad_overlap_candidates(
            self.masks, stitched, min_iou=0.05
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["z"], 1)
        self.assertEqual(candidates[0]["current_source_label"], 1)
        self.assertEqual(candidates[0]["previous_labels"], [1, 2])

    def test_repair_preserves_footprint_and_predecessor_identity(self):
        repaired, audit, _ = repair_cross_z(self.masks, self.image)
        self.assertTrue(np.array_equal(repaired > 0, self.masks > 0))
        self.assertEqual(len(audit), 1)
        self.assertEqual(
            set(np.unique(repaired[1][self.masks[1] > 0])), {1, 2}
        )
        self.assertTrue(np.all(repaired[1, 3:13, 2:8] == 1))
        self.assertTrue(np.all(repaired[1, 3:13, 10:16] == 2))

        restitched = stitch_3d(repaired, stitch_threshold=0.5)
        self.assertTrue(np.all(restitched[1, 3:13, 2:8] == 1))
        self.assertTrue(np.all(restitched[1, 3:13, 10:16] == 2))

    def test_diagonally_attached_residue_is_kept_not_dropped(self):
        # The marker flood is 4-connected, so a pixel touching the merged body
        # only diagonally is unreachable. It must still be assigned, otherwise
        # the footprint zeroing drops it and the foreground support changes.
        masks = self.masks.copy()
        masks[1, 13, 16] = 1
        repaired, audit, _ = repair_cross_z(masks, self.image)
        self.assertTrue(np.array_equal(repaired > 0, masks > 0))
        self.assertNotEqual(int(repaired[1, 13, 16]), 0)
        self.assertEqual(audit[0]["unlabeled_residue_px"], 1)

    def test_oversized_filter_uses_largest_plane_area(self):
        masks = np.zeros((2, 20, 20), dtype=np.uint16)
        masks[:, 1:4, 1:4] = 1
        masks[0, 5:15, 5:15] = 2
        cleaned, removed = filter_oversized_masks(
            masks, max_diameter=8
        )
        self.assertEqual(removed.tolist(), [2])
        self.assertEqual(int(cleaned.max()), 1)
        self.assertTrue(np.all(cleaned[masks == 2] == 0))


if __name__ == "__main__":
    unittest.main()
