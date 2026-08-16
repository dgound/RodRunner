import tempfile
import unittest
from pathlib import Path

from pipeline import dat_io


class TimepointNameTests(unittest.TestCase):
    def test_tiff_names_match_iteration_order_without_loading_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "TP010.tif",
                "TP002_CYC001.tif",
                "TP002_CYC000.tif",
                "TPbad.tif",
                "unrelated.tif",
            ):
                (root / name).touch()

            self.assertEqual(
                dat_io.timepoint_names(str(root), 0, "tiff"),
                ["TP002_CYC000", "TP002_CYC001", "TP010"],
            )
            self.assertEqual(dat_io.n_timepoints(str(root), 0, "tiff"), 3)

    def test_unknown_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown fmt"):
            dat_io.timepoint_names("unused", 0, "unknown")


if __name__ == "__main__":
    unittest.main()
