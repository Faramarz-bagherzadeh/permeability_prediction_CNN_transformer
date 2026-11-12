import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed.")
class ResizeBatchTests(unittest.TestCase):
    def test_minus_one_preserves_original_dimensions_and_binarizes_images(self) -> None:
        import torch

        from model_evaluation import resize_batch

        images = torch.tensor(
            [[[[[0.0, 0.49], [0.5, 1.0]], [[0.25, 0.75], [0.0, 1.0]]]]]
        )
        resized = resize_batch(images, torch.device("cpu"), image_size=-1)

        self.assertEqual(tuple(resized.shape), tuple(images.shape))
        self.assertTrue(torch.equal(resized, (images >= 0.5).to(images.dtype)))


if __name__ == "__main__":
    unittest.main()
