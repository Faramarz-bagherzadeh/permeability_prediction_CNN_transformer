import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed.")
class ConViT3DTests(unittest.TestCase):
    def test_forward_returns_three_values_and_supports_backpropagation(self) -> None:
        import torch

        from model_t_ConVit import ConViT3D

        model = ConViT3D()
        predictions = model(torch.randn(2, 1, 32, 32, 32))
        self.assertEqual(tuple(predictions.shape), (2, 3))

        predictions.sum().backward()
        self.assertIsNotNone(model.patch_embed.weight.grad)


if __name__ == "__main__":
    unittest.main()
