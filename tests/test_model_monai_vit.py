import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("monai"),
    "PyTorch and MONAI are required.",
)
class MONAIViT3DTests(unittest.TestCase):
    def test_forward_returns_three_values_and_supports_backpropagation(self) -> None:
        import torch

        from model_t_MonaiViT import MONAIViT3DRegressor

        model = MONAIViT3DRegressor(image_size=(24, 32, 40))
        predictions = model(torch.randn(2, 1, 24, 32, 40))
        self.assertEqual(tuple(predictions.shape), (2, 3))

        predictions.sum().backward()
        self.assertIsNotNone(model.backbone.classification_head.weight.grad)

    def test_rejects_an_input_size_different_from_the_configured_size(self) -> None:
        import torch

        from model_t_MonaiViT import MONAIViT3DRegressor

        model = MONAIViT3DRegressor(image_size=32)
        with self.assertRaisesRegex(ValueError, "Input spatial dimensions"):
            model(torch.randn(1, 1, 64, 64, 64))


if __name__ == "__main__":
    unittest.main()
