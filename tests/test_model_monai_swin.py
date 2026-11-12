import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torch")
    and importlib.util.find_spec("monai")
    and importlib.util.find_spec("einops"),
    "PyTorch, MONAI, and einops are required.",
)
class MONAISwin3DTests(unittest.TestCase):
    def test_forward_returns_three_values_and_supports_backpropagation(self) -> None:
        import torch

        from model_t_Swin import MONAISwin3DRegressor

        model = MONAISwin3DRegressor(image_size=32)
        predictions = model(torch.randn(2, 1, 32, 32, 32))
        self.assertEqual(tuple(predictions.shape), (2, 3))

        predictions.sum().backward()
        self.assertIsNotNone(model.regressor.weight.grad)

    def test_rejects_an_input_size_different_from_the_configured_size(self) -> None:
        import torch

        from model_t_Swin import MONAISwin3DRegressor

        model = MONAISwin3DRegressor(image_size=32)
        with self.assertRaisesRegex(ValueError, "Input spatial dimensions"):
            model(torch.randn(1, 1, 64, 64, 64))


if __name__ == "__main__":
    unittest.main()
