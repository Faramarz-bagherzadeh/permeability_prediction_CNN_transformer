import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed.")
class ResNet3DTests(unittest.TestCase):
    def test_resnet34_is_larger_and_returns_three_values(self) -> None:
        import torch

        from model_cnn_resnet18 import ResNet18CNN3D, ResNet34CNN3D

        resnet18 = ResNet18CNN3D()
        resnet34 = ResNet34CNN3D().eval()
        resnet18_parameters = sum(parameter.numel() for parameter in resnet18.parameters())
        resnet34_parameters = sum(parameter.numel() for parameter in resnet34.parameters())
        self.assertGreater(resnet34_parameters, resnet18_parameters)

        with torch.no_grad():
            predictions = resnet34(torch.randn(1, 1, 16, 16, 16))
        self.assertEqual(tuple(predictions.shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
