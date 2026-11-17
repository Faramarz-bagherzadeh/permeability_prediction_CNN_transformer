import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed.")
class ModelCapacityTests(unittest.TestCase):
    def test_selected_models_have_comparable_parameter_counts(self) -> None:
        from model_cnn_densenet import DenseNetCNN3D
        from model_cnn_vanila import VanillaCNN3D
        from model_t_ConVit import ConViT3D
        from model_t_Swin import Swin3DRegressor

        models = (
            VanillaCNN3D(),
            DenseNetCNN3D(),
            ConViT3D(),
            Swin3DRegressor(image_size=32),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                trainable_parameters = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                self.assertGreaterEqual(trainable_parameters, 2_000_000)
                self.assertLessEqual(trainable_parameters, 3_000_000)


if __name__ == "__main__":
    unittest.main()
