import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed.")
class TrainerSlurmArrayTests(unittest.TestCase):
    @patch("trainer.train_model")
    def test_model_index_trains_only_the_selected_model(self, train_model) -> None:
        from model_factory import ALL_MODELS, MODEL_REGISTRY
        from trainer import train_from_config

        selected_model = tuple(MODEL_REGISTRY)[2]
        train_from_config(
            {"model": ALL_MODELS, "split_scenario": "scenario-I"},
            model_index=2,
        )

        self.assertEqual(train_model.call_count, 1)
        self.assertEqual(train_model.call_args.args[0]["model"], selected_model)

    @patch("trainer.train_model")
    def test_model_index_requires_all_models_selection(self, train_model) -> None:
        from model_factory import MODEL_REGISTRY
        from trainer import train_from_config

        with self.assertRaisesRegex(ValueError, "only be used"):
            train_from_config(
                {"model": tuple(MODEL_REGISTRY)[0], "split_scenario": "scenario-I"},
                model_index=0,
            )
        train_model.assert_not_called()

    def test_model_index_must_be_in_range(self) -> None:
        from model_factory import ALL_MODELS, MODEL_REGISTRY
        from trainer import train_from_config

        with self.assertRaisesRegex(ValueError, "between 0 and"):
            train_from_config(
                {"model": ALL_MODELS, "split_scenario": "scenario-I"},
                model_index=len(MODEL_REGISTRY),
            )


if __name__ == "__main__":
    unittest.main()
