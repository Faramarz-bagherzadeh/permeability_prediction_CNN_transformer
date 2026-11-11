import unittest

from early_stopping import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_patience_consecutive_non_improving_epochs(self) -> None:
        stopping = EarlyStopping(patience=2)

        self.assertFalse(stopping.step(1.0))
        self.assertFalse(stopping.step(1.1))
        self.assertTrue(stopping.step(1.2))

    def test_improvement_resets_non_improving_epoch_count(self) -> None:
        stopping = EarlyStopping(patience=2)

        self.assertFalse(stopping.step(1.0))
        self.assertFalse(stopping.step(1.1))
        self.assertFalse(stopping.step(0.9))
        self.assertFalse(stopping.step(0.95))
        self.assertTrue(stopping.step(0.96))

    def test_min_delta_controls_significant_improvement(self) -> None:
        stopping = EarlyStopping(patience=2, min_delta=0.01)

        self.assertFalse(stopping.step(1.0))
        self.assertFalse(stopping.step(0.995))
        self.assertTrue(stopping.step(0.99))

    def test_rejects_non_finite_validation_loss(self) -> None:
        stopping = EarlyStopping(patience=1)

        with self.assertRaises(ValueError):
            stopping.step(float("nan"))


if __name__ == "__main__":
    unittest.main()
