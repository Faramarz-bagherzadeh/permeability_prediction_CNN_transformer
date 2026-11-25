import unittest

import numpy as np

from data_split import porosity_extrapolation_split, resolve_split_scenarios


class SplitScenarioSelectionTests(unittest.TestCase):
    def test_minus_one_selects_every_scenario(self) -> None:
        self.assertEqual(
            resolve_split_scenarios(-1),
            ("scenario-I", "scenario-II", "scenario-III"),
        )

    def test_named_scenario_selects_only_itself(self) -> None:
        self.assertEqual(resolve_split_scenarios("scenario-II"), ("scenario-II",))

    def test_rejects_boolean_instead_of_treating_it_as_integer(self) -> None:
        with self.assertRaises(ValueError):
            resolve_split_scenarios(True)


class PorosityExtrapolationSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.porosities = np.arange(100, dtype=np.float64)
        self.fractions = (0.7, 0.15, 0.15)

    def assert_complete_partition(
        self,
        splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self.assertEqual([len(indices) for indices in splits], [70, 15, 15])
        np.testing.assert_array_equal(
            np.sort(np.concatenate(splits)),
            np.arange(self.porosities.size),
        )

    def test_scenario_ii_holds_out_successively_higher_porosity(self) -> None:
        training, validation, test = porosity_extrapolation_split(
            self.porosities,
            self.fractions,
            "scenario-II",
        )

        self.assert_complete_partition((training, validation, test))
        self.assertLess(self.porosities[training].max(), self.porosities[validation].min())
        self.assertLess(self.porosities[validation].max(), self.porosities[test].min())

    def test_scenario_iii_holds_out_successively_lower_porosity(self) -> None:
        training, validation, test = porosity_extrapolation_split(
            self.porosities,
            self.fractions,
            "scenario-III",
        )

        self.assert_complete_partition((training, validation, test))
        self.assertLess(self.porosities[test].max(), self.porosities[validation].min())
        self.assertLess(self.porosities[validation].max(), self.porosities[training].min())


if __name__ == "__main__":
    unittest.main()
