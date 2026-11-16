import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed.")
class RotationAugmentationTests(unittest.TestCase):
    def test_quarter_turns_swap_the_matching_target_axes(self) -> None:
        import torch

        from augmentation import rotate_volume_and_targets

        image = torch.arange(64).reshape(1, 4, 4, 4)
        target = torch.tensor([1.0, 2.0, 3.0])
        cases = (
            ((1, 0, 0), (1, 2), (1.0, 3.0, 2.0)),
            ((0, 1, 0), (1, 3), (3.0, 2.0, 1.0)),
            ((0, 0, 1), (2, 3), (2.0, 1.0, 3.0)),
        )
        for rotations, image_dims, expected_target in cases:
            with self.subTest(rotations=rotations):
                rotated_image, rotated_target = rotate_volume_and_targets(
                    image,
                    target,
                    rotations,
                )
                self.assertTrue(
                    torch.equal(
                        rotated_image,
                        torch.rot90(image, 1, dims=image_dims),
                    )
                )
                self.assertTrue(
                    torch.equal(rotated_target, torch.tensor(expected_target))
                )

    def test_random_batch_rotations_preserve_shapes_and_target_values(self) -> None:
        import torch

        from augmentation import random_rotate_batch

        images = torch.randn(2, 1, 4, 4, 4)
        targets = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        rotated_images, rotated_targets = random_rotate_batch(images, targets)

        self.assertEqual(tuple(rotated_images.shape), tuple(images.shape))
        self.assertEqual(tuple(rotated_targets.shape), tuple(targets.shape))
        for actual, expected in zip(rotated_targets, targets):
            self.assertEqual(sorted(actual.tolist()), sorted(expected.tolist()))

    def test_odd_quarter_turn_rejects_a_non_square_plane(self) -> None:
        import torch

        from augmentation import rotate_volume_and_targets

        image = torch.randn(1, 3, 4, 5)
        target = torch.tensor([1.0, 2.0, 3.0])

        with self.assertRaisesRegex(ValueError, "equal dimensions"):
            rotate_volume_and_targets(image, target, (1, 0, 0))

    def test_random_rotations_preserve_non_cubic_shapes(self) -> None:
        import torch

        from augmentation import random_rotate_batch

        images = torch.randn(2, 1, 3, 4, 5)
        targets = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        rotated_images, rotated_targets = random_rotate_batch(images, targets)

        self.assertEqual(tuple(rotated_images.shape), tuple(images.shape))
        self.assertTrue(torch.equal(rotated_targets, targets))


if __name__ == "__main__":
    unittest.main()
