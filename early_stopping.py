"""Early stopping based on validation-loss progress."""

import math


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be a positive integer.")
        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, (int, float))
            or not math.isfinite(min_delta)
            or min_delta < 0
        ):
            raise ValueError("min_delta must be a finite non-negative number.")

        self.patience = patience
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, validation_loss: float) -> bool:
        if not math.isfinite(validation_loss):
            raise ValueError("validation_loss must be finite.")

        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
