from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Config:
    target: str = "SPY"
    horizon: int = 5
    train_size: int = 1260
    calibration_size: int = 756
    test_size: int = 126
    n_components: int = 3
    n_states: int = 3
    hmm_restarts: int = 3
    hmm_iterations: int = 200
    ridge_alpha: float = 100.0
    cost_bps: float = 5.0
    seed: int = 42

    def __post_init__(self):
        integer_fields = ("horizon", "train_size", "calibration_size", "test_size",
                          "n_components", "n_states", "hmm_restarts", "hmm_iterations")
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_states < 2:
            raise ValueError("n_states must be at least 2")
        if self.calibration_size < max(50, 10 * self.n_states):
            raise ValueError("calibration_size is too small to fit the HMM")
        if self.n_components >= self.calibration_size:
            raise ValueError("n_components must be smaller than calibration_size")
        if self.train_size - self.calibration_size - self.horizon - 1 < 50:
            raise ValueError("Need at least 50 mature return-training rows after calibration")
        if not 0 < self.ridge_alpha < float("inf"):
            raise ValueError("ridge_alpha must be finite and positive")
        if not 0 <= self.cost_bps < 1000:
            raise ValueError("cost_bps must be between 0 and 1000")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("target cannot be empty")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 - self.hmm_restarts:
            raise ValueError("seed must be a nonnegative 32-bit integer")

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as handle:
            return cls(**tomllib.load(handle))

    def to_dict(self):
        return asdict(self)

