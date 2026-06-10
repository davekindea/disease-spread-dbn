"""Global configuration and constants for the SEIR DBN model."""

from dataclasses import dataclass

# Latent SEIR states
STATES = ("S", "E", "I", "R")
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)

# Observations: negative test, positive test, missing
OBS_NEG = 0
OBS_POS = 1
OBS_MISSING = -1


@dataclass
class ModelParams:
    """Epidemiological and observation parameters."""

    beta: float = 0.30   # transmission rate per infectious contact
    sigma: float = 0.20  # E -> I progression rate
    gamma: float = 0.10  # I -> R recovery rate
    sensitivity: float = 0.90   # P(pos | I)
    specificity: float = 0.95   # P(neg | not I)

    def as_array(self):
        import numpy as np
        return np.array([self.beta, self.sigma, self.gamma])

    @classmethod
    def from_array(cls, arr, sensitivity=0.90, specificity=0.95):
        return cls(
            beta=float(arr[0]),
            sigma=float(arr[1]),
            gamma=float(arr[2]),
            sensitivity=sensitivity,
            specificity=specificity,
        )


@dataclass
class SimConfig:
    """Simulation and experiment settings."""

    n_nodes: int = 20
    n_timesteps: int = 50
    network_kind: str = "ws"  # er, ws, ba
    test_probability: float = 0.70
    seed: int = 42
    patient_zero: int = 0
