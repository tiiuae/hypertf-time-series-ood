from abc import ABC, abstractmethod

import numpy as np


class BaseAugmentation(ABC):
    def __init__(self, p: float = 1.0):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability {p} must be between 0 and 1")
        self.p = p

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray: ...

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
