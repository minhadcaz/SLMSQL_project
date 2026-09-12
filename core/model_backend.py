"""Common model contract; imports for the real backend are lazy."""
from abc import ABC, abstractmethod

from core.types import Candidate


class ModelBackend(ABC):
    name = "abstract"

    def start_request(self, seed: int) -> None:
        """Reset request randomness, not at each sampling round."""

    @abstractmethod
    def generate(self, prompt: str, n: int, parameters: dict,
                 deadline: float | None = None) -> list[Candidate]:
        """Return up to n candidates before the cooperative deadline."""
        raise NotImplementedError


def create_backend(config: dict) -> ModelBackend:
    if config["model"]["backend"] == "mock":
        from core.mock_model import MockModelBackend
        return MockModelBackend()
    from core.hf_model import HFModelBackend
    return HFModelBackend(config)
