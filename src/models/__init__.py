"""Model interfaces and placeholders for sleep-stage experiments."""

from __future__ import annotations


class SleepStageModel:
    """Base model contract for sleep-stage experimentation.

    The placeholder keeps the API explicit while ensuring unsupported training
    operations do not fail silently.
    """

    def __init__(self, num_classes: int = 5, signal_channels: int = 1, taxonomy: str = "standard"):
        self.num_classes = num_classes
        self.signal_channels = signal_channels
        self.taxonomy = taxonomy

    def fit(self, *args, **kwargs):
        """Fit the model.

        This placeholder is intentionally unavailable until a concrete model is
        implemented in a later stage.
        """
        raise NotImplementedError(
            "Model fitting is not implemented yet; training is unavailable in this phase."
        )

    def predict(self, *args, **kwargs):
        """Predict sleep stages.

        This placeholder is intentionally unavailable until a concrete model is
        implemented in a later stage.
        """
        raise NotImplementedError(
            "Model prediction is not implemented yet; training is unavailable in this phase."
        )


class BaselineSleepStageModel(SleepStageModel):
    """Minimal baseline model contract for the repository.

    The class is intentionally lightweight and captures only the API surface that
    the experiment registry and stage definitions rely on.
    """

    def __init__(
        self,
        num_classes: int = 5,
        signal_channels: int = 1,
        taxonomy: str = "standard",
    ):
        super().__init__(
            num_classes=num_classes,
            signal_channels=signal_channels,
            taxonomy=taxonomy,
        )


__all__ = ["BaselineSleepStageModel", "SleepStageModel"]
