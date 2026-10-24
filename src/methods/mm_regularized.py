"""Backward-compatible names for the unified ordinary MM implementation."""

from .dlight_sb_mm import (
    DLightSBMM,
    _exponential_log_penalty,
    _exponential_log_update,
)


class DLightSBMMRegularized(DLightSBMM):
    """Compatibility alias for old configs and checkpoints.

    New experiments should instantiate :class:`DLightSBMM` and set
    ``regularization_lambda_alpha`` and ``regularization_lambda_cores`` directly.
    """

    def __init__(
        self,
        *args,
        regularization_lambda_alpha=0.01,
        regularization_lambda_cores=0.5,
        clip_updates=False,
        **kwargs,
    ):
        super().__init__(
            *args,
            regularization_lambda_alpha=regularization_lambda_alpha,
            regularization_lambda_cores=regularization_lambda_cores,
            clip_updates=clip_updates,
            **kwargs,
        )


__all__ = [
    "DLightSBMMRegularized",
    "_exponential_log_penalty",
    "_exponential_log_update",
]
