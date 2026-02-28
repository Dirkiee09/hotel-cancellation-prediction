"""Serving and inference package."""

from .inference import ModelArtifacts, load_artifacts, predict_proba

__all__ = ["ModelArtifacts", "load_artifacts", "predict_proba"]
