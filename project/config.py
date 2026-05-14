"""Application configuration using the Singleton pattern."""

import os


class Config:
    """Centralized configuration for models, API keys, and dataset selection.

    Uses the singleton pattern — access via ``Config.get_instance()``.

    Attributes:
        model: Active summarization model ('bart' or 't5').
        device: Compute device ('cpu' or 'cuda').
        detoxify_model: Detoxify model variant ('original', 'unbiased', etc.).
        dataset: Dataset identifier ('reddit_tldr' or 'multi_news').
        api_keys: Dictionary of external API keys.
        perspective_url: Formatted Perspective API endpoint URL.
    """

    _instance = None

    def __init__(self):
        self.model = "bart"
        self.device = "cpu"
        self.detoxify_model = "original"
        self.dataset = "reddit_tldr"

        self.api_keys = {
            "perspective": os.environ.get("PERSPECTIVE_API_KEY", ""),
        }

        self.perspective_url = (
            "https://commentanalyzer.googleapis.com/v1alpha1/"
            f"comments:analyze?key={self.api_keys['perspective']}"
        )

    @classmethod
    def get_instance(cls):
        """Return the singleton Config instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance