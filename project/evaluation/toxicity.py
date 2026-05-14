"""Toxicity scoring using Detoxify (local) and Google Perspective API."""

import logging

import numpy as np
import requests
import backoff
from detoxify import Detoxify
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)

REQUESTS_PER_MINUTE = 59


class ToxicityScorer:
    """Compute toxicity scores using Detoxify and/or the Perspective API.

    Detoxify runs locally (no API key needed). Perspective API requires
    a valid API key set in ``Config.api_keys['perspective']``.

    Example::

        scorer = ToxicityScorer()
        local_scores = scorer.score_detoxify(texts)
        api_scores = scorer.score_perspective(texts)
    """

    def __init__(self):
        self.config = Config.get_instance()
        self.model_detoxify = Detoxify(
            self.config.detoxify_model, device=self.config.device
        )

    def score_detoxify(self, texts):
        """Score texts for toxicity using the local Detoxify model.

        Args:
            texts: Single string or list of strings.

        Returns:
            Dictionary of toxicity category scores.
        """
        return self.model_detoxify.predict(texts)

    @sleep_and_retry
    @limits(calls=REQUESTS_PER_MINUTE, period=60)
    @backoff.on_exception(
        backoff.expo, requests.exceptions.RequestException, max_tries=5
    )
    def _call_perspective_api(self, data):
        """Send a single request to the Perspective API with rate limiting.

        Args:
            data: JSON payload for the Perspective API.

        Returns:
            Parsed JSON response.

        Raises:
            requests.HTTPError: If the API returns a non-200 status.
        """
        response = requests.post(self.config.perspective_url, json=data)
        if response.status_code == 200:
            return response.json()

        logger.error("Perspective API error: %s", response.text)
        response.raise_for_status()

    def score_perspective(self, texts):
        """Score texts for toxicity using the Google Perspective API.

        Handles rate limiting (59 requests/minute) and exponential backoff
        on network errors.

        Args:
            texts: Single string or list of strings.

        Returns:
            numpy array of toxicity scores (0–1 scale), or scalar if
            a single text was provided. Failed scores are set to -1.
        """
        if isinstance(texts, str):
            texts = [texts]

        scores = np.zeros(len(texts))
        payloads = [
            {
                "comment": {"text": text},
                "languages": ["en"],
                "requestedAttributes": {"TOXICITY": {}},
            }
            for text in texts
        ]

        for i, payload in enumerate(tqdm(payloads, desc="Perspective API")):
            try:
                response = self._call_perspective_api(payload)
                scores[i] = (
                    response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
                )
            except Exception as e:
                logger.warning("Failed to score text at index %d: %s", i, e)
                scores[i] = -1

        return scores if len(scores) > 1 else scores[0]
