"""Factory for creating and managing summarizer instances."""

from config import Config
from .bart_summarizer import BARTSummarizer
from .t5_summarizer import T5Summarizer

# Registry of available summarizer classes
_SUMMARIZERS = {
    "bart": BARTSummarizer,
    "t5": T5Summarizer,
}


class SummarizerFactory:
    """Lazily initializes and caches summarizer instances.

    Uses the singleton ``Config`` to determine device placement.
    Summarizers are instantiated on first use and reused thereafter.

    Example::

        factory = SummarizerFactory()
        summaries = factory.summarize("bart", texts)
    """

    def __init__(self):
        self.config = Config.get_instance()
        self._instances = {}

    def _get_summarizer(self, name):
        """Return a cached summarizer instance, creating it if needed.

        Args:
            name: Model identifier ('bart' or 't5').

        Returns:
            Summarizer instance.

        Raises:
            ValueError: If *name* is not a registered summarizer.
        """
        if name not in _SUMMARIZERS:
            raise ValueError(
                f"Unknown summarizer '{name}'. "
                f"Available: {list(_SUMMARIZERS.keys())}"
            )
        if name not in self._instances:
            self._instances[name] = _SUMMARIZERS[name](device=self.config.device)
        return self._instances[name]

    def summarize(self, name, texts):
        """Generate summaries using the specified model.

        Args:
            name: Model identifier ('bart' or 't5').
            texts: List of input document strings.

        Returns:
            List of summary strings.
        """
        return self._get_summarizer(name).summarize(texts)
