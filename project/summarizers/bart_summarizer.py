"""BART-Large-CNN summarizer with hierarchical chunking for long documents."""

from transformers import pipeline, AutoTokenizer
from datasets import Dataset

from .base import BaseSummarizer

# Model constants
MODEL_NAME = "facebook/bart-large-cnn"
MAX_TOKENS = 1019  # Reserve ~5 tokens for the 'summarize:' prompt prefix
CHUNK_OVERLAP = 50


class BARTSummarizer(BaseSummarizer):
    """Summarizer using the BART-Large-CNN model.

    For documents exceeding the 1024-token context window, applies
    hierarchical chunking with overlapping segments and multi-stage
    summarization (inherited from ``BaseSummarizer``).

    Args:
        device: Device string ('cpu' or 'cuda'). Note: the HuggingFace
            pipeline is initialized on device 0 (GPU) by default.
    """

    def __init__(self, device="cpu"):
        self.summarizer = pipeline(
            "summarization",
            model=MODEL_NAME,
            device=0,
            num_workers=16,
            batch_size=32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_tokens = MAX_TOKENS
        self.overlap = CHUNK_OVERLAP
        self.device = device

    def summarize(self, texts):
        """Generate summaries for a list of texts using batched processing.

        Args:
            texts: List of input document strings.

        Returns:
            List of summary strings.
        """
        dataset = Dataset.from_dict({"text": texts})
        results = dataset.map(self._summarize_batch, batched=True, batch_size=16)
        return results["bart_summary"]

    def _summarize_batch(self, batch):
        """Process a batch of texts through the summarization pipeline.

        Args:
            batch: Dictionary with 'text' key containing list of strings.

        Returns:
            Dictionary with 'bart_summary' key containing list of summaries.
        """
        summaries = [self.summarize_single(text) for text in batch["text"]]
        return {"bart_summary": summaries}
