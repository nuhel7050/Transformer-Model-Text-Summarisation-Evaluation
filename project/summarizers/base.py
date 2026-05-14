"""Base summarizer with shared hierarchical chunking and summarization logic."""

from abc import ABC, abstractmethod

import numpy as np


class BaseSummarizer(ABC):
    """Abstract base class for text summarizers.

    Provides shared functionality for hierarchical summarization
    of long documents via token-level chunking and multi-stage
    summarization. Subclasses must set ``summarizer``, ``tokenizer``,
    ``max_tokens``, and ``overlap`` before using inherited methods.
    """

    @abstractmethod
    def summarize(self, texts):
        """Generate summaries for a list of input texts.

        Args:
            texts: List of input document strings.

        Returns:
            List of summary strings.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared summarization helpers
    # ------------------------------------------------------------------

    def run_summarizer(self, text, max_length=200, min_length=30):
        """Run the HuggingFace summarization pipeline on a single text.

        Args:
            text: Input text string.
            max_length: Maximum number of tokens in the output summary.
            min_length: Minimum number of tokens in the output summary.

        Returns:
            Generated summary string.
        """
        return self.summarizer(
            "summarize:" + text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True,
        )[0]["summary_text"]

    def summarize_single(self, text):
        """Summarize a single document using hierarchical chunking if needed.

        Short documents are summarized directly. Long documents are split
        into overlapping chunks, each chunk is summarized, chunk summaries
        are grouped and re-summarized, and a final summary is produced.

        Args:
            text: Input document string.

        Returns:
            Summary string.
        """
        total_tokens = self.get_token_count(text)

        if total_tokens <= self.max_tokens:
            return self.run_summarizer(text, max_length=total_tokens)

        # Stage 1: Split into overlapping chunks and summarize each
        chunks = self.chunk_text(text)
        chunk_summaries = [self.run_summarizer(chunk) for chunk in chunks]

        # Stage 2: Group chunk summaries and re-summarize
        grouped_summaries = self.group_and_summarize(chunk_summaries)

        # Stage 3: Produce final summary
        if len(grouped_summaries) == 1:
            return grouped_summaries[0]

        final_input = " ".join(grouped_summaries)
        final_token_count = self.get_token_count(final_input)

        if final_token_count <= self.max_tokens:
            return self.run_summarizer(final_input, max_length=final_token_count)

        # Fallback: concatenate top-level summaries
        return final_input

    def group_and_summarize(self, summaries):
        """Group chunk summaries to fit within token limit and re-summarize.

        Args:
            summaries: List of summary strings from individual chunks.

        Returns:
            List of grouped summary strings.
        """
        grouped = []
        current_group = []
        current_token_count = 0

        for summary in summaries:
            tokens = self.get_token_count(summary)
            if current_token_count + tokens > self.max_tokens:
                grouped_text = " ".join(current_group)
                grouped.append(self.run_summarizer(grouped_text, max_length=tokens))
                current_group = [summary]
                current_token_count = tokens
            else:
                current_group.append(summary)
                current_token_count += tokens

        if current_group:
            grouped_text = " ".join(current_group)
            grouped.append(self.run_summarizer(grouped_text))

        return grouped

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def get_token_count(self, text):
        """Count the number of tokens in a text string.

        Args:
            text: Input text string.

        Returns:
            Integer token count.
        """
        return len(self.tokenizer(text, padding=False, truncation=False)["input_ids"])

    def chunk_text(self, text):
        """Split text into overlapping token-level chunks.

        Each chunk contains at most ``self.max_tokens`` tokens, with
        ``self.overlap`` tokens of overlap between consecutive chunks.

        Args:
            text: Input text string.

        Returns:
            List of chunk strings.
        """
        tokens = self.tokenizer(text, padding=False, truncation=False)["input_ids"]
        chunks = []
        step = self.max_tokens - self.overlap

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunk_str = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_str)
            if len(chunk_tokens) < self.max_tokens:
                break

        return chunks

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def get_embeddings(self, text):
        """Compute sentence embeddings, chunking if text exceeds token limit.

        Args:
            text: Input text string.

        Returns:
            Embedding vector as a list of floats.
        """
        total_tokens = self.get_token_count(text)
        if total_tokens <= self.max_tokens:
            return self._get_sentence_embedding(text)
        return self._get_chunked_embeddings(text).tolist()

    def _get_sentence_embedding(self, text):
        """Compute embedding for a single text that fits within token limit."""
        return self.model.encode(text).tolist()

    def _get_chunked_embeddings(self, text):
        """Compute mean embedding across text chunks."""
        chunks = self.chunk_text(text)
        embeddings = [np.array(self._get_sentence_embedding(c)) for c in chunks]
        return np.mean(embeddings, axis=0)
