"""ROUGE-1 through ROUGE-5 evaluation with weighted averaging."""

import logging

import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

logger = logging.getLogger(__name__)

ROUGE_METRICS = ["rouge1", "rouge2", "rouge3", "rouge4", "rouge5"]
ROUGE_WEIGHTS = [0.3, 0.25, 0.2, 0.15, 0.1]


class RougeScorer:
    """Wrapper around ``rouge_score`` for ROUGE-1, ROUGE-2, and ROUGE-L.

    Example::

        scorer = RougeScorer()
        results = scorer.score(summaries, references)
    """

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def score(self, summaries, references):
        """Compute ROUGE scores for paired summaries and references.

        Args:
            summaries: List of generated summary strings.
            references: List of reference summary strings.

        Returns:
            List of score dictionaries (one per document pair).
        """
        return [
            self.scorer.score(ref, summary)
            for ref, summary in zip(references, summaries)
        ]


def evaluate_rouge_scores(df, reference_col="document", summary_cols=None):
    """Compute ROUGE-1 to ROUGE-5 scores and a weighted average.

    Evaluates each summary column against the reference column, appends
    per-document scores to the DataFrame, and saves results to CSV.

    Args:
        df: DataFrame with reference and summary columns.
        reference_col: Name of the reference text column.
        summary_cols: List of summary column names. Defaults to
            ``['bart_summary', 't5_summary']``.

    Returns:
        DataFrame with added ROUGE score columns.
    """
    if summary_cols is None:
        summary_cols = ["bart_summary", "t5_summary"]

    scorer = rouge_scorer.RougeScorer(ROUGE_METRICS, use_stemmer=True)
    df = df.copy()

    # Initialize score columns
    for col in summary_cols:
        for metric in ROUGE_METRICS:
            df[f"{col}_{metric}"] = 0.0

    # Compute per-document scores
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating ROUGE"):
        reference = row[reference_col]
        for col in summary_cols:
            summary = row[col]
            try:
                scores = scorer.score(reference, summary)
                for metric in ROUGE_METRICS:
                    df.loc[idx, f"{col}_{metric}"] = scores[metric].fmeasure
            except Exception as e:
                logger.warning("ROUGE scoring failed at index %d for %s: %s", idx, col, e)
                for metric in ROUGE_METRICS:
                    df.loc[idx, f"{col}_{metric}"] = 0.0

    # Compute weighted average
    for col in summary_cols:
        weighted_avg = sum(
            df[f"{col}_{metric}"] * weight
            for metric, weight in zip(ROUGE_METRICS, ROUGE_WEIGHTS)
        )
        df[f"{col}_rouge_weighted_avg"] = weighted_avg

    output_path = "data/metric_rouge_values.csv"
    df.to_csv(output_path, index=False)
    logger.info("ROUGE results saved to %s", output_path)
    return df


def generate_report(pre_tox, post_tox, rouge_scores):
    """Print a formatted toxicity reduction and ROUGE report.

    Args:
        pre_tox: List of pre-summarization toxicity scores.
        post_tox: List of post-summarization toxicity scores.
        rouge_scores: List of ROUGE score dictionaries.
    """
    print("=== Toxicity Reduction ===")
    for i, (pre, post) in enumerate(zip(pre_tox, post_tox)):
        print(f"Doc {i}: Before = {pre:.3f}, After = {post:.3f}, Δ = {pre - post:.3f}")

    print("\n=== ROUGE Scores ===")
    for i, scores in enumerate(rouge_scores):
        r1 = scores["rouge1"].fmeasure
        r2 = scores["rouge2"].fmeasure
        rl = scores["rougeL"].fmeasure
        print(f"Doc {i}: R1={r1:.3f}, R2={r2:.3f}, RL={rl:.3f}")
