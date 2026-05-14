"""Entry point for the text summarization evaluation pipeline."""

from data.dataset_loader import DataSetLoader


def main():
    """Load and preprocess the dataset for summarization evaluation."""
    loader = DataSetLoader()
    loader.load(debug_mode=False, batch_size=128)


if __name__ == "__main__":
    main()