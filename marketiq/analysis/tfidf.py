"""TF-IDF vectorizer and market keyword extraction module."""

from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from marketiq.utils.logger import get_logger

logger = get_logger("analysis.tfidf")


class MarketTfidfVectorizer:
    """Wrapper around scikit-learn TfidfVectorizer tailored for market social text."""

    def __init__(
        self,
        max_features: int = 100,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> None:
        """Initialize MarketTfidfVectorizer.

        Args:
            max_features (int): Maximum vocabulary size to extract.
            ngram_range (tuple[int, int]): Lower and upper boundary for n-grams.
            min_df (int): Minimum document frequency for terms (default 1).
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
        )

    def fit_transform(self, texts: list[str]) -> csr_matrix:
        """Fit the vectorizer on input texts and return a sparse CSR TF-IDF matrix.

        Args:
            texts (list[str]): List of cleaned tweet content strings.

        Returns:
            csr_matrix: Sparse SciPy matrix of TF-IDF feature scores.
        """
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            logger.warning("Empty text list provided to TF-IDF vectorizer.")
            return csr_matrix((len(texts), 0))

        logger.info(f"Fitting TF-IDF Vectorizer on {len(valid_texts)} text documents...")
        return self.vectorizer.fit_transform(valid_texts)

    def transform(self, texts: list[str]) -> csr_matrix:
        """Transform new text documents into TF-IDF vector space using fitted vocabulary.

        Args:
            texts (list[str]): Input text strings.

        Returns:
            csr_matrix: Sparse SciPy matrix of TF-IDF feature scores.
        """
        if not hasattr(self.vectorizer, "vocabulary_"):
            raise RuntimeError("MarketTfidfVectorizer must be fitted before calling transform().")
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return csr_matrix((len(texts), len(self.vectorizer.get_feature_names_out())))
        return self.vectorizer.transform(valid_texts)

    def get_top_keywords(
        self,
        texts: list[str],
        top_n: int = 20,
        matrix: Optional[csr_matrix] = None,
    ) -> list[tuple[str, float]]:
        """Extract top N keywords ranked by mean TF-IDF score across all documents.

        Args:
            texts (list[str]): Cleaned tweet content texts.
            top_n (int): Number of top features to return.
            matrix (Optional[csr_matrix]): Precomputed TF-IDF sparse matrix to avoid refitting.

        Returns:
            list[tuple[str, float]]: List of (feature_name, mean_tfidf_score) pairs.
        """
        if matrix is None:
            if hasattr(self.vectorizer, "vocabulary_"):
                matrix = self.transform(texts)
            else:
                matrix = self.fit_transform(texts)

        if matrix.shape[1] == 0:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        # Compute mean across sparse matrix columns efficiently
        mean_scores = np.asarray(matrix.mean(axis=0)).ravel()

        # Sort features by mean score descending
        sorted_indices = np.argsort(mean_scores)[::-1][:top_n]
        top_keywords = [(feature_names[i], float(mean_scores[i])) for i in sorted_indices]

        logger.info(f"Extracted top {len(top_keywords)} keywords via TF-IDF.")
        return top_keywords
