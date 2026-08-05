"""Feature engineering module for calculating engagement and market sentiment indicators."""

import re

import pandas as pd

from marketiq.models.tweet import Tweet
from marketiq.utils.logger import get_logger

logger = get_logger("analysis.features")

BULLISH_WORDS = {
    "bull", "bullish", "buy", "buying", "call", "long", "breakout", "target",
    "upward", "rally", "profit", "gain", "green", "ce", "upside", "support",
    "ath", "high", "moon", "boom", "strong"
}

BEARISH_WORDS = {
    "bear", "bearish", "sell", "selling", "put", "short", "crash", "dump",
    "downward", "fall", "loss", "red", "pe", "downside", "resistance",
    "breakdown", "low", "panic", "drop", "stoploss", "weak"
}


class FeatureExtractor:
    """Extracts quantitative features and financial sentiment metrics from Tweet collections."""

    @staticmethod
    def calculate_engagement_score(likes: int, replies: int, reposts: int) -> float:
        """Calculate weighted engagement score based on user interactions.

        Weights explanation:
            - Likes (1.0x): Baseline user endorsement.
            - Reposts (1.5x): Amplifies reach and broadcast interest across user networks.
            - Replies (2.0x): Indicates stronger user engagement and active discussion.

        Args:
            likes (int): Like count.
            replies (int): Reply count.
            reposts (int): Repost/Retweet count.

        Returns:
            float: Composite engagement metric score.
        """
        return float(likes * 1.0 + reposts * 1.5 + replies * 2.0)

    @staticmethod
    def calculate_sentiment_score(text: str) -> tuple[float, int, int]:
        """Compute keyword-based sentiment polarity score bound between [-1.0, 1.0].

        Counts repeated occurrences of financial terms to accurately reflect sentiment strength.

        Args:
            text (str): Cleaned tweet text.

        Returns:
            tuple[float, int, int]: (polarity_score, bullish_count, bearish_count)
        """
        if not text:
            return 0.0, 0, 0

        tokens = re.findall(r"\b\w+\b", text.lower())
        bull_hits = sum(1 for word in tokens if word in BULLISH_WORDS)
        bear_hits = sum(1 for word in tokens if word in BEARISH_WORDS)

        total_hits = bull_hits + bear_hits
        if total_hits == 0:
            return 0.0, 0, 0

        score = (bull_hits - bear_hits) / total_hits
        return score, bull_hits, bear_hits

    def extract_features(self, tweets: list[Tweet]) -> pd.DataFrame:
        """Transform a list of Tweet objects into an enriched Pandas DataFrame.

        Engineered Features Added:
            - engagement_score
            - tweet_length
            - word_count
            - hashtag_count
            - mention_count
            - sentiment_score
            - bullish_keyword_count
            - bearish_keyword_count

        Args:
            tweets (list[Tweet]): List of Tweet domain objects.

        Returns:
            pd.DataFrame: Enriched feature DataFrame.
        """
        logger.info(f"Extracting engineered features for {len(tweets)} tweets...")
        if not tweets:
            return pd.DataFrame()

        rows = []
        for t in tweets:
            content_text = t.content or ""
            engagement = self.calculate_engagement_score(t.likes, t.replies, t.reposts)
            sentiment, bull_count, bear_count = self.calculate_sentiment_score(content_text)

            rows.append(
                {
                    "username": t.username,
                    "timestamp": t.timestamp,
                    "content": content_text,
                    "likes": t.likes,
                    "replies": t.replies,
                    "reposts": t.reposts,
                    "hashtags": t.hashtags,
                    "mentions": t.mentions,
                    "engagement_score": engagement,
                    "tweet_length": len(content_text),
                    "word_count": len(content_text.split()),
                    "hashtag_count": len(t.hashtags),
                    "mention_count": len(t.mentions),
                    "sentiment_score": sentiment,
                    "bullish_keyword_count": bull_count,
                    "bearish_keyword_count": bear_count,
                }
            )

        df = pd.DataFrame(rows)
        logger.info("Feature extraction complete. DataFrame dimensions: %s", df.shape)
        return df
