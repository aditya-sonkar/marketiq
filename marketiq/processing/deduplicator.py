"""SHA-256 deduplication module for MarketIQ tweets."""

import hashlib
from typing import Optional, Set

from marketiq.models.tweet import Tweet
from marketiq.utils.logger import get_logger

logger = get_logger("processing.deduplicator")


class TweetDeduplicator:
    """Handles SHA-256 cryptographic deduplication for Tweet objects."""

    def __init__(self, existing_hashes: Optional[Set[str]] = None) -> None:
        """Initialize TweetDeduplicator with an optional set of known SHA-256 hashes.

        Args:
            existing_hashes (Optional[Set[str]]): Initial hash set to prevent duplication.
        """
        self.seen_hashes: Set[str] = set(existing_hashes) if existing_hashes else set()

    @staticmethod
    def compute_sha256(tweet: Tweet) -> str:
        """Compute a deterministic SHA-256 digest for a Tweet object.

        The digest is computed from lowercased username, ISO formatted timestamp, and normalized content text.

        Args:
            tweet (Tweet): Input Tweet object.

        Returns:
            str: 64-character hexadecimal SHA-256 hash string.
        """
        normalized_content = tweet.content.strip().lower()
        raw_key = f"{tweet.username.lower()}:{tweet.timestamp.isoformat()}:{normalized_content}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def is_duplicate(self, tweet: Tweet) -> bool:
        """Check if a tweet has already been seen based on its SHA-256 hash.

        Args:
            tweet (Tweet): Tweet object to check.

        Returns:
            bool: True if duplicate, False if unique.
        """
        tweet_hash = self.compute_sha256(tweet)
        return tweet_hash in self.seen_hashes

    def add(self, tweet: Tweet) -> None:
        """Register a tweet's SHA-256 hash into the seen set.

        Args:
            tweet (Tweet): Tweet object to register.
        """
        tweet_hash = self.compute_sha256(tweet)
        self.seen_hashes.add(tweet_hash)

    def deduplicate(self, tweets: list[Tweet]) -> list[Tweet]:
        """Filter out duplicate Tweet objects from a batch list.

        Args:
            tweets (list[Tweet]): Input list of Tweet domain objects.

        Returns:
            list[Tweet]: Deduplicated list of unique Tweet objects.
        """
        initial_count = len(tweets)
        logger.info(f"Starting SHA-256 deduplication on batch of {initial_count} tweets...")

        unique_tweets: list[Tweet] = []
        duplicate_count = 0

        for tweet in tweets:
            if self.is_duplicate(tweet):
                duplicate_count += 1
                continue

            self.add(tweet)
            unique_tweets.append(tweet)

        logger.info(
            f"Deduplication complete: Retained {len(unique_tweets)} unique tweets "
            f"(Removed {duplicate_count} duplicates)."
        )
        return unique_tweets
