"""Text cleaning and normalization pipeline for MarketIQ tweets."""

import html
import re
import unicodedata
from dataclasses import replace

from marketiq.models.tweet import Tweet
from marketiq.utils.logger import get_logger

logger = get_logger("processing.cleaner")

# Regex patterns for text cleaning
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")
WHITESPACE_PATTERN = re.compile(r"\s+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
MENTION_PATTERN = re.compile(r"@(\w+)")


def clean_text(text: str) -> str:
    """Clean and normalize raw tweet body text.

    Steps:
        1. Decode HTML entities (&amp;, &lt;, &gt;, etc.).
        2. Strip URLs (http://, https://, www.).
        3. Apply Unicode NFKC normalization.
        4. Remove non-printable ASCII/Unicode control characters.
        5. Collapse multiple whitespace and newlines to a single space.
        6. Strip leading and trailing whitespace.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned and normalized text string.
    """
    if not text:
        return ""

    # 1. Unescape HTML entities
    cleaned = html.unescape(text)

    # 2. Remove URLs
    cleaned = URL_PATTERN.sub("", cleaned)

    # 3. Unicode NFKC normalization (combines accents, standardizes symbols)
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # 4. Remove control characters
    cleaned = CONTROL_CHAR_PATTERN.sub("", cleaned)

    # 5. Normalize whitespace
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)

    # 6. Strip leading/trailing whitespace
    return cleaned.strip()


class TweetCleaner:
    """Class for batch processing and cleaning Tweet domain objects."""

    @staticmethod
    def extract_hashtags(text: str) -> list[str]:
        """Extract clean, lowercase hashtags without leading '#' symbol.

        Args:
            text (str): Input text string.

        Returns:
            list[str]: Unique list of extracted lowercase hashtags.
        """
        if not text:
            return []
        matches = HASHTAG_PATTERN.findall(text)
        seen = set()
        result = []
        for tag in matches:
            lowered = tag.lower().strip()
            if lowered and lowered not in seen:
                seen.add(lowered)
                result.append(lowered)
        return result

    @staticmethod
    def extract_mentions(text: str) -> list[str]:
        """Extract clean, lowercase user mentions without leading '@' symbol.

        Args:
            text (str): Input text string.

        Returns:
            list[str]: Unique list of extracted lowercase user handles.
        """
        if not text:
            return []
        matches = MENTION_PATTERN.findall(text)
        seen = set()
        result = []
        for mention in matches:
            lowered = mention.lower().strip()
            if lowered and lowered not in seen:
                seen.add(lowered)
                result.append(lowered)
        return result

    def clean_tweet(self, tweet: Tweet) -> Tweet:
        """Clean an individual Tweet object's content and metadata.

        Args:
            tweet (Tweet): Input Tweet domain object.

        Returns:
            Tweet: New Tweet instance with cleaned content, hashtags, and mentions.
        """
        cleaned_content = clean_text(tweet.content)

        # Merge existing hashtags/mentions with re-extracted ones from cleaned text
        extracted_tags = set(self.extract_hashtags(cleaned_content))
        existing_tags = {h.lstrip("#").lower() for h in tweet.hashtags if h}
        merged_hashtags = sorted(list(extracted_tags.union(existing_tags)))

        extracted_mentions = set(self.extract_mentions(cleaned_content))
        existing_mentions = {m.lstrip("@").lower() for m in tweet.mentions if m}
        merged_mentions = sorted(list(extracted_mentions.union(existing_mentions)))

        return replace(
            tweet,
            content=cleaned_content,
            hashtags=merged_hashtags,
            mentions=merged_mentions,
        )

    def clean_tweets(self, tweets: list[Tweet], drop_empty: bool = True) -> list[Tweet]:
        """Batch clean a list of Tweet objects.

        Args:
            tweets (list[Tweet]): Raw scraped Tweet objects.
            drop_empty (bool): Whether to drop tweets whose content becomes empty after cleaning.

        Returns:
            list[Tweet]: Cleaned Tweet objects.
        """
        logger.info(f"Starting batch cleaning for {len(tweets)} tweets...")
        cleaned_list: list[Tweet] = []

        for tweet in tweets:
            try:
                cleaned_tweet = self.clean_tweet(tweet)
                if drop_empty and not cleaned_tweet.content:
                    continue
                cleaned_list.append(cleaned_tweet)
            except Exception as e:
                logger.warning(f"Error cleaning tweet from @{tweet.username}: {e}")

        logger.info(
            f"Batch cleaning complete. Retained {len(cleaned_list)}/{len(tweets)} valid tweets."
        )
        return cleaned_list
