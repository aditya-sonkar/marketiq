"""Tweet domain model representing scraped X/Twitter posts for MarketIQ."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Tweet:
    """Domain model representing a single scraped tweet.

    Attributes:
        username (str): Handle or identifier of the tweet author.
        timestamp (datetime): UTC creation timestamp of the tweet.
        content (str): Raw body text of the tweet.
        likes (int): Number of likes/favorites (>= 0).
        replies (int): Number of replies (>= 0).
        reposts (int): Number of retweets/reposts (>= 0).
        hashtags (list[str]): List of normalized hashtags (no '#', lowercase).
        mentions (list[str]): List of normalized user handles (no '@', lowercase).
    """

    username: str
    timestamp: datetime
    content: str
    likes: int = 0
    replies: int = 0
    reposts: int = 0
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate attribute bounds and sanitize inputs post-initialization."""
        if not isinstance(self.username, str):
            raise TypeError(f"username must be a string, got {type(self.username).__name__}")
        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"timestamp must be a datetime object, got {type(self.timestamp).__name__}")
        if not isinstance(self.content, str):
            raise TypeError(f"content must be a string, got {type(self.content).__name__}")

        # Sanitize whitespace
        self.username = self.username.strip()
        self.content = self.content.strip()

        # Ensure non-negative engagement metrics
        self.likes = max(0, int(self.likes))
        self.replies = max(0, int(self.replies))
        self.reposts = max(0, int(self.reposts))

        # Normalize hashtags (remove '#', convert to lowercase)
        if not isinstance(self.hashtags, list):
            self.hashtags = list(self.hashtags)
        self.hashtags = [
            h.strip().lstrip("#").lower()
            for h in self.hashtags
            if isinstance(h, str) and h.strip()
        ]

        # Normalize mentions (remove '@', convert to lowercase)
        if not isinstance(self.mentions, list):
            self.mentions = list(self.mentions)
        self.mentions = [
            m.strip().lstrip("@").lower()
            for m in self.mentions
            if isinstance(m, str) and m.strip()
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert the Tweet model instance into a dictionary payload.

        Returns:
            dict[str, Any]: Dictionary representation suitable for Pandas/PyArrow conversion.
        """
        return {
            "username": self.username,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "likes": self.likes,
            "replies": self.replies,
            "reposts": self.reposts,
            "hashtags": self.hashtags,
            "mentions": self.mentions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tweet":
        """Factory method to construct a Tweet model instance from a dictionary.

        Args:
            data (dict[str, Any]): Dictionary containing tweet fields.

        Returns:
            Tweet: Instantiated Tweet model object.

        Raises:
            ValueError: If timestamp format is invalid or missing.
        """
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError as e:
                raise ValueError(f"Invalid timestamp format '{ts}': {e}") from e
        elif not isinstance(ts, datetime):
            raise ValueError(f"timestamp must be a valid datetime instance or ISO string, got {type(ts).__name__}")

        return cls(
            username=str(data.get("username", "")),
            timestamp=ts,
            content=str(data.get("content", "")),
            likes=int(data.get("likes", 0)),
            replies=int(data.get("replies", 0)),
            reposts=int(data.get("reposts", 0)),
            hashtags=list(data.get("hashtags", [])),
            mentions=list(data.get("mentions", [])),
        )
