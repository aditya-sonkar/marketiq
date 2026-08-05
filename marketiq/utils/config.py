"""Configuration management module for MarketIQ."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from marketiq.utils.logger import get_logger

logger = get_logger("config")

# Load environment variables from .env if present
load_dotenv()


@dataclass(slots=True)
class Settings:
    """Centralized application configuration settings.

    Attributes:
        search_hashtags (list[str]): Hashtags to scrape from X/Twitter.
        max_tweets (int): Target maximum number of tweets to collect.
        headless (bool): Whether to run browser automation in headless mode.
        scroll_delay (float): Base delay in seconds between scroll actions.
        max_scrolls (int): Upper bound on the number of scroll iterations.
        page_load_timeout (int): Webdriver page load timeout in seconds.
        max_workers (int): Maximum concurrent scraper worker threads.
        max_retries (int): Maximum page load retry attempts for search navigation.
        min_scroll_pixels (int): Minimum pixel distance per scroll step.
        max_scroll_pixels (int): Maximum pixel distance per scroll step.
        reading_pause_probability (float): Probability of human reading pause.
        reading_pause_min (float): Minimum human reading pause in seconds.
        reading_pause_max (float): Maximum human reading pause in seconds.
        scroll_jitter_min (float): Minimum scroll delay jitter in seconds.
        scroll_jitter_max (float): Maximum scroll delay jitter in seconds.
        output_dir (Path): Base output data directory.
        raw_data_dir (Path): Storage directory for raw scraped payloads.
        processed_data_dir (Path): Storage directory for cleaned Parquet files.
        log_level (str): Logging verbosity level (DEBUG, INFO, WARNING, ERROR).
        user_data_dir (Optional[str]): Path to Chrome user data directory for persistent profile.
        profile_dir (Optional[str]): Chrome profile directory name (e.g., 'Default' or 'MarketIQ').
    """

    search_hashtags: list[str] = field(
        default_factory=lambda: ["#nifty50", "#sensex", "#banknifty", "#intraday"]
    )
    max_tweets: int = 2000
    headless: bool = False
    scroll_delay: float = 3.0
    max_scrolls: int = 1000
    page_load_timeout: int = 30
    max_workers: int = 4
    max_retries: int = 3
    no_progress_limit: int = 8
    auth_timeout: int = 1800
    screenshot_retention_days: int = 7
    retry_base_delay: float = 2.0
    retry_max_delay: float = 30.0
    min_scroll_pixels: int = 800
    max_scroll_pixels: int = 1400
    reading_pause_probability: float = 0.15
    reading_pause_min: float = 6.0
    reading_pause_max: float = 12.0
    scroll_jitter_min: float = 0.5
    scroll_jitter_max: float = 2.5
    output_dir: Path = field(default_factory=lambda: Path("data"))
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    log_level: str = "INFO"
    user_data_dir: Optional[str] = field(default_factory=lambda: os.getenv("CHROME_USER_DATA_DIR"))
    profile_dir: Optional[str] = field(default_factory=lambda: os.getenv("CHROME_PROFILE_DIR", "Default"))

    def __post_init__(self) -> None:
        """Validate bounds and ensure target directories exist."""
        # Convert path attributes to Path objects
        self.output_dir = Path(self.output_dir)
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)

        # Validate numeric bounds
        if self.max_tweets <= 0:
            raise ValueError(f"max_tweets must be > 0, got {self.max_tweets}")
        if self.scroll_delay <= 0:
            raise ValueError(f"scroll_delay must be > 0, got {self.scroll_delay}")
        if self.max_scrolls <= 0:
            raise ValueError(f"max_scrolls must be > 0, got {self.max_scrolls}")
        if self.page_load_timeout <= 0:
            raise ValueError(f"page_load_timeout must be > 0, got {self.page_load_timeout}")
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be > 0, got {self.max_workers}")
        if self.max_retries <= 0:
            raise ValueError(f"max_retries must be > 0, got {self.max_retries}")
        if self.no_progress_limit <= 0:
            raise ValueError(f"no_progress_limit must be > 0, got {self.no_progress_limit}")
        if self.min_scroll_pixels <= 0:
            raise ValueError(f"min_scroll_pixels must be > 0, got {self.min_scroll_pixels}")
        if self.max_scroll_pixels < self.min_scroll_pixels:
            raise ValueError(
                f"max_scroll_pixels ({self.max_scroll_pixels}) must be >= min_scroll_pixels ({self.min_scroll_pixels})"
            )
        if not (0.0 <= self.reading_pause_probability <= 1.0):
            raise ValueError(f"reading_pause_probability must be between 0.0 and 1.0, got {self.reading_pause_probability}")
        if self.reading_pause_min < 0:
            raise ValueError(f"reading_pause_min must be >= 0, got {self.reading_pause_min}")
        if self.reading_pause_max < self.reading_pause_min:
            raise ValueError(f"reading_pause_max ({self.reading_pause_max}) must be >= reading_pause_min ({self.reading_pause_min})")
        if self.scroll_jitter_min < 0:
            raise ValueError(f"scroll_jitter_min must be >= 0, got {self.scroll_jitter_min}")
        if self.scroll_jitter_max < self.scroll_jitter_min:
            raise ValueError(f"scroll_jitter_max ({self.scroll_jitter_max}) must be >= scroll_jitter_min ({self.scroll_jitter_min})")

        # Validate and standardize log level string
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        self.log_level = self.log_level.upper()
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}")

        # Bound max_workers based on available system core count (I/O bound browsers)
        cpu_count = os.cpu_count() or 4
        cpu_bound = max(cpu_count * 4, 4)
        if self.max_workers > cpu_bound:
            logger.warning("Capping max_workers from %d to %d based on system limits.", self.max_workers, cpu_bound)
            self.max_workers = cpu_bound

        # Validate and normalize search hashtags
        if not isinstance(self.search_hashtags, list):
            self.search_hashtags = list(self.search_hashtags)
        self.search_hashtags = [
            tag.strip() if tag.strip().startswith("#") else f"#{tag.strip()}"
            for tag in self.search_hashtags
            if tag and isinstance(tag, str) and tag.strip()
        ]
        if not self.search_hashtags:
            raise ValueError("At least one valid search hashtag must be provided.")

        # Ensure required data directories exist
        self._ensure_directories()

        logger.info(
            "Settings initialized: max_tweets=%d, max_workers=%d, max_retries=%d, headless=%s",
            self.max_tweets,
            self.max_workers,
            self.max_retries,
            self.headless,
        )

    def _ensure_directories(self) -> None:
        """Create necessary directories on disk if they do not exist."""
        for path in (self.output_dir, self.raw_data_dir, self.processed_data_dir, Path("logs")):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error("Failed to create directory %s: %s", path, e)
                raise


def load_settings_from_env() -> Settings:
    """Factory function to build Settings from environment variables with defaults.

    Returns:
        Settings: Instantiated and validated Settings object.
    """
    hashtags_raw = os.getenv("SEARCH_HASHTAGS")
    hashtags = (
        [h.strip() for h in hashtags_raw.split(",") if h.strip()]
        if hashtags_raw
        else ["#nifty50", "#sensex", "#banknifty", "#intraday"]
    )

    headless_raw = os.getenv("HEADLESS", "false").strip().lower()
    headless = headless_raw in ("true", "1", "yes")

    return Settings(
        search_hashtags=hashtags,
        max_tweets=int(os.getenv("MAX_TWEETS", "2000")),
        headless=headless,
        scroll_delay=float(os.getenv("SCROLL_DELAY", "3.0")),
        max_scrolls=int(os.getenv("MAX_SCROLLS", "1000")),
        page_load_timeout=int(os.getenv("PAGE_LOAD_TIMEOUT", "30")),
        max_workers=int(os.getenv("MAX_WORKERS", "4")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        no_progress_limit=int(os.getenv("NO_PROGRESS_LIMIT", "8")),
        auth_timeout=int(os.getenv("AUTH_TIMEOUT", "1800")),
        screenshot_retention_days=int(os.getenv("SCREENSHOT_RETENTION_DAYS", "7")),
        retry_base_delay=float(os.getenv("RETRY_BASE_DELAY", "2.0")),
        retry_max_delay=float(os.getenv("RETRY_MAX_DELAY", "30.0")),
        min_scroll_pixels=int(os.getenv("MIN_SCROLL_PIXELS", "800")),
        max_scroll_pixels=int(os.getenv("MAX_SCROLL_PIXELS", "1400")),
        reading_pause_probability=float(os.getenv("READING_PAUSE_PROBABILITY", "0.15")),
        reading_pause_min=float(os.getenv("READING_PAUSE_MIN", "6.0")),
        reading_pause_max=float(os.getenv("READING_PAUSE_MAX", "12.0")),
        scroll_jitter_min=float(os.getenv("SCROLL_JITTER_MIN", "0.5")),
        scroll_jitter_max=float(os.getenv("SCROLL_JITTER_MAX", "2.5")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "data")),
        raw_data_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
        processed_data_dir=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        user_data_dir=os.getenv("CHROME_USER_DATA_DIR"),
        profile_dir=os.getenv("CHROME_PROFILE_DIR", "Default"),
    )


# Module-level singleton instance for default convenience
settings = load_settings_from_env()
