"""Twitter / X market intelligence scraper implementation with concurrent worker support."""

import hashlib
import random
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Optional, Set, Tuple

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)

from marketiq.models.tweet import Tweet
from marketiq.scraper.browser import BrowserManager
from marketiq.utils.config import Settings, settings as default_settings
from marketiq.utils.logger import get_logger

logger = get_logger("scraper.twitter")

# Precompiled regex for parsing metric numbers
ENGAGEMENT_RE = re.compile(r"([\d,.]+\s*[kKmMbB]?)")


def parse_engagement_number(text: Optional[str]) -> int:
    """Safely parse engagement strings (e.g., '1.2K', '3.4M', '45', '1,200') into integers.

    Args:
        text (Optional[str]): Metric text or aria-label snippet.

    Returns:
        int: Parsed count value (>= 0).
    """
    if not text:
        return 0

    match = ENGAGEMENT_RE.search(text.replace(",", ""))
    if not match:
        return 0

    token = match.group(1).strip().upper()
    try:
        if token.endswith("K"):
            return int(float(token[:-1]) * 1_000)
        elif token.endswith("M"):
            return int(float(token[:-1]) * 1_000_000)
        elif token.endswith("B"):
            return int(float(token[:-1]) * 1_000_000_000)
        return int(float(token))
    except (ValueError, TypeError):
        return 0


def cleanup_old_screenshots(logs_dir: Path = Path("logs"), days: int = 7) -> int:
    """Remove debug PNG screenshots older than specified days.

    Args:
        logs_dir (Path): Output logs directory.
        days (int): Maximum age threshold in days.

    Returns:
        int: Number of deleted screenshot files.
    """
    if not logs_dir.exists():
        return 0

    cutoff = time.time() - (days * 86400)
    deleted_count = 0
    for p in logs_dir.glob("*.png"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                deleted_count += 1
        except Exception as e:
            logger.debug("Failed deleting screenshot file %s: %s", p, e)
    if deleted_count > 0:
        logger.info("Cleaned up %d debug screenshot files older than %d days.", deleted_count, days)
    return deleted_count


class TwitterScraper:
    """Production-grade scraper for retrieving market tweets from X/Twitter with concurrent capabilities.

    Attributes:
        settings (Settings): Configured setting bounds and timeouts.
        browser_manager (BrowserManager): Driver manager instance.
        seen_tweet_hashes (Set[str]): SHA-256 hashes of seen tweets for deduplication.
        hash_lock (Lock): Thread safety lock for synchronizing seen_tweet_hashes updates.
        duplicate_count (int): Counter tracking number of duplicate tweets skipped.
    """

    SEARCH_BASE_URL = "https://x.com/search?q={query}&f=live"
    TWEET_CONTAINER_LOCATOR = (By.CSS_SELECTOR, 'article[data-testid="tweet"]')
    USER_NAME_LOCATOR = (By.CSS_SELECTOR, '[data-testid="User-Name"]')
    TWEET_TEXT_LOCATOR = (By.CSS_SELECTOR, '[data-testid="tweetText"]')
    TIMESTAMP_LOCATOR = (By.CSS_SELECTOR, "time")
    STATUS_LINK_LOCATOR = (By.CSS_SELECTOR, 'a[href*="/status/"]')
    REPLY_LOCATOR = (By.CSS_SELECTOR, '[data-testid="reply"]')
    RETWEET_LOCATOR = (By.CSS_SELECTOR, '[data-testid="retweet"]')
    LIKE_LOCATOR = (By.CSS_SELECTOR, '[data-testid="like"]')
    LOGIN_WALL_LOCATOR = (By.CSS_SELECTOR, '[data-testid="loginButton"], a[href="/login"]')

    def __init__(
        self,
        settings: Optional[Settings] = None,
        browser_manager: Optional[BrowserManager] = None,
    ) -> None:
        """Initialize TwitterScraper dependencies."""
        self.settings = settings or default_settings
        self.browser_manager = browser_manager or BrowserManager(self.settings)
        self.seen_tweet_hashes: Set[str] = set()
        self.hash_lock = Lock()
        self.duplicate_count = 0

    def check_for_login_wall(self, driver: WebDriver) -> bool:
        """Detect whether X/Twitter is displaying a login wall, rate limit, or error banner overlay.

        Args:
            driver (WebDriver): Active browser driver.

        Returns:
            bool: True if login wall or rate limit is present, False otherwise.
        """
        try:
            elements = driver.find_elements(*self.LOGIN_WALL_LOCATOR)
            if elements or "/login" in driver.current_url:
                logger.warning("X/Twitter login wall detected.")
                return True

            body_text = (driver.execute_script("return (document.body ? document.body.innerText : '').toLowerCase();") or "")
            if (
                "rate limit exceeded" in body_text
                or "something went wrong" in body_text
                or "try again later" in body_text
            ):
                logger.warning("X/Twitter rate limit or error banner detected.")
                return True
        except Exception as e:
            logger.debug("Failed checking login wall locator: %s", e)
        return False

    def save_debug_screenshot(self, driver: WebDriver, prefix: str, hashtag: str) -> None:
        """Save a timestamped screenshot of browser window for debugging.

        Args:
            driver (WebDriver): Driver instance to capture screenshot from.
            prefix (str): Label prefix for screenshot filename.
            hashtag (str): Target hashtag query string.
        """
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_tag = hashtag.replace("#", "").replace(" ", "_")
            filename = f"{prefix}_{clean_tag}_{timestamp_str}.png"
            screenshot_path = Path("logs") / filename
            driver.save_screenshot(str(screenshot_path))
            logger.info("Saved debug screenshot to: %s", screenshot_path)
        except Exception as e:
            logger.debug("Failed to capture debug screenshot: %s", e)

    def search_hashtag(self, driver: WebDriver, hashtag: str, max_retries: Optional[int] = None) -> bool:
        """Navigate to search query with automatic retry logic and login wall handling.

        Args:
            driver (WebDriver): Active Chrome driver instance.
            hashtag (str): Target hashtag (e.g., '#nifty50').
            max_retries (Optional[int]): Number of page load attempts. Defaults to self.settings.max_retries.

        Returns:
            bool: True if page loaded and tweets rendered, False otherwise.
        """
        retries = max_retries or self.settings.max_retries
        encoded_query = urllib.parse.quote(hashtag)
        url = self.SEARCH_BASE_URL.format(query=encoded_query)

        for attempt in range(1, retries + 1):
            logger.info("[%s] Navigating to search (Attempt %d/%d): %s", hashtag, attempt, retries, url)
            try:
                driver.get(url)

                if self.save_debug_screenshot(driver, "login_wall", hashtag):
                    logger.warning("[%s] Login wall hit on attempt %d/%d. Retrying in 5s...", hashtag, attempt, retries)
                    time.sleep(random.uniform(5.0, 10.0))
                    continue

                wait = WebDriverWait(driver, self.settings.page_load_timeout)
                wait.until(EC.visibility_of_any_elements_located(self.TWEET_CONTAINER_LOCATOR))
                logger.info("[%s] Search page loaded successfully.", hashtag)
                return True
            except TimeoutException:
                logger.warning("[%s] Page load timeout on attempt %d/%d.", hashtag, attempt, retries)
                self.save_debug_screenshot(driver, "timeout", hashtag)
            except WebDriverException as e:
                logger.warning("[%s] WebDriver error on attempt %d/%d: %s", hashtag, attempt, retries, e)
                self.save_debug_screenshot(driver, "error", hashtag)

            time.sleep(random.uniform(2.5, 5.0))

        logger.error("[%s] Failed to load search page after %d attempts.", hashtag, retries)
        return False

    def extract_single_tweet(self, element: WebElement, cutoff_time: datetime) -> Optional[Tweet]:
        """Parse raw DOM element into a Tweet model with SHA-256 deduplication and 24-hour filtering.

        Args:
            element (WebElement): Tweet article DOM element.
            cutoff_time (datetime): Minimum allowed UTC creation timestamp.

        Returns:
            Optional[Tweet]: Parsed Tweet object, or None if skipped/invalid.
        """
        try:
            # 1. Username
            username = "unknown"
            try:
                user_elem = element.find_element(*self.USER_NAME_LOCATOR)
                match = re.search(r"@([A-Za-z0-9_]{1,15})", user_elem.text.strip())
                if match:
                    username = match.group(1)
            except Exception as e:
                logger.debug("Could not extract username element: %s", e)

            # 2. Timestamp & 24-Hour Filter
            timestamp = datetime.now(timezone.utc)
            try:
                time_elem = element.find_element(*self.TIMESTAMP_LOCATOR)
                datetime_str = time_elem.get_attribute("datetime")
                if datetime_str:
                    timestamp = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
            except Exception as e:
                logger.debug("Could not extract timestamp element: %s", e)

            if timestamp < cutoff_time:
                logger.debug("Skipping tweet from @%s older than 24 hours (%s).", username, timestamp)
                return None

            # 3. Content Text
            content = ""
            try:
                content_elem = element.find_element(*self.TWEET_TEXT_LOCATOR)
                content = content_elem.text.strip()
            except Exception as e:
                logger.debug("Could not extract content element: %s", e)

            if not content and username == "unknown":
                return None

            # 4. Immutable Tweet ID Extraction & SHA-256 Deduplication
            tweet_id = None
            try:
                link_elem = element.find_element(*self.STATUS_LINK_LOCATOR)
                href = link_elem.get_attribute("href") or ""
                match = re.search(r"/status/(\d+)", href)
                if match:
                    tweet_id = match.group(1)
            except Exception as e:
                logger.debug("Could not extract tweet status link: %s", e)

            raw_hash_input = tweet_id if tweet_id else f"{username.lower()}:{timestamp.isoformat()}:{content}"
            sha256_hash = hashlib.sha256(raw_hash_input.encode("utf-8")).hexdigest()

            with self.hash_lock:
                if sha256_hash in self.seen_tweet_hashes:
                    self.duplicate_count += 1
                    logger.debug("Duplicate tweet skipped: @%s (id=%s)", username, tweet_id)
                    return None
                self.seen_tweet_hashes.add(sha256_hash)

            # 5. Metrics
            replies = self._extract_metric_count(element, self.REPLY_LOCATOR)
            reposts = self._extract_metric_count(element, self.RETWEET_LOCATOR)
            likes = self._extract_metric_count(element, self.LIKE_LOCATOR)

            # 6. Hashtags & Mentions
            hashtags = re.findall(r"#(\w+)", content)
            mentions = re.findall(r"@(\w+)", content)

            return Tweet(
                username=username,
                timestamp=timestamp,
                content=content,
                likes=likes,
                replies=replies,
                reposts=reposts,
                hashtags=hashtags,
                mentions=mentions,
            )

        except StaleElementReferenceException:
            return None
        except Exception as e:
            logger.debug("Error parsing tweet element: %s", e)
            return None

    def _extract_metric_count(self, parent: WebElement, locator: Tuple[str, str]) -> int:
        """Extract engagement count from element text or aria-label attribute."""
        try:
            elem = parent.find_element(*locator)
            text = elem.text.strip()
            if text:
                return parse_engagement_number(text)
            aria_label = elem.get_attribute("aria-label")
            return parse_engagement_number(aria_label)
        except Exception as e:
            logger.debug("Failed extracting engagement metric: %s", e)
            return 0

    def extract_visible_tweets(self, driver: WebDriver, cutoff_time: datetime) -> list[Tweet]:
        """Scrape all currently visible tweet elements from the active browser tab.

        Args:
            driver (WebDriver): Active browser driver.
            cutoff_time (datetime): Minimum allowed UTC creation timestamp.

        Returns:
            list[Tweet]: Newly extracted Tweet instances.
        """
        extracted: list[Tweet] = []

        try:
            articles = driver.find_elements(*self.TWEET_CONTAINER_LOCATOR)
            for article in articles:
                tweet = self.extract_single_tweet(article, cutoff_time)
                if tweet:
                    extracted.append(tweet)
        except Exception as e:
            logger.warning("Error querying visible DOM tweets: %s", e)

        return extracted

    def scroll(self, driver: WebDriver) -> Tuple[bool, int]:
        """Execute human-like randomized smooth scroll action and return (success, current_scroll_y).

        Args:
            driver (WebDriver): Active browser driver.

        Returns:
            Tuple[bool, int]: (Success flag, new document Y scroll offset).
        """
        try:
            # Human-like random scroll distance with smooth behavior using configured pixel bounds
            pixels = random.randint(self.settings.min_scroll_pixels, self.settings.max_scroll_pixels)
            driver.execute_script("window.scrollBy({top: arguments[0], behavior: 'smooth'});", pixels)

            # Short settling pause for smooth scroll JS animation
            time.sleep(0.2)

            # Occasional human reading pause
            if random.random() < self.settings.reading_pause_probability:
                pause = random.uniform(self.settings.reading_pause_min, self.settings.reading_pause_max)
                time.sleep(pause)

            current_y = int(driver.execute_script("return window.scrollY || window.pageYOffset;"))
            return True, current_y
        except Exception as e:
            logger.warning("Browser scroll failed: %s", e)
            return False, 0

    def scrape_hashtag(self, hashtag: str, max_tweets: int) -> list[Tweet]:
        """Scrape tweets for a target hashtag using a dedicated managed browser context.

        Args:
            hashtag (str): Target hashtag query string.
            max_tweets (int): Maximum tweets to collect for this topic.

        Returns:
            list[Tweet]: Collected Tweet objects.
        """
        collected: list[Tweet] = []
        bm = BrowserManager(self.settings)

        try:
            with bm as driver:
                if not self.search_hashtag(driver, hashtag):
                    return collected

                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                scroll_count = 0
                consecutive_pos_matches = 0
                last_y = int(driver.execute_script("return window.scrollY || window.pageYOffset;"))

                while len(collected) < max_tweets and scroll_count < self.settings.max_scrolls:
                    new_tweets = self.extract_visible_tweets(driver, cutoff_time)
                    if new_tweets:
                        remaining = max_tweets - len(collected)
                        added_tweets = new_tweets[:remaining]
                        collected.extend(added_tweets)
                        logger.info(
                            "[%s] Progress: %d/%d tweets (+ %d new in scroll #%d)",
                            hashtag,
                            len(collected),
                            max_tweets,
                            len(added_tweets),
                            scroll_count + 1,
                        )

                    # Perform human-like smooth scroll down
                    success, new_y = self.scroll(driver)
                    scroll_count += 1

                    if success and new_y == last_y:
                        consecutive_pos_matches += 1
                        if consecutive_pos_matches >= 5:
                            logger.info("[%s] Reached bottom of feed (scroll Y position unchanged across 5 scrolls).", hashtag)
                            break
                    else:
                        consecutive_pos_matches = 0
                        last_y = new_y

                    # Config-driven randomized delay between scrolls with jitter bounds
                    sleep_duration = self.settings.scroll_delay + random.uniform(
                        self.settings.scroll_jitter_min, self.settings.scroll_jitter_max
                    )
                    time.sleep(sleep_duration)

        except Exception:
            logger.exception("[%s] Worker thread failed.", hashtag)

        return collected

    def scrape(self, concurrent: bool = True) -> list[Tweet]:
        """Execute market intelligence scraping pipeline sequentially or concurrently across hashtags.

        Args:
            concurrent (bool): If True, executes hashtag scraping in parallel using ThreadPoolExecutor.

        Returns:
            list[Tweet]: Consolidated list of extracted Tweet model objects.
        """
        # Housekeeping: Clean old screenshots once at the start of the scraping run
        cleanup_old_screenshots(days=7)

        all_tweets: list[Tweet] = []
        hashtags = self.settings.search_hashtags
        target_per_hashtag = max(1, self.settings.max_tweets // len(hashtags))

        logger.info(
            "Starting MarketIQ Twitter Scraper (Concurrent=%s, Hashtags: %s, Target per hashtag: ~%d, Max Total: %d)",
            concurrent,
            hashtags,
            target_per_hashtag,
            self.settings.max_tweets,
        )

        if concurrent and len(hashtags) > 1:
            max_workers = min(self.settings.max_workers, len(hashtags))
            logger.info("Launching ThreadPoolExecutor with %d concurrent worker threads...", max_workers)
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_hashtag = {
                        executor.submit(self.scrape_hashtag, tag, target_per_hashtag): tag
                        for tag in hashtags
                    }
                    for future in as_completed(future_to_hashtag):
                        tag = future_to_hashtag[future]
                        try:
                            tweets = future.result()
                            all_tweets.extend(tweets)
                            logger.info("Completed concurrent scraping worker for [%s]: %d tweets.", tag, len(tweets))
                        except Exception:
                            logger.exception("Worker for [%s] failed.", tag)
            except KeyboardInterrupt:
                logger.warning("Scraping execution interrupted by user (Ctrl+C). Cleaning up worker threads...")
                raise
        else:
            for hashtag in hashtags:
                if len(all_tweets) >= self.settings.max_tweets:
                    logger.info("Reached global max tweet quota (%d).", self.settings.max_tweets)
                    break
                tweets = self.scrape_hashtag(hashtag, target_per_hashtag)
                all_tweets.extend(tweets)

        # Truncate to global max_tweets quota
        if len(all_tweets) > self.settings.max_tweets:
            all_tweets = all_tweets[: self.settings.max_tweets]

        logger.info(
            "Scraper pipeline execution complete. Total 24h tweets collected: %d (Duplicates removed: %d)",
            len(all_tweets),
            self.duplicate_count,
        )
        return all_tweets
