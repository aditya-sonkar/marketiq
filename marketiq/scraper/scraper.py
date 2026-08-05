"""Twitter / X market intelligence scraper implementation with concurrent worker support."""

import hashlib
import random
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
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

# Precompiled regular expressions for performance
ENGAGEMENT_RE = re.compile(r"([\d,.]+\s*[kKmMbB]?)")
HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@(\w+)")
STATUS_ID_RE = re.compile(r"/status/(\d+)")
USERNAME_RE = re.compile(r"@([A-Za-z0-9_]{1,15})")


class PageStatus(Enum):
    """Enumeration of browser page states encountered during search navigation."""

    OK = auto()
    LOGIN_REQUIRED = auto()
    RATE_LIMITED = auto()


def parse_engagement_number(text: Optional[str]) -> int:
    """Safely parse engagement strings (e.g., '1.2K', '3.4M', '45', '1,200') into integers.

    Args:
        text (Optional[str]): Metric text or aria-label snippet.

    Returns:
        int: Parsed integer value, or 0 if unparseable.
    """
    if not text:
        return 0
    match = ENGAGEMENT_RE.search(text)
    if not match:
        return 0

    raw = match.group(1).replace(",", "").strip().lower()
    try:
        if raw.endswith("k"):
            return int(float(raw[:-1]) * 1_000)
        if raw.endswith("m"):
            return int(float(raw[:-1]) * 1_000_000)
        if raw.endswith("b"):
            return int(float(raw[:-1]) * 1_000_000_000)
        return int(float(raw))
    except (ValueError, TypeError):
        return 0


def cleanup_old_screenshots(days: Optional[int] = None) -> None:
    """Clean up timestamped debug screenshot PNG files older than specified days in logs/.

    Args:
        days (Optional[int]): Maximum age threshold in days. Defaults to settings.screenshot_retention_days.
    """
    retention_days = days if days is not None else default_settings.screenshot_retention_days
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return

    cutoff = time.time() - (retention_days * 86400)
    for screenshot in logs_dir.glob("*.png"):
        try:
            if screenshot.stat().st_mtime < cutoff:
                screenshot.unlink()
                logger.debug("Cleaned up old screenshot: %s", screenshot.name)
        except Exception as e:
            logger.debug("Failed to delete screenshot %s: %s", screenshot.name, e)


class TwitterScraper:
    """Production-grade scraper for retrieving market tweets from X/Twitter with concurrent capabilities.

    Attributes:
        settings (Settings): Configured setting bounds and timeouts.
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
    LOGIN_WALL_LOCATOR = (
        By.CSS_SELECTOR,
        '[data-testid="loginButton"], a[href*="/login"], a[href="/i/flow/login"], [data-testid="sheetDialog"]',
    )
    SPINNER_LOCATOR = (By.CSS_SELECTOR, '[data-testid="spinner"], [role="progressbar"]')
    RATE_LIMIT_PHRASES = (
        "rate limit exceeded",
        "something went wrong",
        "try again later",
    )

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize TwitterScraper dependencies."""
        self.settings = settings or default_settings
        self.seen_tweet_hashes: Set[str] = set()
        self.hash_lock = Lock()
        self.duplicate_count = 0

    def _get_backoff_delay(self, attempt: int) -> float:
        """Calculate config-driven exponential backoff delay with random jitter for retries."""
        base = self.settings.retry_base_delay * (2 ** (attempt - 1))
        delay = min(base, self.settings.retry_max_delay)
        return delay + random.uniform(0.0, 1.0)

    def check_page_status(self, driver: WebDriver) -> PageStatus:
        """Detect whether X/Twitter page status is OK, LOGIN_REQUIRED, or RATE_LIMITED.

        Args:
            driver (WebDriver): Active browser driver.

        Returns:
            PageStatus: Status classification of the current browser state.
        """
        try:
            current_url = (driver.current_url or "").lower()
            if "/login" in current_url or "/i/flow/login" in current_url:
                logger.warning("X/Twitter login URL detected: %s", driver.current_url)
                return PageStatus.LOGIN_REQUIRED

            elements = driver.find_elements(*self.LOGIN_WALL_LOCATOR)
            if elements:
                logger.warning("X/Twitter login wall element detected.")
                return PageStatus.LOGIN_REQUIRED

            body_text = (
                driver.execute_script("return (document.body ? document.body.innerText : '').toLowerCase();")
                or ""
            )
            for phrase in self.RATE_LIMIT_PHRASES:
                if phrase in body_text:
                    logger.warning("X/Twitter rate limit or error banner detected: '%s'", phrase)
                    return PageStatus.RATE_LIMITED
        except Exception as e:
            logger.debug("Failed checking page status: %s", e)
        return PageStatus.OK

    def is_authenticated(self, driver: WebDriver) -> bool:
        """Check if active browser session is authenticated on X/Twitter using DOM elements.

        Args:
            driver (WebDriver): Active browser driver.

        Returns:
            bool: True if user session is authenticated, False if login wall is present.
        """
        try:
            current_url = (driver.current_url or "").lower()
            if "/login" in current_url or "/i/flow/login" in current_url:
                return False

            if driver.find_elements(*self.LOGIN_WALL_LOCATOR):
                return False

            authenticated_selectors = (
                '[data-testid="AppTabBar_Home_Link"],'
                '[data-testid="SideNav_AccountSwitcher_Button"],'
                'article[data-testid="tweet"]'
            )
            return bool(driver.find_elements(By.CSS_SELECTOR, authenticated_selectors))
        except Exception as e:
            logger.debug("Error checking authentication status: %s", e)
        return False

    def is_loading_spinner_visible(self, driver: WebDriver) -> bool:
        """Check if X/Twitter network loading spinner is actively visible on DOM."""
        try:
            return len(driver.find_elements(*self.SPINNER_LOCATOR)) > 0
        except Exception:
            return False

    def wait_for_authentication(self, driver: WebDriver, hashtag: str, timeout: Optional[int] = None) -> bool:
        """Poll browser state using WebDriverWait until user completes authentication in Chrome.

        Args:
            driver (WebDriver): Active browser driver.
            hashtag (str): Target hashtag query string.
            timeout (Optional[int]): Maximum wait timeout in seconds. Defaults to self.settings.auth_timeout.

        Returns:
            bool: True once authenticated and ready to scrape, False if session fails or times out.
        """
        wait_timeout = timeout or self.settings.auth_timeout
        logger.info("Login required. Waiting for user authentication (timeout: %ds)...", wait_timeout)
        try:
            wait = WebDriverWait(driver, wait_timeout, poll_frequency=2.5)
            wait.until(lambda d: self.is_authenticated(d))
            logger.info("Login detected. Continuing scraping...")

            encoded_query = urllib.parse.quote(hashtag)
            target_url = self.SEARCH_BASE_URL.format(query=encoded_query)
            driver.get(target_url)

            if self.check_page_status(driver) != PageStatus.OK:
                logger.warning("[%s] Page status after login navigation is not OK.", hashtag)
                return False

            return True
        except TimeoutException:
            logger.error("[%s] Authentication wait timed out after %d seconds.", hashtag, wait_timeout)
            return False
        except WebDriverException as e:
            logger.warning("WebDriver error while waiting for authentication: %s", e)
            return False
        except Exception as e:
            logger.warning("Unexpected error during authentication wait: %s", e)
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
        """Navigate to search query with automatic retry logic, login wall, and rate limit handling.

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

                status = self.check_page_status(driver)
                if status == PageStatus.LOGIN_REQUIRED or not self.is_authenticated(driver):
                    self.save_debug_screenshot(driver, "login_wall", hashtag)
                    if self.wait_for_authentication(driver, hashtag):
                        wait = WebDriverWait(driver, self.settings.page_load_timeout)
                        wait.until(EC.visibility_of_any_elements_located(self.TWEET_CONTAINER_LOCATOR))
                        logger.info("[%s] Search page loaded successfully after authentication.", hashtag)
                        return True
                    else:
                        logger.warning("[%s] Authentication wait failed on attempt %d/%d.", hashtag, attempt, retries)
                        continue
                elif status == PageStatus.RATE_LIMITED:
                    self.save_debug_screenshot(driver, "rate_limit", hashtag)
                    logger.warning("[%s] Rate limit detected on attempt %d/%d. Sleeping before retry...", hashtag, attempt, retries)
                    time.sleep(random.uniform(10.0, 20.0))
                    continue

                wait = WebDriverWait(driver, self.settings.page_load_timeout)
                wait.until(EC.visibility_of_any_elements_located(self.TWEET_CONTAINER_LOCATOR))
                logger.info("[%s] Search page loaded successfully.", hashtag)
                return True
            except TimeoutException:
                if not self.is_authenticated(driver):
                    logger.info("[%s] Page load timeout due to unauthenticated session. Triggering authentication wait...", hashtag)
                    if self.wait_for_authentication(driver, hashtag):
                        return True
                logger.warning("[%s] Page load timeout on attempt %d/%d.", hashtag, attempt, retries)
                self.save_debug_screenshot(driver, "timeout", hashtag)
            except WebDriverException as e:
                logger.warning("[%s] WebDriver error on attempt %d/%d: %s", hashtag, attempt, retries, e)
                self.save_debug_screenshot(driver, "error", hashtag)

            backoff_delay = self._get_backoff_delay(attempt)
            logger.debug("[%s] Backing off for %.2fs before retry attempt %d...", hashtag, backoff_delay, attempt + 1)
            time.sleep(backoff_delay)

        logger.error("[%s] Failed to load search page after %d attempts.", hashtag, retries)
        return False

    def _safe_find_text(self, parent: WebElement, locator: Tuple[str, str]) -> str:
        """Safely extract stripped text from a child element locator."""
        try:
            return parent.find_element(*locator).text.strip()
        except Exception:
            return ""

    def _extract_status_id(self, element: WebElement) -> Optional[str]:
        """Extract permanent Tweet numeric status ID from status link href."""
        try:
            link_elem = element.find_element(*self.STATUS_LINK_LOCATOR)
            href = link_elem.get_attribute("href") or ""
            match = STATUS_ID_RE.search(href)
            return match.group(1) if match else None
        except Exception:
            return None

    def extract_single_tweet(
        self, element: WebElement, cutoff_time: datetime, status_id: Optional[str] = None
    ) -> Optional[Tweet]:
        """Parse raw DOM element into a Tweet model with SHA-256 deduplication and 24-hour filtering.

        Args:
            element (WebElement): Tweet article DOM element.
            cutoff_time (datetime): Minimum allowed UTC creation timestamp.
            status_id (Optional[str]): Pre-extracted permanent status ID if available.

        Returns:
            Optional[Tweet]: Parsed Tweet object, or None if skipped/invalid.
        """
        try:
            # 1. Username
            username = "unknown"
            user_text = self._safe_find_text(element, self.USER_NAME_LOCATOR)
            if user_text:
                match = USERNAME_RE.search(user_text)
                if match:
                    username = match.group(1)

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
            content = self._safe_find_text(element, self.TWEET_TEXT_LOCATOR)
            if not content:
                try:
                    full_text = element.text.strip()
                    lines = [line.strip() for line in full_text.split("\n") if line.strip() and not line.startswith("@")]
                    content = " ".join(lines)
                except Exception:
                    content = ""

            if not content and username == "unknown":
                return None

            # 4. Immutable Tweet ID Extraction & SHA-256 Deduplication
            tweet_id = status_id or self._extract_status_id(element)
            raw_hash_input = tweet_id if tweet_id else f"{username.lower()}:{timestamp.isoformat()}:{content}"
            sha256_hash = hashlib.sha256(raw_hash_input.encode("utf-8")).hexdigest()

            with self.hash_lock:
                if sha256_hash in self.seen_tweet_hashes:
                    self.duplicate_count += 1
                    logger.debug("Duplicate tweet skipped: @%s (id=%s)", username, tweet_id)
                    return None
                self.seen_tweet_hashes.add(sha256_hash)

            # 5. Engagement Metrics
            replies = self._extract_metric_count(element, self.REPLY_LOCATOR)
            reposts = self._extract_metric_count(element, self.RETWEET_LOCATOR)
            likes = self._extract_metric_count(element, self.LIKE_LOCATOR)

            # 6. Hashtags & Mentions
            hashtags = HASHTAG_RE.findall(content)
            mentions = MENTION_RE.findall(content)

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
        except Exception:
            return 0

    def extract_visible_tweets(
        self, driver: WebDriver, cutoff_time: datetime, seen_status_ids: set[str]
    ) -> list[Tweet]:
        """Scrape only newly rendered tweet elements using permanent status IDs, avoiding DOM re-processing.

        Args:
            driver (WebDriver): Active browser driver.
            cutoff_time (datetime): Minimum allowed UTC creation timestamp.
            seen_status_ids (set[str]): Set of immutable status IDs already parsed in this session.

        Returns:
            list[Tweet]: Newly extracted Tweet instances.
        """
        extracted: list[Tweet] = []

        try:
            articles = driver.find_elements(*self.TWEET_CONTAINER_LOCATOR)
            for article in articles:
                status_id = self._extract_status_id(article)
                if status_id and status_id in seen_status_ids:
                    continue

                tweet = self.extract_single_tweet(article, cutoff_time, status_id=status_id)
                if tweet:
                    if status_id:
                        seen_status_ids.add(status_id)
                    extracted.append(tweet)
        except Exception as e:
            logger.warning("Error querying visible DOM tweets: %s", e)

        return extracted

    def scroll(self, driver: WebDriver) -> None:
        """Execute human-like randomized scroll action.

        Args:
            driver (WebDriver): Active browser driver.
        """
        try:
            old_count = len(driver.find_elements(*self.TWEET_CONTAINER_LOCATOR))
            pixels = random.randint(self.settings.min_scroll_pixels, self.settings.max_scroll_pixels)
            driver.execute_script("window.scrollBy(0, arguments[0]);", pixels)

            try:
                wait = WebDriverWait(driver, 4.0)
                wait.until(lambda d: len(d.find_elements(*self.TWEET_CONTAINER_LOCATOR)) > old_count)
            except TimeoutException:
                time.sleep(1.0)

            if random.random() < self.settings.reading_pause_probability:
                pause = random.uniform(self.settings.reading_pause_min, self.settings.reading_pause_max)
                time.sleep(pause)
        except Exception as e:
            logger.warning("Browser scroll failed: %s", e)

    def scrape_hashtag(self, hashtag: str, max_tweets: int) -> list[Tweet]:
        """Scrape tweets for a target hashtag using a dedicated managed browser context.

        Args:
            hashtag (str): Target hashtag query string.
            max_tweets (int): Maximum tweets to collect for this topic.

        Returns:
            list[Tweet]: Collected Tweet objects.
        """
        collected: list[Tweet] = []
        # Dedicated BrowserManager per worker thread for thread-isolated Chrome contexts
        bm = BrowserManager(self.settings)

        try:
            with bm as driver:
                if not self.search_hashtag(driver, hashtag):
                    return collected

                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.settings.cutoff_hours)
                scroll_count = 0
                consecutive_no_growth = 0
                seen_status_ids: set[str] = set()

                while len(collected) < max_tweets and scroll_count < self.settings.max_scrolls:
                    seen_count_before = len(seen_status_ids)
                    new_tweets = self.extract_visible_tweets(driver, cutoff_time, seen_status_ids)
                    added_count = 0
                    if new_tweets:
                        remaining = max_tweets - len(collected)
                        added_tweets = new_tweets[:remaining]
                        added_count = len(added_tweets)
                        collected.extend(added_tweets)
                        logger.info(
                            "[%s] Progress: %d/%d tweets (+ %d new in scroll #%d)",
                            hashtag,
                            len(collected),
                            max_tweets,
                            added_count,
                            scroll_count + 1,
                        )

                    self.scroll(driver)
                    scroll_count += 1

                    if added_count == 0 and len(seen_status_ids) == seen_count_before:
                        status = self.check_page_status(driver)
                        if status == PageStatus.LOGIN_REQUIRED or not self.is_authenticated(driver):
                            self.save_debug_screenshot(driver, "login_wall", hashtag)
                            logger.info("[%s] Login required to continue scrolling. Waiting for user authentication...", hashtag)
                            if not self.wait_for_authentication(driver, hashtag):
                                logger.warning("[%s] Authentication wait failed or timed out.", hashtag)
                                break
                            consecutive_no_growth = 0
                            continue

                        if self.is_loading_spinner_visible(driver):
                            logger.debug("[%s] Network spinner active. Pausing for X/Twitter payload...", hashtag)
                            time.sleep(2.0)
                            continue

                        consecutive_no_growth += 1
                        if consecutive_no_growth >= self.settings.no_progress_limit:
                            logger.info(
                                "[%s] Reached bottom of feed (no new unique status IDs across %d scrolls).",
                                hashtag,
                                self.settings.no_progress_limit,
                            )
                            break
                    else:
                        consecutive_no_growth = 0

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
        start_time = time.time()
        cleanup_old_screenshots(days=self.settings.screenshot_retention_days)

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

        if len(all_tweets) > self.settings.max_tweets:
            all_tweets = all_tweets[: self.settings.max_tweets]

        elapsed = time.time() - start_time
        rate = len(all_tweets) / elapsed if elapsed > 0 else 0.0

        logger.info("=========================================================")
        logger.info("                SCRAPER EXECUTION SUMMARY                ")
        logger.info("=========================================================")
        logger.info("  Hashtags Target     : %s", hashtags)
        logger.info("  Total Tweets Saved  : %d / %d", len(all_tweets), self.settings.max_tweets)
        logger.info("  Duplicates Removed  : %d", self.duplicate_count)
        logger.info("  Scraping Throughput : %.2f tweets/sec", rate)
        logger.info("  Total Elapsed Time  : %.2fs", elapsed)
        logger.info("=========================================================")

        return all_tweets
