"""Browser automation manager for handling Selenium WebDriver lifecycles."""

import random
from pathlib import Path
from types import TracebackType
from typing import Optional, Type
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from marketiq.utils.config import Settings, settings as default_settings
from marketiq.utils.logger import get_logger

logger = get_logger("scraper.browser")


class BrowserManager:
    """Manages Chrome WebDriver initialization, configuration, and lifecycle cleanup.

    Attributes:
        settings (Settings): Configuration settings for driver options and timeouts.
        driver (Optional[webdriver.Chrome]): Active Selenium Chrome WebDriver instance.
    """

    WINDOW_SIZES = [
        "1366,768",
        "1536,864",
        "1600,900",
        "1920,1080",
    ]

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize BrowserManager with configuration settings.

        Args:
            settings (Optional[Settings]): Settings instance. Defaults to global settings.
        """
        self.settings = settings or default_settings
        self._driver: Optional[webdriver.Chrome] = None

    def build_options(self, include_user_profile: bool = True) -> Options:
        """Build standard Chrome Options according to configuration settings.

        Args:
            include_user_profile (bool): Whether to include user data profile arguments.

        Returns:
            Options: Configured Selenium Chrome Options instance.
        """
        options = Options()

        if self.settings.headless:
            options.add_argument("--headless=new")

        # Stealth, background execution, and stability flags
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-features=VizDisplayCompositor,CalculateNativeWinOcclusion")
        options.add_argument("--remote-allow-origins=*")

        # Exclude automation flags to prevent detection
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        # Randomize window size for natural browser fingerprint
        options.add_argument(f"--window-size={random.choice(self.WINDOW_SIZES)}")

        # Configure user data directory for persistent login state
        if include_user_profile and self.settings.user_data_dir:
            user_data_path = Path(self.settings.user_data_dir).resolve()
            user_data_path.mkdir(parents=True, exist_ok=True)
            options.add_argument(f"--user-data-dir={user_data_path}")

            if self.settings.profile_dir:
                options.add_argument(
                    f"--profile-directory={self.settings.profile_dir}"
                )

        options.page_load_strategy = "eager"
        return options

    def create_driver(self) -> webdriver.Chrome:
        """Instantiate and configure a new Selenium Chrome WebDriver using Selenium Manager.

        Returns:
            webdriver.Chrome: Operational Chrome WebDriver instance.

        Raises:
            RuntimeError: If WebDriver initialization fails after retries.
        """
        if self._driver is not None:
            logger.debug("Returning existing active Chrome WebDriver instance.")
            return self._driver

        options = self.build_options()
        logger.info(
            "Initializing Chrome WebDriver (Headless: %s, Timeout: %ss)...",
            self.settings.headless,
            self.settings.page_load_timeout,
        )

        try:
            # Use native Selenium 4.10+ Selenium Manager for driver resolution
            driver = webdriver.Chrome(options=options)
        except Exception as e:
            logger.warning("Native Selenium Manager driver creation failed: %s. Attempting ChromeDriverManager fallback...", e)
            try:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)
            except Exception as fallback_error:
                if self.settings.user_data_dir:
                    logger.warning("Chrome crash detected. Retrying initialization without custom profile directory...")
                    fallback_options = self.build_options(include_user_profile=False)
                    try:
                        driver = webdriver.Chrome(options=fallback_options)
                    except Exception:
                        service = Service(ChromeDriverManager().install())
                        driver = webdriver.Chrome(service=service, options=fallback_options)
                else:
                    logger.error("Failed to create Chrome WebDriver: %s", fallback_error)
                    raise RuntimeError(
                        f"Could not initialize Chrome WebDriver: {fallback_error}"
                    ) from fallback_error

        # Inject CDP command to override navigator.webdriver before document load
        try:
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined
                        });
                    """
                },
            )
        except Exception as e:
            logger.debug("Could not inject CDP webdriver override: %s", e)

        # Configure driver timeouts
        driver.set_page_load_timeout(self.settings.page_load_timeout)
        driver.implicitly_wait(0)

        self._driver = driver
        logger.info("Chrome WebDriver initialized successfully.")

        return self._driver

    def quit_driver(self) -> None:
        """Safely close all tabs and quit the active Chrome WebDriver instance."""
        if self._driver is not None:
            logger.info("Quitting Chrome WebDriver instance...")
            try:
                self._driver.quit()
            except Exception as e:
                logger.warning("Error encountered while quitting Chrome WebDriver: %s", e)
            finally:
                self._driver = None

    def __enter__(self) -> webdriver.Chrome:
        """Context manager entry point.

        Returns:
            webdriver.Chrome: Active Chrome WebDriver instance.
        """
        return self.create_driver()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit point. Ensures cleanup of WebDriver."""
        self.quit_driver()
