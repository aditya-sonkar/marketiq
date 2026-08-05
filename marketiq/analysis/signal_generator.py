"""Trading signal generator for computing composite market indicators and confidence intervals."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketiq.utils.logger import get_logger

logger = get_logger("analysis.signals")


@dataclass(slots=True)
class SignalSummary:
    """Dataclass holding statistical summary metrics of generated trading signals.

    Attributes:
        total_tweets (int): Total number of analyzed tweets.
        mean_signal (float): Mean composite sentiment signal (mu).
        std_dev (float): Standard deviation of signal distribution (sigma).
        ci_95_lower (float): Lower bound of the 95% confidence interval.
        ci_95_upper (float): Upper bound of the 95% confidence interval.
        signal_label (str): Quantitative market trading signal ('BUY', 'SELL', 'HOLD').
        confidence_score (float): Statistical confidence percentage score (0-100%).
        bullish_percentage (float): Percentage of positive sentiment tweets.
        bearish_percentage (float): Percentage of negative sentiment tweets.
    """

    total_tweets: int
    mean_signal: float
    std_dev: float
    ci_95_lower: float
    ci_95_upper: float
    signal_label: str
    confidence_score: float
    bullish_percentage: float
    bearish_percentage: float


class SignalGenerator:
    """Generates engagement-weighted trading signals and statistical confidence intervals."""

    def __init__(
        self,
        buy_threshold: float = 0.05,
        sell_threshold: float = -0.05,
    ) -> None:
        """Initialize SignalGenerator with trading action thresholds.

        Args:
            buy_threshold (float): Minimum positive signal mean required for 'BUY' action.
            sell_threshold (float): Maximum negative signal mean required for 'SELL' action.
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def calculate_composite_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate engagement-weighted sentiment signal for each row in DataFrame.

        Formula:
            weighted_signal = sentiment_score * (1 + log(1 + engagement_score)) * keyword_multiplier

        Args:
            df (pd.DataFrame): Input DataFrame containing sentiment_score and engagement_score.

        Returns:
            pd.DataFrame: Copy of DataFrame with added 'weighted_signal' column.

        Raises:
            ValueError: If required columns are missing from input DataFrame.
        """
        if df.empty:
            df_copy = df.copy()
            df_copy["weighted_signal"] = 0.0
            return df_copy

        required = {"sentiment_score", "engagement_score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required DataFrame columns for signal calculation: {missing}")

        df_copy = df.copy()
        engagement_weight = np.log1p(df_copy["engagement_score"]) + 1.0

        # Balanced keyword multiplier taking both bullish and bearish word frequency into account
        bull_count = df_copy["bullish_keyword_count"] if "bullish_keyword_count" in df_copy.columns else 0
        bear_count = df_copy["bearish_keyword_count"] if "bearish_keyword_count" in df_copy.columns else 0
        keyword_multiplier = 1.0 + (0.05 * (bull_count - bear_count))
        keyword_multiplier = np.clip(keyword_multiplier, a_min=0.5, a_max=None)

        df_copy["weighted_signal"] = df_copy["sentiment_score"] * engagement_weight * keyword_multiplier
        return df_copy

    def generate_summary(self, df: pd.DataFrame) -> SignalSummary:
        """Compute statistical signal metrics including mean, standard deviation, and 95% CI.

        Args:
            df (pd.DataFrame): DataFrame containing 'weighted_signal' or feature columns.

        Returns:
            SignalSummary: Statistically robust signal summary dataclass.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to SignalGenerator. Returning default HOLD summary.")
            return SignalSummary(
                total_tweets=0,
                mean_signal=0.0,
                std_dev=0.0,
                ci_95_lower=0.0,
                ci_95_upper=0.0,
                signal_label="HOLD",
                confidence_score=0.0,
                bullish_percentage=0.0,
                bearish_percentage=0.0,
            )

        if "weighted_signal" not in df.columns:
            df = self.calculate_composite_signal(df)

        signals = df["weighted_signal"].to_numpy()
        # Filter out any non-finite values (NaN / Inf) for calculation safety
        signals = signals[np.isfinite(signals)]
        n = len(signals)

        if n == 0:
            logger.warning("No valid finite signals found in DataFrame. Returning default HOLD summary.")
            return SignalSummary(
                total_tweets=0,
                mean_signal=0.0,
                std_dev=0.0,
                ci_95_lower=0.0,
                ci_95_upper=0.0,
                signal_label="HOLD",
                confidence_score=0.0,
                bullish_percentage=0.0,
                bearish_percentage=0.0,
            )

        mean_val = float(np.mean(signals))
        std_val = float(np.std(signals, ddof=1)) if n > 1 else 0.0

        # Standard Error of Mean (SEM) and 95% Confidence Interval (z = 1.96)
        sem = std_val / math.sqrt(n) if n > 0 else 0.0
        ci_lower = mean_val - (1.96 * sem)
        ci_upper = mean_val + (1.96 * sem)

        # Categorize Trading Action Signal Label
        if mean_val >= self.buy_threshold:
            label = "BUY"
        elif mean_val <= self.sell_threshold:
            label = "SELL"
        else:
            label = "HOLD"

        # Calculate Confidence Score (0 - 100%) based on signal strength
        signal_strength = (abs(mean_val) / (std_val + 1e-6)) * 100.0
        confidence_score = float(min(100.0, max(0.0, signal_strength)))

        # Calculate sentiment percentages
        bullish_count = int(np.sum(signals > 0))
        bearish_count = int(np.sum(signals < 0))
        bull_pct = float((bullish_count / n) * 100.0) if n > 0 else 0.0
        bear_pct = float((bearish_count / n) * 100.0) if n > 0 else 0.0

        summary = SignalSummary(
            total_tweets=n,
            mean_signal=round(mean_val, 4),
            std_dev=round(std_val, 4),
            ci_95_lower=round(ci_lower, 4),
            ci_95_upper=round(ci_upper, 4),
            signal_label=label,
            confidence_score=round(confidence_score, 2),
            bullish_percentage=round(bull_pct, 2),
            bearish_percentage=round(bear_pct, 2),
        )

        logger.info("=========================================================")
        logger.info("              TRADING SIGNAL SUMMARY                     ")
        logger.info("=========================================================")
        logger.info(f"  Tweets Analysed : {summary.total_tweets}")
        logger.info(f"  Mean Signal     : {summary.mean_signal:+.4f}")
        logger.info(f"  Std Dev (sigma) : {summary.std_dev:.4f}")
        logger.info(f"  95% Conf. Int.  : [{summary.ci_95_lower:+.4f}, {summary.ci_95_upper:+.4f}]")
        logger.info(f"  Final Signal    : {summary.signal_label}")
        logger.info(f"  Confidence Score: {summary.confidence_score:.2f}%")
        logger.info("=========================================================")

        return summary
