"""Visualization module for rendering MarketIQ market intelligence charts."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd

from marketiq.analysis.signal_generator import SignalSummary
from marketiq.utils.config import Settings, settings as default_settings
from marketiq.utils.logger import get_logger

logger = get_logger("analysis.visualization")

COLORS = {
    "timeline": "#1da1f2",
    "hashtags": "#2ecc71",
    "engagement": "#9b59b6",
    "signals": "#e74c3c",
}


class MarketVisualizer:
    """Generates and saves publication-quality visualizations for MarketIQ market signals."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize MarketVisualizer.

        Args:
            settings (Optional[Settings]): Settings instance.
        """
        self.settings = settings or default_settings
        self.plots_dir = self.settings.processed_data_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("ggplot")

    def plot_tweets_over_time(self, df: pd.DataFrame, filename: str = "tweets_over_time.png") -> Path:
        """Generate and save time-series chart of tweet volume over time.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'timestamp'.
            filename (str): Target plot image filename.

        Returns:
            Path: Path to generated chart image.
        """
        save_path = self.plots_dir / filename
        if df.empty or "timestamp" not in df.columns:
            logger.warning("Empty DataFrame provided for time-series plot.")
            return save_path

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        df_copy = df.copy()
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
        df_copy.set_index("timestamp", inplace=True)

        # Resample by 1-hour interval
        hourly_counts = df_copy.resample("1h").size()
        hourly_counts.plot(kind="line", ax=ax, marker="o", color=COLORS["timeline"], linewidth=2)

        ax.set_title("Market Discussion Volume Over Time (24h)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (UTC)", fontsize=12)
        ax.set_ylabel("Tweet Count", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved time-series plot to: {save_path}")
        return save_path

    def plot_top_hashtags(
        self, df: pd.DataFrame, top_n: int = 10, filename: str = "top_hashtags.png"
    ) -> Path:
        """Generate and save horizontal bar chart of top stock market hashtags.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'hashtags'.
            top_n (int): Number of top hashtags to plot.
            filename (str): Output filename.

        Returns:
            Path: Path to generated chart image.
        """
        save_path = self.plots_dir / filename
        if df.empty or "hashtags" not in df.columns:
            return save_path

        all_hashtags = []
        for tags in df["hashtags"]:
            if isinstance(tags, (list, tuple)):
                all_hashtags.extend([f"#{t.lstrip('#')}" for t in tags if t])

        if not all_hashtags:
            return save_path

        series = pd.Series(all_hashtags).value_counts().head(top_n).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        series.plot(kind="barh", ax=ax, color=COLORS["hashtags"], edgecolor="black")

        ax.set_title(f"Top {top_n} Market Hashtags", fontsize=14, fontweight="bold")
        ax.set_xlabel("Frequency Count", fontsize=12)
        ax.set_ylabel("Hashtag", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved top hashtags plot to: {save_path}")
        return save_path

    def plot_engagement_distribution(
        self, df: pd.DataFrame, filename: str = "engagement_distribution.png"
    ) -> Path:
        """Generate and save histogram of log-scaled user engagement scores.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'engagement_score'.
            filename (str): Output plot filename.

        Returns:
            Path: Path to saved chart image.
        """
        save_path = self.plots_dir / filename
        if df.empty or "engagement_score" not in df.columns:
            return save_path

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        scores = df["engagement_score"].to_numpy()

        ax.hist(scores, bins=30, color=COLORS["engagement"], edgecolor="black", alpha=0.7)
        ax.set_title("User Engagement Score Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Engagement Score (Likes + Reposts*1.5 + Replies*2.0)", fontsize=11)
        ax.set_ylabel("Tweet Count", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved engagement distribution plot to: {save_path}")
        return save_path

    def plot_signal_distribution(
        self,
        df: pd.DataFrame,
        summary: Optional[SignalSummary] = None,
        filename: str = "signal_distribution.png",
    ) -> Path:
        """Generate and save market sentiment signal distribution with 95% Confidence Interval bounds.

        Args:
            df (pd.DataFrame): Input DataFrame containing 'weighted_signal' or 'sentiment_score'.
            summary (Optional[SignalSummary]): Statistical summary dataclass for overlaying mean & CI bounds.
            filename (str): Target image output path.

        Returns:
            Path: Path to saved chart image.
        """
        save_path = self.plots_dir / filename
        signal_col = "weighted_signal" if "weighted_signal" in df.columns else "sentiment_score"

        if df.empty or signal_col not in df.columns:
            return save_path

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        signals = df[signal_col].to_numpy()

        ax.hist(signals, bins=30, color=COLORS["signals"], edgecolor="black", alpha=0.7)

        if summary is not None:
            ax.axvline(
                summary.mean_signal,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Mean Signal ({summary.mean_signal:.2f})",
            )
            ax.axvline(
                summary.ci_95_lower,
                color="blue",
                linestyle=":",
                linewidth=1.5,
                label=f"95% CI Lower ({summary.ci_95_lower:.2f})",
            )
            ax.axvline(
                summary.ci_95_upper,
                color="blue",
                linestyle=":",
                linewidth=1.5,
                label=f"95% CI Upper ({summary.ci_95_upper:.2f})",
            )
            ax.legend(loc="upper right")

        ax.set_title("Composite Trading Signal Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Weighted Sentiment Signal Score", fontsize=12)
        ax.set_ylabel("Tweet Count", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved trading signal distribution plot to: {save_path}")
        return save_path

    def generate_all_plots(self, df: pd.DataFrame, summary: Optional[SignalSummary] = None) -> list[Path]:
        """Generate all four pipeline analysis charts in a single operation.

        Args:
            df (pd.DataFrame): Feature-engineered DataFrame.
            summary (Optional[SignalSummary]): Calculated signal summary.

        Returns:
            list[Path]: List of successfully generated chart image file paths.
        """
        logger.info("Generating full suite of market intelligence charts...")
        paths = [
            self.plot_tweets_over_time(df),
            self.plot_top_hashtags(df),
            self.plot_engagement_distribution(df),
            self.plot_signal_distribution(df, summary),
        ]
        valid_paths = [p for p in paths if p.exists()]
        logger.info(f"Successfully generated {len(valid_paths)}/{len(paths)} market intelligence visualization plots.")
        return valid_paths
