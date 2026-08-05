"""MarketIQ Pipeline Driver: End-to-End Market Intelligence Data Pipeline."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from marketiq.analysis.features import FeatureExtractor
from marketiq.analysis.signal_generator import SignalGenerator
from marketiq.analysis.tfidf import MarketTfidfVectorizer
from marketiq.analysis.visualization import MarketVisualizer
from marketiq.models.tweet import Tweet
from marketiq.processing.cleaner import TweetCleaner
from marketiq.processing.deduplicator import TweetDeduplicator
from marketiq.scraper.scraper import TwitterScraper
from marketiq.storage.parquet_writer import ParquetStorage
from marketiq.utils.config import Settings, settings
from marketiq.utils.logger import get_logger

logger = get_logger("main")

# Synthetic templates for testing and offline sample mode
SAMPLE_TEMPLATES = [
    ("trader_raj", "Nifty showing strong support at 24500 level. Bullish momentum expected! #nifty50 #intraday", 120, 15, 30, ["#nifty50", "#intraday"], ["@niftytrader"]),
    ("sensex_expert", "Sensex hits new intra-day high. Great breakout in banking stocks #sensex #banknifty", 250, 45, 80, ["#sensex", "#banknifty"], []),
    ("market_bear", "BankNifty seeing heavy selling pressure near 52000 resistance. Put option volume spiking! #banknifty #intraday", 95, 32, 20, ["#banknifty", "#intraday"], []),
    ("option_king", "Bought 24600 CE call option for Nifty intraday target 24800. Strict stoploss at 24450 #nifty50", 180, 22, 40, ["#nifty50"], ["@optiontrader"]),
    ("dalal_street", "Sensex consolidates in tight range. Awaiting RBI policy announcement for directional trigger #sensex", 60, 8, 12, ["#sensex"], []),
]


@dataclass(slots=True)
class PipelineResult:
    """Encapsulates the generated artifacts and execution metrics of a pipeline run."""

    parquet_path: Path
    feature_path: Path
    keyword_path: Path
    summary_path: Path
    plot_paths: list[Path]
    elapsed_seconds: float


def generate_sample_tweets(count: int = 50, seed: int | None = None) -> list[Tweet]:
    """Generate synthetic stock market tweets for testing and dry-run execution.

    Args:
        count (int): Number of synthetic tweets to generate.
        seed (int | None): Optional random seed for reproducible sample runs.

    Returns:
        list[Tweet]: List of synthetic Tweet objects.
    """
    rng = random.Random(seed)
    logger.info("Generating %d synthetic Indian stock market tweets (seed=%s)...", count, seed)

    tweets = []
    now = datetime.now(timezone.utc)
    for i in range(count):
        tpl = rng.choice(SAMPLE_TEMPLATES)
        # Distribute synthetic timestamps across the last 24 hours
        time_offset = timedelta(minutes=(count - i) * 12 + rng.randint(0, 5))
        t = Tweet(
            username=f"{tpl[0]}_{i}",
            timestamp=now - time_offset,
            content=tpl[1],
            likes=tpl[2] + rng.randint(5, 50),
            replies=tpl[3] + rng.randint(1, 10),
            reposts=tpl[4] + rng.randint(2, 20),
            hashtags=tpl[5],
            mentions=tpl[6],
        )
        tweets.append(t)
    return tweets


def run_pipeline(cfg: Settings, sample_mode: bool = False, seed: int | None = None) -> PipelineResult | None:
    """Execute the full end-to-end MarketIQ market intelligence pipeline.

    Args:
        cfg (Settings): Operational configuration settings.
        sample_mode (bool): If True, uses synthetic tweet generator for offline testing.
        seed (int | None): Optional random seed for reproducible synthetic dataset runs.

    Returns:
        PipelineResult | None: Encapsulated pipeline result artifacts or None if exited early.
    """
    start_time = time.perf_counter()

    logger.info("=========================================================")
    logger.info("   Starting MarketIQ Pipeline Execution")
    logger.info("=========================================================")

    # Step 1: Scrape Tweets (or use sample mode)
    if sample_mode:
        raw_tweets = generate_sample_tweets(count=min(100, cfg.max_tweets), seed=seed)
    else:
        logger.info("Step 1: Scraping live market tweets from X/Twitter...")
        scraper = TwitterScraper(cfg)
        raw_tweets = scraper.scrape()

        if not raw_tweets:
            logger.warning("No live tweets were scraped. Falling back to synthetic sample dataset for pipeline verification.")
            raw_tweets = generate_sample_tweets(count=min(100, cfg.max_tweets), seed=seed)

    logger.info("Step 1 Complete: Acquired %d raw tweets.", len(raw_tweets))

    # Step 2: Clean Tweet Content
    logger.info("Step 2: Cleaning and normalizing text payloads...")
    cleaner = TweetCleaner()
    cleaned_tweets = cleaner.clean_tweets(raw_tweets, drop_empty=True)
    logger.info("Step 2 Complete: Cleaned %d tweets.", len(cleaned_tweets))

    # Step 3: SHA-256 Deduplication
    logger.info("Step 3: Executing SHA-256 deduplication...")
    deduplicator = TweetDeduplicator()
    unique_tweets = deduplicator.deduplicate(cleaned_tweets)

    if not unique_tweets:
        logger.warning("No unique tweets remain after deduplication. Exiting pipeline early.")
        return None

    logger.info("Step 3 Complete: Retained %d unique tweets.", len(unique_tweets))

    # Step 4: Storage - Write Parquet
    logger.info("Step 4: Persisting data batch to Parquet storage...")
    storage = ParquetStorage(cfg)
    parquet_path = storage.write_batch(unique_tweets, compression="snappy")
    logger.info("Step 4 Complete: Saved Parquet dataset to %s", parquet_path)

    # Step 5: Feature Engineering & CSV Storage
    logger.info("Step 5: Extracting quantitative features and engagement scores...")
    extractor = FeatureExtractor()
    df_features = extractor.extract_features(unique_tweets)

    if df_features.empty:
        logger.warning("No valid features extracted from tweets. Exiting pipeline early.")
        return None

    feature_path = cfg.processed_data_dir / "features.csv"
    df_features.to_csv(feature_path, index=False)
    logger.info("Step 5 Complete: Feature matrix shape: %s. Saved features to %s", df_features.shape, feature_path)

    # Step 6: TF-IDF Market Term Extraction & JSON Storage
    logger.info("Step 6: Running TF-IDF n-gram keyword extraction...")
    texts = df_features["content"].tolist()
    top_keywords: list[tuple[str, float]] = []

    if not any(t and t.strip() for t in texts):
        logger.warning("No valid text content available for TF-IDF keyword extraction.")
    else:
        tfidf = MarketTfidfVectorizer(max_features=50, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(texts)
        logger.info("TF-IDF Matrix Shape: %d documents × %d features", tfidf_matrix.shape[0], tfidf_matrix.shape[1])

        top_keywords = tfidf.get_top_keywords(texts, top_n=10, matrix=tfidf_matrix)

    keyword_path = cfg.processed_data_dir / "top_keywords.json"
    with open(keyword_path, "w", encoding="utf-8") as f:
        json.dump([{"keyword": k, "score": s} for k, s in top_keywords], f, indent=4)
    logger.info("Saved top keywords to %s", keyword_path)

    logger.info("Top 10 Market TF-IDF Keywords:")
    for kw, score in top_keywords:
        logger.info("  - %-20s: %.4f", kw, score)

    # Step 7: Trading Signal Generation & JSON Export
    logger.info("Step 7: Generating composite market signal & 95% Confidence Intervals...")
    signal_gen = SignalGenerator()
    summary = signal_gen.generate_summary(df_features)

    summary_path = cfg.processed_data_dir / "signal_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=4)
    logger.info("Saved signal summary to %s", summary_path)

    logger.info("---------------------------------------------------------")
    logger.info("               MARKET SIGNAL RESULTS")
    logger.info("---------------------------------------------------------")
    logger.info("Trading Signal Action     : %s", summary.signal_label)
    logger.info("Signal Confidence Score   : %.2f%%", summary.confidence_score)
    logger.info("Mean Signal Score (mu)    : %+.4f", summary.mean_signal)
    logger.info("Standard Deviation (sigma): %.4f", summary.std_dev)
    logger.info("95%% Confidence Interval   : [%+.4f, %+.4f]", summary.ci_95_lower, summary.ci_95_upper)
    logger.info("Bullish vs Bearish Ratio  : %.2f%% Bullish / %.2f%% Bearish", summary.bullish_percentage, summary.bearish_percentage)
    logger.info("---------------------------------------------------------")

    # Step 8: Visualization
    logger.info("Step 8: Rendering and saving visualization charts...")
    visualizer = MarketVisualizer(cfg)
    plot_paths = visualizer.generate_all_plots(df_features, summary)

    elapsed = time.perf_counter() - start_time

    logger.info("=========================================================")
    logger.info("   Pipeline Execution Completed in %.2f seconds", elapsed)
    logger.info("=========================================================")
    logger.info("Outputs Generated (Base Directory: %s):", cfg.output_dir.resolve())
    logger.info("  ✔ Dataset Parquet  : %s", parquet_path)
    logger.info("  ✔ Feature CSV      : %s", feature_path)
    logger.info("  ✔ Top Keywords JSON: %s", keyword_path)
    logger.info("  ✔ Signal JSON      : %s", summary_path)
    for p in plot_paths:
        logger.info("  ✔ Chart Plot       : %s", p)
    logger.info("=========================================================")

    return PipelineResult(
        parquet_path=parquet_path,
        feature_path=feature_path,
        keyword_path=keyword_path,
        summary_path=summary_path,
        plot_paths=plot_paths,
        elapsed_seconds=round(elapsed, 2),
    )


def main() -> None:
    """CLI entry point for MarketIQ pipeline."""
    parser = argparse.ArgumentParser(description="MarketIQ: Real-Time Market Intelligence Pipeline")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run pipeline in offline sample mode using synthetic data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sample dataset generation.",
    )
    parser.add_argument(
        "--max-tweets",
        type=int,
        default=None,
        help="Override maximum tweets to scrape per execution run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory path for generated data artifacts.",
    )
    args = parser.parse_args()

    # Create explicit copy of settings for execution
    run_cfg = replace(settings)

    if args.max_tweets is not None:
        if args.max_tweets <= 0:
            parser.error("--max-tweets must be greater than 0")
        run_cfg = replace(run_cfg, max_tweets=args.max_tweets)

    if args.output_dir:
        out_path = Path(args.output_dir)
        run_cfg = replace(
            run_cfg,
            output_dir=out_path,
            raw_data_dir=out_path / "raw",
            processed_data_dir=out_path / "processed",
        )

    # Guarantee required output directories exist explicitly
    run_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
    run_cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_pipeline(run_cfg, sample_mode=args.sample, seed=args.seed)
        if result:
            logger.info("Pipeline finished successfully in %.2fs.", result.elapsed_seconds)
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception:
        logger.exception("Pipeline execution encountered a critical error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
