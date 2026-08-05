"""Parquet storage engine using PyArrow for efficient column-oriented data persistence."""

from datetime import timezone
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from marketiq.models.tweet import Tweet
from marketiq.utils.config import Settings, settings as default_settings
from marketiq.utils.logger import get_logger

logger = get_logger("storage.parquet")


def get_parquet_schema() -> pa.Schema:
    """Define the explicit PyArrow column schema for MarketIQ tweets.

    Returns:
        pa.Schema: Strongly typed PyArrow schema.
    """
    return pa.schema(
        [
            ("username", pa.string()),
            ("timestamp", pa.timestamp("us", tz="UTC")),
            ("content", pa.string()),
            ("likes", pa.int64()),
            ("replies", pa.int64()),
            ("reposts", pa.int64()),
            ("hashtags", pa.list_(pa.string())),
            ("mentions", pa.list_(pa.string())),
        ]
    )


class ParquetStorage:
    """Handles writing, appending, and reading Tweet data using PyArrow Parquet files."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize ParquetStorage with settings configuration.

        Args:
            settings (Optional[Settings]): Settings instance.
        """
        self.settings = settings or default_settings
        self.schema = get_parquet_schema()

    def tweets_to_table(self, tweets: list[Tweet]) -> pa.Table:
        """Convert a list of Tweet domain objects into a PyArrow Table.

        Args:
            tweets (list[Tweet]): List of Tweet objects.

        Returns:
            pa.Table: PyArrow Table conforming to the project schema.
        """
        if not tweets:
            return pa.Table.from_batches([], schema=self.schema)

        records = []
        for t in tweets:
            ts = t.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            records.append(
                {
                    "username": t.username,
                    "timestamp": ts,
                    "content": t.content,
                    "likes": t.likes,
                    "replies": t.replies,
                    "reposts": t.reposts,
                    "hashtags": t.hashtags,
                    "mentions": t.mentions,
                }
            )

        df = pd.DataFrame(records)
        return pa.Table.from_pandas(df, schema=self.schema)

    def write_batch(
        self,
        tweets: list[Tweet],
        destination: Optional[Union[str, Path]] = None,
        compression: str = "snappy",
    ) -> Path:
        """Write a batch of Tweet objects to a Parquet file using PyArrow.

        Args:
            tweets (list[Tweet]): List of Tweet domain objects.
            destination (Optional[Union[str, Path]]): Destination file path. Defaults to processed_data_dir/tweets.parquet.
            compression (str): Parquet compression codec ('snappy', 'gzip', etc.).

        Returns:
            Path: Resolved absolute path to the written Parquet file.
        """
        if destination is None:
            target_path = self.settings.processed_data_dir / "tweets.parquet"
        else:
            target_path = Path(destination)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        table = self.tweets_to_table(tweets)
        logger.info(f"Writing {len(tweets)} tweets to Parquet file: {target_path} (Compression: {compression})...")

        pq.write_table(table, target_path, compression=compression)
        logger.info(f"Parquet write complete. File size: {target_path.stat().st_size / 1024:.2f} KB.")
        return target_path

    def read_table(self, source: Union[str, Path]) -> pa.Table:
        """Read a Parquet file into a PyArrow Table.

        Args:
            source (Union[str, Path]): Path to Parquet file.

        Returns:
            pa.Table: Loaded PyArrow Table.
        """
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found at: {file_path}")

        logger.info(f"Reading Parquet file: {file_path}")
        return pq.read_table(file_path)

    def read_dataframe(self, source: Union[str, Path]) -> pd.DataFrame:
        """Read a Parquet file directly into a Pandas DataFrame.

        Args:
            source (Union[str, Path]): Path to Parquet file.

        Returns:
            pd.DataFrame: Pandas DataFrame representation.
        """
        table = self.read_table(source)
        return table.to_pandas()

    def read_tweets(self, source: Union[str, Path]) -> list[Tweet]:
        """Read a Parquet file and instantiate Tweet domain objects.

        Args:
            source (Union[str, Path]): Path to Parquet file.

        Returns:
            list[Tweet]: Reconstructed Tweet model instances.
        """
        df = self.read_dataframe(source)
        tweets: list[Tweet] = []

        for row in df.itertuples(index=False):
            hashtags = list(row.hashtags) if isinstance(row.hashtags, (list, np.ndarray)) else []
            mentions = list(row.mentions) if isinstance(row.mentions, (list, np.ndarray)) else []

            tweet = Tweet(
                username=row.username,
                timestamp=pd.to_datetime(row.timestamp).to_pydatetime(),
                content=row.content,
                likes=int(row.likes),
                replies=int(row.replies),
                reposts=int(row.reposts),
                hashtags=hashtags,
                mentions=mentions,
            )
            tweets.append(tweet)

        logger.info(f"Successfully loaded {len(tweets)} Tweet objects from Parquet.")
        return tweets
