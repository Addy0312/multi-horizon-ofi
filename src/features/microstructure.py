"""
Microstructure feature extraction from limit order book data.

Features include:
    - Mid-price
    - Bid-ask spread
    - Volume imbalance
    - Weighted mid-price
    - Order book depth
    - Price level distances / depth profile
"""

import numpy as np
import pandas as pd


def compute_mid_price(df: pd.DataFrame) -> pd.Series:
    """Mid-price = (best_ask + best_bid) / 2."""
    return ((df["ask_price_1"] + df["bid_price_1"]) / 2.0).rename("mid_price")


def compute_spread(df: pd.DataFrame) -> pd.Series:
    """Bid-ask spread = best_ask − best_bid."""
    return (df["ask_price_1"] - df["bid_price_1"]).rename("spread")


def compute_volume_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Volume imbalance at Level 1:
        VI = (bid_size_1 − ask_size_1) / (bid_size_1 + ask_size_1)

    Ranges from −1 (all ask) to +1 (all bid).
    """
    total = df["bid_size_1"] + df["ask_size_1"]
    vi = (df["bid_size_1"] - df["ask_size_1"]) / total.replace(0, np.nan)
    return vi.fillna(0.0).rename("volume_imbalance")


def compute_weighted_mid_price(df: pd.DataFrame) -> pd.Series:
    """
    Weighted mid-price using Level 1 sizes as weights:
        WMP = (ask_price * bid_size + bid_price * ask_size) /
              (bid_size + ask_size)

    When bid_size is large relative to ask_size, the WMP is pulled
    toward the ask (price is about to rise).
    """
    total = df["bid_size_1"] + df["ask_size_1"]
    wmp = (
        df["ask_price_1"] * df["bid_size_1"]
        + df["bid_price_1"] * df["ask_size_1"]
    ) / total.replace(0, np.nan)
    return wmp.fillna(compute_mid_price(df)).rename("weighted_mid_price")


def compute_book_depth(df: pd.DataFrame, max_level: int = 5) -> pd.DataFrame:
    """
    Total visible depth on each side across levels 1..max_level.

    Returns bid_depth, ask_depth, total_depth.
    """
    bid_depth = sum(df[f"bid_size_{i}"] for i in range(1, max_level + 1))
    ask_depth = sum(df[f"ask_size_{i}"] for i in range(1, max_level + 1))
    return pd.DataFrame(
        {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "total_depth": bid_depth + ask_depth,
        },
        index=df.index,
    )


def compute_depth_imbalance(
    df: pd.DataFrame, max_level: int = 5
) -> pd.Series:
    """
    Multi-level depth imbalance:
        DI = (Σ bid_size_i − Σ ask_size_i) / (Σ bid_size_i + Σ ask_size_i)
    """
    depth = compute_book_depth(df, max_level)
    di = (depth["bid_depth"] - depth["ask_depth"]) / depth[
        "total_depth"
    ].replace(0, np.nan)
    return di.fillna(0.0).rename("depth_imbalance")


def compute_price_distances(
    df: pd.DataFrame, max_level: int = 5
) -> pd.DataFrame:
    """
    Distance of each price level from the mid-price (in price units).
    Useful for understanding the depth profile / book shape.
    """
    mid = compute_mid_price(df)
    result = {}
    for lvl in range(1, max_level + 1):
        result[f"ask_dist_{lvl}"] = df[f"ask_price_{lvl}"] - mid
        result[f"bid_dist_{lvl}"] = mid - df[f"bid_price_{lvl}"]
    return pd.DataFrame(result, index=df.index)


def compute_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Log return of mid-price over `horizon` events:
        r_t = log(mid_{t+h}) − log(mid_t)
    """
    mid = compute_mid_price(df)
    log_mid = np.log(mid)
    ret = log_mid.shift(-horizon) - log_mid
    return ret.rename(f"return_{horizon}")


def compute_all_features(
    df: pd.DataFrame, max_level: int = 5
) -> pd.DataFrame:
    """
    Compute all microstructure features and return a single DataFrame.
    """
    features = pd.concat(
        [
            compute_mid_price(df),
            compute_spread(df),
            compute_volume_imbalance(df),
            compute_weighted_mid_price(df),
            compute_book_depth(df, max_level),
            compute_depth_imbalance(df, max_level),
        ],
        axis=1,
    )
    return features
