"""
Order Flow Imbalance (OFI) calculation at single and multiple levels.

References:
    - Cont, Kukanov & Stoikov (2014): Single-level OFI
    - Xu, Gould & Howison (2019): Multi-level OFI (Levels 1–10)
"""

import numpy as np
import pandas as pd


def compute_single_level_ofi(df: pd.DataFrame) -> pd.Series:
    """
    Compute event-by-event single-level OFI (Level 1) as defined by
    Cont, Kukanov & Stoikov (2014).

    OFI_t = ΔV^{bid}_t − ΔV^{ask}_t

    where:
        ΔV^{bid} = change in bid volume contribution
        ΔV^{ask} = change in ask volume contribution

    The contribution logic handles price changes:
        - If bid price rises, previous bid queue is entirely replaced → ΔV^{bid} = new_size
        - If bid price drops, current bid queue lost → ΔV^{bid} = −old_size
        - If bid price unchanged, ΔV^{bid} = new_size − old_size
    Symmetric logic for ask side (inverted sign).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed LOB dataframe with columns:
        bid_price_1, bid_size_1, ask_price_1, ask_size_1

    Returns
    -------
    pd.Series
        OFI values (length = len(df), first row = 0).
    """
    bp = df["bid_price_1"].values
    bs = df["bid_size_1"].values
    ap = df["ask_price_1"].values
    as_ = df["ask_size_1"].values

    n = len(df)
    ofi = np.zeros(n, dtype=np.float64)

    for t in range(1, n):
        # --- Bid side contribution ---
        if bp[t] > bp[t - 1]:
            delta_bid = bs[t]
        elif bp[t] < bp[t - 1]:
            delta_bid = -bs[t - 1]
        else:
            delta_bid = bs[t] - bs[t - 1]

        # --- Ask side contribution ---
        if ap[t] < ap[t - 1]:
            delta_ask = -as_[t]
        elif ap[t] > ap[t - 1]:
            delta_ask = as_[t - 1]
        else:
            delta_ask = as_[t] - as_[t - 1]

        ofi[t] = delta_bid - delta_ask

    return pd.Series(ofi, index=df.index, name="ofi_1")


def _level_ofi_arrays(
    bp: np.ndarray,
    bs: np.ndarray,
    ap: np.ndarray,
    as_: np.ndarray,
) -> np.ndarray:
    """Vectorised single-level OFI on raw numpy arrays (no pandas overhead)."""
    n = len(bp)
    ofi = np.zeros(n, dtype=np.float64)

    delta_bid = np.where(
        bp[1:] > bp[:-1],
        bs[1:],
        np.where(bp[1:] < bp[:-1], -bs[:-1], bs[1:] - bs[:-1]),
    )
    delta_ask = np.where(
        ap[1:] < ap[:-1],
        -as_[1:],
        np.where(ap[1:] > ap[:-1], as_[:-1], as_[1:] - as_[:-1]),
    )
    ofi[1:] = delta_bid - delta_ask
    return ofi


def compute_multi_level_ofi(
    df: pd.DataFrame, max_level: int = 5
) -> pd.DataFrame:
    """
    Compute multi-level OFI for levels 1..max_level.

    For each level k the OFI is computed independently using the
    bid/ask price & size at that level.

    Following Xu, Gould & Howison (2019) we also produce a cumulative
    OFI that sums contributions from levels 1..k:
        OFI_cumul_k = Σ_{i=1}^{k} OFI_i

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed LOB dataframe with columns like
        bid_price_1 .. bid_price_{max_level}, etc.
    max_level : int
        Number of depth levels to use (default 5).

    Returns
    -------
    pd.DataFrame
        Columns: ofi_1 .. ofi_{max_level},
                 ofi_cumul_1 .. ofi_cumul_{max_level}
    """
    result = {}
    cumul = np.zeros(len(df), dtype=np.float64)

    for lvl in range(1, max_level + 1):
        bp = df[f"bid_price_{lvl}"].values
        bs = df[f"bid_size_{lvl}"].values
        ap = df[f"ask_price_{lvl}"].values
        as_ = df[f"ask_size_{lvl}"].values

        ofi_lvl = _level_ofi_arrays(bp, bs, ap, as_)
        result[f"ofi_{lvl}"] = ofi_lvl

        cumul = cumul + ofi_lvl
        result[f"ofi_cumul_{lvl}"] = cumul.copy()

    return pd.DataFrame(result, index=df.index)


def aggregate_ofi_to_time_bars(
    ofi_df: pd.DataFrame,
    timestamps: pd.Series,
    bar_size: str = "1s",
) -> pd.DataFrame:
    """
    Aggregate event-level OFI into fixed time bars by summing.

    Parameters
    ----------
    ofi_df : pd.DataFrame
        Event-level OFI columns (output of compute_multi_level_ofi).
    timestamps : pd.Series
        datetime column aligned with ofi_df.
    bar_size : str
        Pandas offset alias for bar width (e.g. '1s', '5s', '1min').

    Returns
    -------
    pd.DataFrame
        Time-indexed bars with summed OFI per bar.
    """
    tmp = ofi_df.copy()
    tmp["datetime"] = timestamps.values
    tmp = tmp.set_index("datetime")
    return tmp.resample(bar_size).sum()
