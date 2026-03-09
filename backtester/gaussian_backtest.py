"""
Gaussian Channel Strategy v3.3 — Python Backtester
====================================================
Implements the Gaussian Channel indicator (DonovanWall, open source) with
trend-direction-based entry/exit logic reverse-engineered from the v3.3
public description.

Core idea:
  - Enter long when channel turns green (filter rising) + price confirms above filter
  - Stay in as long as the trend holds — don't exit on pullbacks
  - Exit only when channel flips red (trend reversal)
  - Optional: late entries when price breaks above upper band mid-trend
  - Optional: short trades (mirror logic)

Usage:
    1. pip install -r requirements.txt
    2. set TWELVE_DATA_API_KEY=your_key
    3. python gaussian_backtest.py
    4. python gaussian_backtest.py --symbols BTC/USD ETH/USD --interval 1day
    5. python gaussian_backtest.py --enable-short
    6. python gaussian_backtest.py --optimize

Created By: Wooanaz
Created On: 3/9/2026
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

# Results directory (next to this script)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

try:
    from twelvedata import TDClient
except ImportError:
    print("ERROR: twelvedata package not installed.  Run:  pip install twelvedata")
    sys.exit(1)


# ── Constants ───────────────────────────────────────────────────
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD"]
DEFAULT_INTERVAL = "1day"
DEFAULT_BARS = 1500  # ~6 yrs daily — long history for trend strategies


# ╔══════════════════════════════════════════════════════════════╗
# ║                    DATA FETCHING                            ║
# ╚══════════════════════════════════════════════════════════════╝

def fetch_ohlcv(td: TDClient, symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    """Fetch OHLCV from Twelve Data, returns clean DataFrame sorted oldest-first."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ts = td.time_series(
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                timezone="America/New_York",
            )
            df = ts.as_pandas()
            if df is None or df.empty:
                raise ValueError(f"No data returned for {symbol}")
            break
        except Exception as e:
            if "API credits" in str(e) and attempt < max_retries - 1:
                print(f"      Rate limited, waiting 65s...")
                time.sleep(65)
            else:
                raise

    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║              GAUSSIAN CHANNEL CALCULATIONS                  ║
# ╚══════════════════════════════════════════════════════════════╝

def gaussian_filter_pole(alpha: float, source: np.ndarray, poles: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply an N-pole Gaussian (Ehlers) IIR filter.

    This is an iterative (recursive) filter — each output depends on
    previous outputs, so it must be computed bar-by-bar.

    Returns (filter_n_pole, filter_1_pole).
    """
    n = len(source)

    # Binomial coefficients for each pole count (Pascal's triangle rows)
    binom = {
        1: [1, 1],
        2: [1, 2, 1],
        3: [1, 3, 3, 1],
        4: [1, 4, 6, 4, 1],
        5: [1, 5, 10, 10, 5, 1],
        6: [1, 6, 15, 20, 15, 6, 1],
        7: [1, 7, 21, 35, 35, 21, 7, 1],
        8: [1, 8, 28, 56, 70, 56, 28, 8, 1],
        9: [1, 9, 36, 84, 126, 126, 84, 36, 9, 1],
    }

    def _run_filter(p: int) -> np.ndarray:
        """Run a single p-pole filter."""
        coeffs = binom[p]
        x = 1 - alpha
        f = np.zeros(n)

        for i in range(n):
            # a^p * source[i]
            val = (alpha ** p) * source[i]

            # Add recursive terms: alternating sign, binomial weights
            for j in range(1, p + 1):
                prev = f[i - j] if i - j >= 0 else 0.0
                sign = (-1) ** (j + 1)  # +, -, +, -, ...
                val += sign * coeffs[j] * (x ** j) * prev

            f[i] = val
        return f

    filt_n = _run_filter(poles)
    filt_1 = _run_filter(1)

    return filt_n, filt_1


def compute_gaussian_channel(df: pd.DataFrame, src_type: str = "hlc3",
                             poles: int = 4, period: int = 144,
                             mult: float = 1.414,
                             reduced_lag: bool = False,
                             fast_response: bool = False) -> pd.DataFrame:
    """
    Compute the Gaussian Channel indicator on an OHLCV DataFrame.

    Adds columns: filt, hband, lband, trend_up, trend_down, trend_flip_up, trend_flip_down
    """
    df = df.copy()

    # Source calculation
    if src_type == "hlc3":
        src = (df["high"] + df["low"] + df["close"]) / 3
    elif src_type == "close":
        src = df["close"]
    elif src_type == "ohlc4":
        src = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    elif src_type == "hl2":
        src = (df["high"] + df["low"]) / 2
    else:
        src = df["close"]

    df["src"] = src.values

    # True Range
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]  # first bar

    # Beta and Alpha (Ehlers formula)
    pi_val = 4 * np.arcsin(1)  # = pi, matching Pine's math.asin(1)*4
    beta = (1 - np.cos(pi_val / period)) / (1.414 ** (2 / poles) - 1)
    alpha = -beta + np.sqrt(beta ** 2 + 2 * beta)

    # Lag reduction
    lag = int((period - 1) / (2 * poles))
    if reduced_lag:
        src_data = src + (src - src.shift(lag))
        tr_data = tr + (tr - tr.shift(lag))
        src_data = src_data.fillna(src)
        tr_data = tr_data.fillna(tr)
    else:
        src_data = src
        tr_data = tr

    # Apply Gaussian filter
    filtn, filt1 = gaussian_filter_pole(alpha, src_data.values, poles)
    filtntr, filt1tr = gaussian_filter_pole(alpha, tr_data.values, poles)

    # Fast response mode: average N-pole with 1-pole
    if fast_response:
        filt = (filtn + filt1) / 2
        filttr = (filtntr + filt1tr) / 2
    else:
        filt = filtn
        filttr = filtntr

    df["filt"] = filt
    df["hband"] = filt + filttr * mult
    df["lband"] = filt - filttr * mult

    # Trend detection
    df["trend_up"] = df["filt"] > df["filt"].shift(1)
    df["trend_down"] = df["filt"] < df["filt"].shift(1)
    df["trend_flip_up"] = df["trend_up"] & ~df["trend_up"].shift(1).fillna(False)
    df["trend_flip_down"] = df["trend_down"] & ~df["trend_down"].shift(1).fillna(False)

    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║                    STRATEGY CONFIG                          ║
# ╚══════════════════════════════════════════════════════════════╝

class GaussianConfig:
    """Holds all parameters for a single Gaussian Channel strategy run."""

    def __init__(self, **kwargs):
        # Strategy version: "v3.3" (default), "v3.0", "v3.1"
        self.strategy_version: str = kwargs.get("strategy_version", "v3.3")

        # Gaussian Channel params
        self.src_type: str = kwargs.get("src_type", "hlc3")
        self.poles: int = kwargs.get("poles", 4)
        self.period: int = kwargs.get("period", 144)
        self.mult: float = kwargs.get("mult", 1.414)
        self.reduced_lag: bool = kwargs.get("reduced_lag", False)
        self.fast_response: bool = kwargs.get("fast_response", False)

        # Strategy params
        self.enable_short: bool = kwargs.get("enable_short", False)
        self.enable_late_entry: bool = kwargs.get("enable_late_entry", True)

        # Exit mode: "trend_flip" (default v3.3), "filter_cross", "lband_cross"
        #   trend_flip   = exit when channel color flips (most conservative, fewest trades)
        #   filter_cross = exit when close drops below filter mid-line
        #   lband_cross  = exit when close drops below lower band
        self.exit_mode: str = kwargs.get("exit_mode", "trend_flip")

        # v3.1-specific params
        self.stoch_rsi_k: int = kwargs.get("stoch_rsi_k", 3)
        self.stoch_rsi_d: int = kwargs.get("stoch_rsi_d", 3)
        self.stoch_rsi_len: int = kwargs.get("stoch_rsi_len", 14)
        self.stoch_rsi_stoch_len: int = kwargs.get("stoch_rsi_stoch_len", 14)
        self.atr_len: int = kwargs.get("atr_len", 14)
        self.atr_sl_mult: float = kwargs.get("atr_sl_mult", 1.5)
        self.use_atr_sl: bool = kwargs.get("use_atr_sl", True)
        self.equity_pct: float = kwargs.get("equity_pct", 1.0)  # v3.1 uses 0.75

    def short_name(self) -> str:
        parts = [f"{self.strategy_version}", f"GC({self.poles}p/{self.period}/{self.mult})"]
        parts.append(f"src={self.src_type}")
        if self.reduced_lag:
            parts.append("RL")
        if self.fast_response:
            parts.append("FR")
        if self.enable_short:
            parts.append("SHORT")
        if self.strategy_version == "v3.3" and self.enable_late_entry:
            parts.append("LATE")
        exit_labels = {
            "trend_flip": "TrendFlip",
            "filter_cross": "FiltX",
            "lband_cross": "LbandX",
        }
        if self.strategy_version == "v3.3":
            parts.append(exit_labels.get(self.exit_mode, self.exit_mode))
        elif self.strategy_version == "v3.1":
            parts.append("StochRSI")
            if self.use_atr_sl:
                parts.append(f"ATR_SL({self.atr_sl_mult}x)")
            parts.append(f"{int(self.equity_pct*100)}%eq")
        return " ".join(parts)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    INDICATOR HELPERS                        ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_stoch_rsi(df: pd.DataFrame, rsi_len: int = 14, stoch_len: int = 14,
                      k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """Compute Stochastic RSI, adds columns: stoch_k, stoch_d."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=rsi_len, min_periods=rsi_len).mean()
    avg_loss = loss.rolling(window=rsi_len, min_periods=rsi_len).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(window=stoch_len, min_periods=stoch_len).min()
    rsi_max = rsi.rolling(window=stoch_len, min_periods=stoch_len).max()
    stoch_rsi = ((rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)) * 100

    df["stoch_k"] = stoch_rsi.rolling(window=k_smooth, min_periods=k_smooth).mean()
    df["stoch_d"] = df["stoch_k"].rolling(window=d_smooth, min_periods=d_smooth).mean()
    return df


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Compute ATR, adds column: atr."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    df["atr"] = tr.rolling(window=length, min_periods=length).mean()
    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║                    TRADE SIMULATION                         ║
# ╚══════════════════════════════════════════════════════════════╝

def simulate_trades(df: pd.DataFrame, cfg: GaussianConfig) -> list[dict]:
    """Dispatch to the correct strategy version's simulator."""
    if cfg.strategy_version == "v3.0":
        return _simulate_v30(df, cfg)
    elif cfg.strategy_version == "v3.1":
        return _simulate_v31(df, cfg)
    else:
        return _simulate_v33(df, cfg)


def _simulate_v30(df: pd.DataFrame, cfg: GaussianConfig) -> list[dict]:
    """
    v3.0 — Original Gaussian Channel strategy.
    Entry: crossover(close, hband)
    Exit:  crossunder(close, hband)
    100% equity, long only.
    """
    close = df["close"].values
    hband = df["hband"].values
    dates = df.index
    n = len(close)

    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None

    for i in range(1, n):
        if in_position:
            # Exit: close crosses under hband
            if close[i] < hband[i] and close[i - 1] >= hband[i - 1]:
                _append_trade(trades, "long", entry_price, close[i], entry_date, dates[i], "XunderHband")
                in_position = False
            continue

        # Entry: close crosses over hband
        if close[i] > hband[i] and close[i - 1] <= hband[i - 1]:
            entry_price = close[i]
            entry_date = dates[i]
            in_position = True

    if in_position:
        pnl = (close[-1] / entry_price - 1) * 100
        trades.append({"side": "long", "entry_price": entry_price, "entry_date": entry_date,
                       "exit_price": close[-1], "exit_date": dates[-1], "exit_reason": "OPEN",
                       "pnl_pct": pnl})
    return trades


def _simulate_v31(df: pd.DataFrame, cfg: GaussianConfig) -> list[dict]:
    """
    v3.1 — Enhanced with StochRSI filter + ATR stop loss.
    Entry: channelGreen AND close > hband AND (stochK > 80 or stochK < 20)
    Exit:  crossunder(close, hband) OR ATR stop loss hit
    75% equity (handled by equity_pct in config for compounding).
    """
    # Compute extra indicators
    df = compute_stoch_rsi(df, rsi_len=cfg.stoch_rsi_len, stoch_len=cfg.stoch_rsi_stoch_len,
                           k_smooth=cfg.stoch_rsi_k, d_smooth=cfg.stoch_rsi_d)
    df = compute_atr(df, length=cfg.atr_len)

    close = df["close"].values
    hband = df["hband"].values
    trend_up = df["trend_up"].values
    stoch_k = df["stoch_k"].values
    atr = df["atr"].values
    dates = df.index
    n = len(close)

    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    stop_loss = 0.0

    for i in range(1, n):
        if in_position:
            # Exit: ATR stop loss (if enabled)
            if cfg.use_atr_sl and close[i] <= stop_loss:
                _append_trade(trades, "long", entry_price, stop_loss, entry_date, dates[i], "ATR_SL")
                in_position = False
                continue
            # Exit: close crosses under hband
            if close[i] < hband[i] and close[i - 1] >= hband[i - 1]:
                _append_trade(trades, "long", entry_price, close[i], entry_date, dates[i], "XunderHband")
                in_position = False
            continue

        # Entry: channel green + close > hband + StochRSI extreme
        if (trend_up[i]
                and close[i] > hband[i]
                and not np.isnan(stoch_k[i])
                and (stoch_k[i] > 80 or stoch_k[i] < 20)):
            entry_price = close[i]
            entry_date = dates[i]
            atr_val = atr[i] if not np.isnan(atr[i]) else 0
            stop_loss = entry_price - (cfg.atr_sl_mult * atr_val)
            in_position = True

    if in_position:
        pnl = (close[-1] / entry_price - 1) * 100
        trades.append({"side": "long", "entry_price": entry_price, "entry_date": entry_date,
                       "exit_price": close[-1], "exit_date": dates[-1], "exit_reason": "OPEN",
                       "pnl_pct": pnl})
    return trades


def _simulate_v33(df: pd.DataFrame, cfg: GaussianConfig) -> list[dict]:
    """
    v3.3 — Reverse-engineered trend-following strategy.

    Entry: Channel flips green + close > filt (or late crossover close/hband while green)
    Exit:  Configurable (trend_flip / filter_cross / lband_cross)
    """
    close = df["close"].values
    filt = df["filt"].values
    hband = df["hband"].values
    lband = df["lband"].values
    trend_up = df["trend_up"].values
    trend_down = df["trend_down"].values
    trend_flip_up = df["trend_flip_up"].values
    trend_flip_down = df["trend_flip_down"].values
    dates = df.index
    n = len(close)

    trades = []
    in_position = False
    side = ""
    entry_price = 0.0
    entry_date = None

    for i in range(1, n):  # start at 1 so we can check [i-1]
        if in_position:
            # --- EXIT LOGIC ---
            exited = False

            if side == "long":
                if cfg.exit_mode == "trend_flip":
                    # Exit when channel turns red
                    if trend_flip_down[i]:
                        _append_trade(trades, "long", entry_price, close[i], entry_date, dates[i], "TrendFlip")
                        exited = True
                elif cfg.exit_mode == "filter_cross":
                    # Exit when close drops below filter
                    if close[i] < filt[i] and close[i - 1] >= filt[i - 1]:
                        _append_trade(trades, "long", entry_price, close[i], entry_date, dates[i], "FiltX")
                        exited = True
                elif cfg.exit_mode == "lband_cross":
                    # Exit when close drops below lower band
                    if close[i] < lband[i] and close[i - 1] >= lband[i - 1]:
                        _append_trade(trades, "long", entry_price, close[i], entry_date, dates[i], "LbandX")
                        exited = True

            elif side == "short":
                if cfg.exit_mode == "trend_flip":
                    if trend_flip_up[i]:
                        _append_trade(trades, "short", entry_price, close[i], entry_date, dates[i], "TrendFlip")
                        exited = True
                elif cfg.exit_mode == "filter_cross":
                    if close[i] > filt[i] and close[i - 1] <= filt[i - 1]:
                        _append_trade(trades, "short", entry_price, close[i], entry_date, dates[i], "FiltX")
                        exited = True
                elif cfg.exit_mode == "lband_cross":
                    if close[i] > hband[i] and close[i - 1] <= hband[i - 1]:
                        _append_trade(trades, "short", entry_price, close[i], entry_date, dates[i], "HbandX")
                        exited = True

            if exited:
                in_position = False
            continue

        # --- ENTRY LOGIC ---
        # Primary: Channel just flipped + price confirms
        long_flip_entry = (
            trend_flip_up[i] and close[i] > filt[i]
        )
        # Allow entry one bar after flip if price wasn't confirming on flip bar
        long_flip_delayed = (
            i >= 2 and trend_flip_up[i - 1] and close[i - 1] <= filt[i - 1]
            and trend_up[i] and close[i] > filt[i]
        )
        # Late entry: mid-trend, price breaks above upper band (strong confirmation)
        long_late_entry = (
            cfg.enable_late_entry
            and trend_up[i]
            and close[i] > hband[i] and close[i - 1] <= hband[i - 1]
        )

        if long_flip_entry or long_flip_delayed or long_late_entry:
            entry_price = close[i]
            entry_date = dates[i]
            side = "long"
            in_position = True
            continue

        # Short entries (optional)
        if cfg.enable_short:
            short_flip_entry = (
                trend_flip_down[i] and close[i] < filt[i]
            )
            short_flip_delayed = (
                i >= 2 and trend_flip_down[i - 1] and close[i - 1] >= filt[i - 1]
                and trend_down[i] and close[i] < filt[i]
            )
            short_late_entry = (
                cfg.enable_late_entry
                and trend_down[i]
                and close[i] < lband[i] and close[i - 1] >= lband[i - 1]
            )

            if short_flip_entry or short_flip_delayed or short_late_entry:
                entry_price = close[i]
                entry_date = dates[i]
                side = "short"
                in_position = True
                continue

    # Close any open position at the end
    if in_position:
        pnl = (close[-1] / entry_price - 1) * 100 if side == "long" else (1 - close[-1] / entry_price) * 100
        trades.append({
            "side": side, "entry_price": entry_price, "entry_date": entry_date,
            "exit_price": close[-1], "exit_date": dates[-1], "exit_reason": "OPEN",
            "pnl_pct": pnl,
        })

    return trades


def _append_trade(trades: list, side: str, entry_price: float, exit_price: float,
                  entry_date, exit_date, reason: str):
    if side == "long":
        pnl = (exit_price / entry_price - 1) * 100
    else:
        pnl = (1 - exit_price / entry_price) * 100
    trades.append({
        "side": side, "entry_price": entry_price, "entry_date": entry_date,
        "exit_price": exit_price, "exit_date": exit_date, "exit_reason": reason,
        "pnl_pct": pnl,
    })


# ╔══════════════════════════════════════════════════════════════╗
# ║                    COMPOUNDING EQUITY CURVE                 ║
# ╚══════════════════════════════════════════════════════════════╝

def calc_compound_equity(trades: list[dict], initial_capital: float = 1000.0,
                         equity_pct: float = 1.0) -> tuple[float, float]:
    """
    Calculate compounding equity.
    equity_pct controls how much of capital is risked per trade (1.0 = 100%, 0.75 = 75%).
    Returns (final_equity, max_drawdown_pct).
    """
    equity = initial_capital
    peak = initial_capital
    max_dd_pct = 0.0

    for t in trades:
        pct = t["pnl_pct"] / 100.0
        equity += equity * equity_pct * pct
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd_pct:
            max_dd_pct = dd

    return equity, max_dd_pct


# ╔══════════════════════════════════════════════════════════════╗
# ║                    PERFORMANCE METRICS                      ║
# ╚══════════════════════════════════════════════════════════════╝

def calc_metrics(trades: list[dict], initial_capital: float = 1000.0,
                 equity_pct: float = 1.0) -> dict:
    if not trades:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
            "total_pnl": 0, "max_drawdown": 0, "expectancy": 0,
            "final_equity": initial_capital, "compound_dd": 0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    total = len(pnls)
    n_wins = len(wins)

    final_eq, compound_dd = calc_compound_equity(trades, initial_capital, equity_pct)

    return {
        "total": total,
        "wins": n_wins,
        "losses": len(losses),
        "win_rate": round(n_wins / total * 100, 1) if total > 0 else 0,
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "total_pnl": round(sum(pnls), 2),
        "max_drawdown": round(max_dd, 2),
        "expectancy": round(np.mean(pnls), 2) if pnls else 0,
        "final_equity": round(final_eq, 2),
        "compound_dd": round(compound_dd, 2),
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║                    OPTIMIZATION                             ║
# ╚══════════════════════════════════════════════════════════════╝

def build_optimization_configs(enable_short: bool = False) -> list[GaussianConfig]:
    """Grid search over Gaussian Channel parameters."""
    configs = []
    grid = {
        "period": [100, 144, 200],
        "mult": [1.0, 1.414, 2.0],
        "poles": [3, 4, 5],
        "exit_mode": ["trend_flip", "filter_cross", "lband_cross"],
        "enable_late_entry": [True, False],
    }

    keys = list(grid.keys())
    for combo in product(*grid.values()):
        kwargs = dict(zip(keys, combo))
        kwargs["enable_short"] = enable_short
        configs.append(GaussianConfig(**kwargs))

    return configs


# ╔══════════════════════════════════════════════════════════════╗
# ║                    TRADE LOG                                ║
# ╚══════════════════════════════════════════════════════════════╝

def print_trade_log(trades: list[dict], symbol: str):
    """Print individual trades for inspection."""
    if not trades:
        print(f"  No trades for {symbol}")
        return

    print(f"\n  {'─' * 90}")
    print(f"  TRADE LOG — {symbol}")
    print(f"  {'─' * 90}")

    headers = ["#", "Side", "Entry Date", "Entry $", "Exit Date", "Exit $", "PnL %", "Reason"]
    rows = []
    for i, t in enumerate(trades, 1):
        entry_d = t["entry_date"].strftime("%Y-%m-%d") if hasattr(t["entry_date"], "strftime") else str(t["entry_date"])[:10]
        exit_d = t["exit_date"].strftime("%Y-%m-%d") if hasattr(t["exit_date"], "strftime") else str(t["exit_date"])[:10]
        pnl_str = f"{t['pnl_pct']:+.2f}%"
        rows.append([i, t["side"].upper(), entry_d, f"${t['entry_price']:,.2f}",
                      exit_d, f"${t['exit_price']:,.2f}", pnl_str, t["exit_reason"]])

    print(tabulate(rows, headers=headers, tablefmt="pretty", stralign="right"))


# ╔══════════════════════════════════════════════════════════════╗
# ║                    SAVE RESULTS                             ║
# ╚══════════════════════════════════════════════════════════════╝

def save_results(all_results: list[dict], all_trades: dict[str, list[dict]],
                 interval: str, initial_capital: float, mode: str):
    """
    Save backtest results to the results/ folder.
    Creates:
      - results/gc_summary_<timestamp>.csv   — one row per symbol/config
      - results/gc_trades_<timestamp>.csv    — every individual trade
      - results/gc_run_<timestamp>.json      — full run metadata + results
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"gc_{mode}_{interval}_{ts}"

    # ── Summary CSV ─────────────────────────────────────────────
    summary_path = RESULTS_DIR / f"{tag}_summary.csv"
    summary_fields = ["symbol", "config", "total", "wins", "losses", "win_rate",
                      "avg_win", "avg_loss", "profit_factor", "total_pnl",
                      "max_drawdown", "expectancy", "final_equity", "compound_dd"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    # ── Trades CSV ──────────────────────────────────────────────
    trades_path = RESULTS_DIR / f"{tag}_trades.csv"
    trade_rows = []
    for key, trades in all_trades.items():
        for i, t in enumerate(trades, 1):
            entry_d = t["entry_date"].strftime("%Y-%m-%d") if hasattr(t["entry_date"], "strftime") else str(t["entry_date"])[:10]
            exit_d = t["exit_date"].strftime("%Y-%m-%d") if hasattr(t["exit_date"], "strftime") else str(t["exit_date"])[:10]
            trade_rows.append({
                "symbol_config": key,
                "trade_num": i,
                "side": t["side"],
                "entry_date": entry_d,
                "entry_price": round(t["entry_price"], 2),
                "exit_date": exit_d,
                "exit_price": round(t["exit_price"], 2),
                "pnl_pct": round(t["pnl_pct"], 2),
                "exit_reason": t["exit_reason"],
            })

    if trade_rows:
        trade_fields = ["symbol_config", "trade_num", "side", "entry_date",
                        "entry_price", "exit_date", "exit_price", "pnl_pct", "exit_reason"]
        with open(trades_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trade_fields)
            writer.writeheader()
            writer.writerows(trade_rows)

    # ── Full JSON run ───────────────────────────────────────────
    json_path = RESULTS_DIR / f"{tag}_run.json"
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "interval": interval,
        "initial_capital": initial_capital,
        "mode": mode,
        "summary": all_results,
        "trades": {k: [
            {**t,
             "entry_date": t["entry_date"].strftime("%Y-%m-%d") if hasattr(t["entry_date"], "strftime") else str(t["entry_date"])[:10],
             "exit_date": t["exit_date"].strftime("%Y-%m-%d") if hasattr(t["exit_date"], "strftime") else str(t["exit_date"])[:10],
             "entry_price": round(t["entry_price"], 2),
             "exit_price": round(t["exit_price"], 2),
             "pnl_pct": round(t["pnl_pct"], 2),
            } for t in v
        ] for k, v in all_trades.items()},
    }
    with open(json_path, "w") as f:
        json.dump(run_data, f, indent=2, default=str)

    print(f"\n  Results saved to:")
    print(f"    Summary: {summary_path}")
    print(f"    Trades:  {trades_path}")
    print(f"    Full:    {json_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                    MAIN                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Gaussian Channel v3.3 Strategy Backtester")
    parser.add_argument("--api-key", default=os.environ.get("TWELVE_DATA_API_KEY"),
                        help="Twelve Data API key (or set TWELVE_DATA_API_KEY env var)")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Symbols to test (e.g. BTC/USD ETH/USD SOL/USD)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL,
                        help="Candle interval (1day, 4h, 1h, etc.)")
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS,
                        help="Number of bars to fetch per symbol")
    parser.add_argument("--initial-capital", type=float, default=1000.0,
                        help="Starting capital for compounding equity calc")

    # Gaussian Channel params
    parser.add_argument("--poles", type=int, default=4, help="Number of poles (1-9)")
    parser.add_argument("--period", type=int, default=144, help="Sampling period")
    parser.add_argument("--mult", type=float, default=1.414, help="True range multiplier")
    parser.add_argument("--src", default="hlc3", choices=["hlc3", "close", "ohlc4", "hl2"],
                        help="Price source")
    parser.add_argument("--reduced-lag", action="store_true", help="Enable reduced lag mode")
    parser.add_argument("--fast-response", action="store_true", help="Enable fast response mode")

    # Strategy params
    parser.add_argument("--enable-short", action="store_true", help="Enable short trades")
    parser.add_argument("--no-late-entry", action="store_true", help="Disable late (mid-trend) entries")
    parser.add_argument("--exit-mode", default="trend_flip",
                        choices=["trend_flip", "filter_cross", "lband_cross"],
                        help="Exit strategy")
    parser.add_argument("--compare-exits", action="store_true",
                        help="Compare all exit modes side-by-side (v3.3 only)")
    parser.add_argument("--compare-versions", action="store_true",
                        help="Compare v3.0, v3.1, and v3.3 strategies head-to-head")
    parser.add_argument("--optimize", action="store_true",
                        help="Grid search over parameter combinations")
    parser.add_argument("--show-trades", action="store_true",
                        help="Print individual trade log")

    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key provided.")
        print("  Set: TWELVE_DATA_API_KEY=your_key")
        sys.exit(1)

    td = TDClient(apikey=args.api_key)

    # ── Fetch data ──────────────────────────────────────────────
    datasets: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(args.symbols):
        try:
            print(f"  Fetching {sym} ({args.interval}, {args.bars} bars)...")
            datasets[sym] = fetch_ohlcv(td, sym, args.interval, args.bars)
            print(f"    -> {len(datasets[sym])} bars loaded ({datasets[sym].index[0]} to {datasets[sym].index[-1]})")
            if i < len(args.symbols) - 1:
                time.sleep(9)
        except Exception as e:
            print(f"    X Failed to fetch {sym}: {e}")

    if not datasets:
        print("\nNo data loaded. Exiting.")
        sys.exit(1)

    # ── Build configs ───────────────────────────────────────────
    if args.optimize:
        configs = build_optimization_configs(args.enable_short)
        print(f"\n  Optimization mode: testing {len(configs)} configurations per symbol\n")
    elif args.compare_versions:
        base_kw = dict(
            poles=args.poles, period=args.period, mult=args.mult,
            src_type=args.src, reduced_lag=args.reduced_lag,
            fast_response=args.fast_response,
            enable_short=args.enable_short,
        )
        configs = [
            # v3.0: crossover/crossunder hband, 100% equity
            GaussianConfig(strategy_version="v3.0", **base_kw),
            # v3.1: StochRSI filter + ATR SL, 75% equity
            GaussianConfig(strategy_version="v3.1", equity_pct=0.75, **base_kw),
            # v3.3: trend flip entry, trend_flip exit (default), late entry
            GaussianConfig(strategy_version="v3.3", exit_mode="trend_flip",
                           enable_late_entry=not args.no_late_entry, **base_kw),
            # v3.3: trend flip entry, filter_cross exit
            GaussianConfig(strategy_version="v3.3", exit_mode="filter_cross",
                           enable_late_entry=not args.no_late_entry, **base_kw),
            # v3.3: trend flip entry, lband_cross exit
            GaussianConfig(strategy_version="v3.3", exit_mode="lband_cross",
                           enable_late_entry=not args.no_late_entry, **base_kw),
        ]
        print(f"\n  Comparing {len(configs)} strategy versions head-to-head\n")
    elif args.compare_exits:
        configs = [
            GaussianConfig(poles=args.poles, period=args.period, mult=args.mult,
                           src_type=args.src, reduced_lag=args.reduced_lag,
                           fast_response=args.fast_response,
                           enable_short=args.enable_short,
                           enable_late_entry=not args.no_late_entry,
                           exit_mode=em)
            for em in ["trend_flip", "filter_cross", "lband_cross"]
        ]
        print(f"\n  Comparing {len(configs)} exit strategies per symbol\n")
    else:
        configs = [
            GaussianConfig(
                poles=args.poles, period=args.period, mult=args.mult,
                src_type=args.src, reduced_lag=args.reduced_lag,
                fast_response=args.fast_response,
                enable_short=args.enable_short,
                enable_late_entry=not args.no_late_entry,
                exit_mode=args.exit_mode,
            )
        ]
        print(f"\n  Config: {configs[0].short_name()}\n")

    # ── Run backtests ───────────────────────────────────────────
    all_results = []
    all_trades: dict[str, list[dict]] = {}  # key = "SYMBOL [config]" → trade list

    for sym, raw_df in datasets.items():
        best_pf = -1
        best_result = None
        best_trades = None
        best_key = ""

        for cfg in configs:
            df = compute_gaussian_channel(
                raw_df, src_type=cfg.src_type, poles=cfg.poles,
                period=cfg.period, mult=cfg.mult,
                reduced_lag=cfg.reduced_lag, fast_response=cfg.fast_response,
            )
            trades = simulate_trades(df, cfg)
            metrics = calc_metrics(trades, args.initial_capital, cfg.equity_pct)
            metrics["symbol"] = sym
            metrics["config"] = cfg.short_name()
            trade_key = f"{sym} [{cfg.short_name()}]"

            if args.optimize:
                if metrics["total"] >= 3 and metrics["profit_factor"] > best_pf:
                    best_pf = metrics["profit_factor"]
                    best_result = metrics
                    best_trades = trades
                    best_key = trade_key
            else:
                all_results.append(metrics)
                all_trades[trade_key] = trades
                if args.show_trades:
                    print_trade_log(trades, trade_key)

        if args.optimize and best_result:
            all_results.append(best_result)
            all_trades[best_key] = best_trades
            if args.show_trades and best_trades:
                print_trade_log(best_trades, f"{sym} [BEST]")

    # ── Print results ───────────────────────────────────────────
    if not all_results:
        print("No trades generated. Try different parameters or more historical data.")
        sys.exit(0)

    print("\n" + "=" * 120)
    print("  GAUSSIAN CHANNEL — BACKTEST RESULTS")
    print("=" * 120)
    print(f"  Interval: {args.interval}  |  Capital: ${args.initial_capital:,.0f}  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if args.optimize:
        print("  Mode: OPTIMIZATION (best config per symbol, min 3 trades)")
    print("=" * 120 + "\n")

    headers = ["Symbol", "Config", "Trades", "W", "L", "Win %",
               "Avg Win", "Avg Loss", "PF", "Sum PnL %", "DD %",
               "Expect %", "Equity $", "Cmpd DD %"]

    rows = []
    for r in all_results:
        rows.append([
            r["symbol"], r["config"], r["total"], r["wins"], r["losses"],
            f"{r['win_rate']}%", f"{r['avg_win']}%", f"{r['avg_loss']}%",
            r["profit_factor"], f"{r['total_pnl']}%", f"{r['max_drawdown']}%",
            f"{r['expectancy']}%", f"${r['final_equity']:,.2f}", f"{r['compound_dd']}%",
        ])

    print(tabulate(rows, headers=headers, tablefmt="pretty", stralign="right"))

    print("\n--- LEGEND ---")
    print("  PF       = Profit Factor (gross wins / gross losses — above 1.5 is strong)")
    print("  DD %     = Max drawdown on cumulative PnL curve (additive)")
    print("  Equity $ = Final equity with compounding (100% of capital per trade)")
    print("  Cmpd DD  = Max drawdown on compounding equity curve")
    print("  TrendFlip = Exit only when channel flips color (rides full trend)")
    print("  FiltX     = Exit when close crosses below filter mid-line")
    print("  LbandX    = Exit when close crosses below lower band")
    print("  LATE      = Allows mid-trend entries on upper band breakout")
    print()

    # ── Save to files ───────────────────────────────────────────
    mode = "optimize" if args.optimize else "versions" if args.compare_versions else "compare" if args.compare_exits else "single"
    save_results(all_results, all_trades, args.interval, args.initial_capital, mode)


if __name__ == "__main__":
    main()
