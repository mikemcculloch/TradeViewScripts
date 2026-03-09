"""
Multi-Strategy Backtester — Compare TradingView Strategies
==========================================================
Translates Pine Script strategies into Python, runs them against
the same OHLCV data, and compares performance side-by-side.

Strategies included:
  1. EMA Pullback (from existing backtest.py)
  2. Gaussian Channel v3.0 (simple crossover)
  3. Gaussian Channel v3.0 + StochRSI (AI-enhanced)
  4. Andean Oscillator (bull/bear momentum crossover)

Simulates compound equity growth from a starting capital (default $1,000)
with configurable commission to see which strategy gets closest to $1M.

Usage:
    python multi_backtest.py
    python multi_backtest.py --symbols BTC/USD ETH/USD --interval 1day
    python multi_backtest.py --capital 1000 --target 1000000
    python multi_backtest.py --optimize

Created By: Wooanaz
Created On: 3/7/2026
"""

import argparse
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from tabulate import tabulate

try:
    from twelvedata import TDClient
except ImportError:
    print("ERROR: twelvedata package not installed.  Run:  pip install twelvedata")
    sys.exit(1)

# ── Default Assets ──────────────────────────────────────────────
DEFAULT_SYMBOLS = [
    "BTC/USD", "ETH/USD", "XRP/USD",
    "GOOGL", "HIMS", "PLTR",
]

DEFAULT_INTERVAL = "1day"
DEFAULT_BARS = 1000


# ╔══════════════════════════════════════════════════════════════╗
# ║                    DATA FETCHING                            ║
# ╚══════════════════════════════════════════════════════════════╝

def fetch_ohlcv(td: TDClient, symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    """Fetch OHLCV data from Twelve Data and return a clean DataFrame."""
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
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0
        df.attrs["has_volume"] = False
    else:
        df.attrs["has_volume"] = True
    df.dropna(subset=["close"], inplace=True)
    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║                    SHARED INDICATORS                        ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Bar-by-bar true range (not smoothed)."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr


def compute_andean_oscillator(close: np.ndarray, open_: np.ndarray,
                              length: int = 50, sig_length: int = 9
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Andean Oscillator by @alexgrover.

    Uses exponential envelopes to decompose price action into bull & bear
    momentum components.  Returns (bull, bear, signal) arrays.

    bull = sqrt(dn2 - dn1^2)  -- rises when there's bullish momentum
    bear = sqrt(up2 - up1^2)  -- rises when there's bearish momentum
    signal = EMA(max(bull, bear), sig_length)
    """
    n = len(close)
    alpha = 2.0 / (length + 1)

    up1 = np.zeros(n)
    up2 = np.zeros(n)
    dn1 = np.zeros(n)
    dn2 = np.zeros(n)

    # Initialize bar 0
    up1[0] = close[0]
    up2[0] = close[0] ** 2
    dn1[0] = close[0]
    dn2[0] = close[0] ** 2

    for i in range(1, n):
        c = close[i]
        o = open_[i]

        up1[i] = max(c, o, up1[i - 1] - (up1[i - 1] - c) * alpha)
        up2[i] = max(c * c, o * o, up2[i - 1] - (up2[i - 1] - c * c) * alpha)

        dn1[i] = min(c, o, dn1[i - 1] + (c - dn1[i - 1]) * alpha)
        dn2[i] = min(c * c, o * o, dn2[i - 1] + (c * c - dn2[i - 1]) * alpha)

    # Components — clamp to 0 to avoid sqrt of negative due to float precision
    bull = np.sqrt(np.maximum(dn2 - dn1 * dn1, 0.0))
    bear = np.sqrt(np.maximum(up2 - up1 * up1, 0.0))

    # Signal line = EMA of max(bull, bear)
    max_bb = np.maximum(bull, bear)
    sig_alpha = 2.0 / (sig_length + 1)
    signal = np.zeros(n)
    signal[0] = max_bb[0]
    for i in range(1, n):
        signal[i] = sig_alpha * max_bb[i] + (1.0 - sig_alpha) * signal[i - 1]

    return bull, bear, signal


def compute_stoch_rsi(close: pd.Series, rsi_len: int = 14, stoch_len: int = 14,
                       smooth_k: int = 3, smooth_d: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI — returns (K, D) lines."""
    rsi = compute_rsi(close, rsi_len)
    stoch_rsi = (rsi - rsi.rolling(stoch_len).min()) / \
                (rsi.rolling(stoch_len).max() - rsi.rolling(stoch_len).min())
    stoch_rsi = stoch_rsi * 100
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d


# ╔══════════════════════════════════════════════════════════════╗
# ║              GAUSSIAN FILTER (Ehlers N-pole)                ║
# ╚══════════════════════════════════════════════════════════════╝

def _binomial_weight(n: int, k: int) -> int:
    """C(n, k) — binomial coefficient for filter weights."""
    if k < 0 or k > n:
        return 0
    result = 1
    for i in range(min(k, n - k)):
        result = result * (n - i) // (i + 1)
    return result


def gaussian_filter_npole(src: np.ndarray, alpha: float, poles: int) -> np.ndarray:
    """
    Compute an N-pole Gaussian (Ehlers) filter iteratively.

    This is equivalent to the Pine Script f_filt9x recursive formula.
    For pole count N, the filter equation is:

        f[i] = alpha^N * src[i]
             + sum_{k=1..N} (-1)^(k+1) * C(N,k) * (1-alpha)^k * f[i-k]

    where C(N,k) is the binomial coefficient.
    """
    n = len(src)
    f = np.zeros(n)
    x = 1.0 - alpha
    a_n = alpha ** poles

    # Pre-compute signed binomial weights: (-1)^(k+1) * C(N,k) * (1-alpha)^k
    weights = []
    for k in range(1, poles + 1):
        sign = 1.0 if k % 2 == 1 else -1.0
        w = sign * _binomial_weight(poles, k) * (x ** k)
        weights.append(w)

    for i in range(n):
        val = a_n * src[i]
        for k, w in enumerate(weights):
            prev_idx = i - (k + 1)
            if prev_idx >= 0:
                val += w * f[prev_idx]
        f[i] = val

    return f


def compute_gaussian_channel(df: pd.DataFrame, poles: int = 4, period: int = 144,
                              mult: float = 1.414, reduced_lag: bool = False,
                              fast_response: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Gaussian Channel: (filter_line, high_band, low_band).

    Matches the Pine Script Gaussian Channel indicator by DonovanWall.
    """
    pi = 4.0 * np.arcsin(1.0)  # same as Pine: 4*math.asin(1)
    beta = (1.0 - np.cos(2.0 * pi / period)) / (1.414 ** (2.0 / poles) - 1.0)
    alpha = -beta + np.sqrt(beta ** 2 + 2.0 * beta)

    hlc3 = (df["high"].values + df["low"].values + df["close"].values) / 3.0
    tr = compute_true_range(df).values

    lag = int((period - 1) / (2 * poles))

    if reduced_lag:
        src_data = np.copy(hlc3)
        tr_data = np.copy(tr)
        for i in range(lag, len(src_data)):
            src_data[i] = hlc3[i] + (hlc3[i] - hlc3[i - lag])
            tr_data[i] = tr[i] + (tr[i] - tr[i - lag])
    else:
        src_data = hlc3
        tr_data = tr

    # N-pole filter on source
    filtn = gaussian_filter_npole(src_data, alpha, poles)
    # 1-pole filter on source (for fast response averaging)
    filt1 = gaussian_filter_npole(src_data, alpha, 1)

    # N-pole filter on true range
    filtn_tr = gaussian_filter_npole(tr_data, alpha, poles)
    filt1_tr = gaussian_filter_npole(tr_data, alpha, 1)

    if fast_response:
        filt = (filtn + filt1) / 2.0
        filt_tr = (filtn_tr + filt1_tr) / 2.0
    else:
        filt = filtn
        filt_tr = filtn_tr

    hband = filt + filt_tr * mult
    lband = filt - filt_tr * mult

    return filt, hband, lband


# ╔══════════════════════════════════════════════════════════════╗
# ║               COOLDOWN HELPER                               ║
# ╚══════════════════════════════════════════════════════════════╝

def apply_cooldown(signals: np.ndarray, cooldown: int) -> np.ndarray:
    result = np.zeros(len(signals), dtype=bool)
    last_idx = -999
    for i in range(len(signals)):
        if signals[i] and (i - last_idx > cooldown):
            result[i] = True
            last_idx = i
    return result


# ╔══════════════════════════════════════════════════════════════╗
# ║               STRATEGY BASE CLASS                           ║
# ╚══════════════════════════════════════════════════════════════╝

class Strategy(ABC):
    """Base class for all strategies."""

    name: str = "Base"
    commission_pct: float = 0.1  # one-way commission %

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'buy_signal' and 'sell_signal' (or 'close_signal') columns.
        For SL/TP strategies, also add 'atr' column.
        """
        ...

    @property
    def exit_mode(self) -> str:
        """'sl_tp' for ATR-based SL/TP, 'signal' for close-on-signal."""
        return "sl_tp"

    def config_label(self) -> str:
        return self.name


# ╔══════════════════════════════════════════════════════════════╗
# ║               STRATEGY 1: EMA PULLBACK                      ║
# ╚══════════════════════════════════════════════════════════════╝

class EMAPullbackStrategy(Strategy):
    name = "EMA Pullback"
    commission_pct = 0.04

    def __init__(self, fast_len=50, slow_len=200, use_cooldown=True, cooldown_bars=3,
                 filter_sideways=True, sideways_threshold=0.005, use_volume=True,
                 use_candle_confirm=True, use_slope=True, slope_lookback=10,
                 use_rsi=False, rsi_overbought=70.0, rsi_oversold=30.0,
                 sl_atr_mult=2.0, rr_ratio=2.0):
        self.fast_len = fast_len
        self.slow_len = slow_len
        self.use_cooldown = use_cooldown
        self.cooldown_bars = cooldown_bars
        self.filter_sideways = filter_sideways
        self.sideways_threshold = sideways_threshold
        self.use_volume = use_volume
        self.use_candle_confirm = use_candle_confirm
        self.use_slope = use_slope
        self.slope_lookback = slope_lookback
        self.use_rsi = use_rsi
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.sl_atr_mult = sl_atr_mult
        self.rr_ratio = rr_ratio

    @property
    def exit_mode(self) -> str:
        return "sl_tp"

    def config_label(self) -> str:
        parts = [f"EMA({self.fast_len}/{self.slow_len})"]
        if self.use_cooldown: parts.append("CD")
        if self.filter_sideways: parts.append("SW")
        if self.use_volume: parts.append("VOL")
        if self.use_candle_confirm: parts.append("CC")
        if self.use_slope: parts.append("SLP")
        if self.use_rsi: parts.append("RSI")
        parts.append(f"SL{self.sl_atr_mult}x")
        parts.append(f"RR{self.rr_ratio}")
        return " ".join(parts)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = compute_ema(df["close"], self.fast_len)
        df["ema_slow"] = compute_ema(df["close"], self.slow_len)
        df["rsi"] = compute_rsi(df["close"], 14)
        df["atr"] = compute_atr(df, 14)

        has_volume = df.attrs.get("has_volume", df["volume"].sum() > 0)
        df["vol_sma"] = df["volume"].rolling(20).mean() if has_volume else 0

        # Trend
        df["bull_trend"] = df["ema_fast"] > df["ema_slow"]
        df["bear_trend"] = df["ema_fast"] < df["ema_slow"]

        # Sideways
        df["ema_spread"] = (df["ema_fast"] - df["ema_slow"]).abs() / df["ema_slow"]
        df["is_sideways"] = df["ema_spread"] < self.sideways_threshold

        # Price crosses slow EMA
        df["cross_up_slow"] = (df["close"].shift(1) < df["ema_slow"].shift(1)) & (df["close"] > df["ema_slow"])
        df["cross_down_slow"] = (df["close"].shift(1) > df["ema_slow"].shift(1)) & (df["close"] < df["ema_slow"])

        # Price crosses fast EMA
        df["cross_up_fast"] = (df["close"].shift(1) < df["ema_fast"].shift(1)) & (df["close"] > df["ema_fast"])
        df["cross_down_fast"] = (df["close"].shift(1) > df["ema_fast"].shift(1)) & (df["close"] < df["ema_fast"])

        # Filters
        vol_ok = pd.Series(True, index=df.index)
        if self.use_volume and has_volume:
            vol_ok = df["volume"] > df["vol_sma"]

        rsi_buy_ok = pd.Series(True, index=df.index)
        rsi_sell_ok = pd.Series(True, index=df.index)
        if self.use_rsi:
            rsi_buy_ok = df["rsi"] < self.rsi_overbought
            rsi_sell_ok = df["rsi"] > self.rsi_oversold

        candle_buy_ok = pd.Series(True, index=df.index)
        candle_sell_ok = pd.Series(True, index=df.index)
        if self.use_candle_confirm:
            candle_buy_ok = df["close"] > df["open"]
            candle_sell_ok = df["close"] < df["open"]

        slope_buy_ok = pd.Series(True, index=df.index)
        slope_sell_ok = pd.Series(True, index=df.index)
        if self.use_slope:
            slope_buy_ok = df["ema_slow"] > df["ema_slow"].shift(self.slope_lookback)
            slope_sell_ok = df["ema_slow"] < df["ema_slow"].shift(self.slope_lookback)

        sideways_ok = pd.Series(True, index=df.index)
        if self.filter_sideways:
            sideways_ok = ~df["is_sideways"]

        buy_f = vol_ok & rsi_buy_ok & candle_buy_ok & slope_buy_ok & sideways_ok
        sell_f = vol_ok & rsi_sell_ok & candle_sell_ok & slope_sell_ok & sideways_ok

        raw_buy = (df["bull_trend"] & df["cross_up_slow"] & buy_f) | \
                  (df["bull_trend"] & df["cross_up_fast"] & buy_f)
        raw_sell = (df["bear_trend"] & df["cross_down_slow"] & sell_f) | \
                   (df["bear_trend"] & df["cross_down_fast"] & sell_f)

        if self.use_cooldown:
            df["buy_signal"] = apply_cooldown(raw_buy.values, self.cooldown_bars)
            df["sell_signal"] = apply_cooldown(raw_sell.values, self.cooldown_bars)
        else:
            df["buy_signal"] = raw_buy
            df["sell_signal"] = raw_sell

        return df


# ╔══════════════════════════════════════════════════════════════╗
# ║       STRATEGY 2: GAUSSIAN CHANNEL v3.0 (simple)            ║
# ╚══════════════════════════════════════════════════════════════╝

class GaussianChannelStrategy(Strategy):
    """
    Long when close crosses above hband, close when close crosses below hband.
    100% equity, long only, signal-based exit.
    """
    name = "Gaussian Channel"
    commission_pct = 0.1

    def __init__(self, poles=4, period=144, mult=1.414,
                 reduced_lag=False, fast_response=False):
        self.poles = poles
        self.period = period
        self.mult = mult
        self.reduced_lag = reduced_lag
        self.fast_response = fast_response

    @property
    def exit_mode(self) -> str:
        return "signal"

    def config_label(self) -> str:
        parts = [f"Gauss(P{self.poles} T{self.period} M{self.mult})"]
        if self.reduced_lag: parts.append("LAG")
        if self.fast_response: parts.append("FAST")
        return " ".join(parts)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        filt, hband, lband = compute_gaussian_channel(
            df, self.poles, self.period, self.mult,
            self.reduced_lag, self.fast_response
        )
        df["gc_filt"] = filt
        df["gc_hband"] = hband
        df["gc_lband"] = lband

        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_hband = np.roll(hband, 1)
        prev_hband[0] = np.nan

        # Long entry: close crosses above hband (crossover)
        df["buy_signal"] = (prev_close <= prev_hband) & (close > hband)
        # Close position: close crosses below hband (crossunder)
        df["close_signal"] = (prev_close >= prev_hband) & (close < hband)
        df["sell_signal"] = False  # long only

        return df


# ╔══════════════════════════════════════════════════════════════╗
# ║   STRATEGY 3: GAUSSIAN CHANNEL v3.0 + StochRSI (AI)        ║
# ╚══════════════════════════════════════════════════════════════╝

class GaussianChannelStochRSIStrategy(Strategy):
    """
    Enhanced Gaussian Channel with StochRSI confirmation.
    Long when: channel is green (filt rising) AND close > hband AND stochRSI K > 80 or K < 20.
    Close when: close crosses below hband.
    """
    name = "Gaussian+StochRSI"
    commission_pct = 0.1

    def __init__(self, poles=4, period=144, mult=1.414,
                 reduced_lag=False, fast_response=False,
                 rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3):
        self.poles = poles
        self.period = period
        self.mult = mult
        self.reduced_lag = reduced_lag
        self.fast_response = fast_response
        self.rsi_len = rsi_len
        self.stoch_len = stoch_len
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d

    @property
    def exit_mode(self) -> str:
        return "signal"

    def config_label(self) -> str:
        parts = [f"Gauss+StochRSI(P{self.poles} T{self.period} M{self.mult})"]
        if self.reduced_lag: parts.append("LAG")
        if self.fast_response: parts.append("FAST")
        return " ".join(parts)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        filt, hband, lband = compute_gaussian_channel(
            df, self.poles, self.period, self.mult,
            self.reduced_lag, self.fast_response
        )
        df["gc_filt"] = filt
        df["gc_hband"] = hband
        df["gc_lband"] = lband

        k, d = compute_stoch_rsi(df["close"], self.rsi_len, self.stoch_len,
                                  self.smooth_k, self.smooth_d)
        df["stoch_k"] = k
        df["stoch_d"] = d

        # Channel is green = filter rising
        channel_green = np.zeros(len(filt), dtype=bool)
        channel_green[1:] = filt[1:] > filt[:-1]

        close = df["close"].values
        price_above_hband = close > hband

        # StochRSI extreme: K > 80 or K < 20
        stoch_extreme = (k.values > 80) | (k.values < 20)

        # Entry: all three conditions
        df["buy_signal"] = channel_green & price_above_hband & stoch_extreme

        # Exit: close crosses below hband
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_hband = np.roll(hband, 1)
        prev_hband[0] = np.nan
        df["close_signal"] = (prev_close >= prev_hband) & (close < hband)
        df["sell_signal"] = False

        return df


# ╔══════════════════════════════════════════════════════════════╗
# ║   STRATEGY 4: ANDEAN OSCILLATOR                             ║
# ╚══════════════════════════════════════════════════════════════╝

class AndeanOscillatorStrategy(Strategy):
    """
    Andean Oscillator by @alexgrover — translated from Pine Script.

    Three entry modes:
      'cross'    — Long when bull crosses above bear, close when bear crosses above bull.
      'signal'   — Long when bull > signal AND bull > bear, close when bull < signal.
      'momentum' — Long when bull rising AND bull > bear, close when bull starts falling.

    This is a momentum oscillator, so it works best with a trend-following
    exit rather than fixed SL/TP.
    """
    name = "Andean Oscillator"
    commission_pct = 0.1

    def __init__(self, length: int = 50, sig_length: int = 9,
                 mode: str = "cross"):
        self.length = length
        self.sig_length = sig_length
        self.mode = mode  # 'cross', 'signal', 'momentum'

    @property
    def exit_mode(self) -> str:
        return "signal"

    def config_label(self) -> str:
        return f"Andean(L{self.length} S{self.sig_length} {self.mode})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["close"].values
        open_ = df["open"].values

        bull, bear, signal = compute_andean_oscillator(
            close, open_, self.length, self.sig_length
        )
        df["ao_bull"] = bull
        df["ao_bear"] = bear
        df["ao_signal"] = signal

        n = len(close)
        buy = np.zeros(n, dtype=bool)
        close_sig = np.zeros(n, dtype=bool)

        if self.mode == "cross":
            # Bull crosses above bear → long; bear crosses above bull → close
            for i in range(1, n):
                if bull[i] > bear[i] and bull[i - 1] <= bear[i - 1]:
                    buy[i] = True
                if bear[i] > bull[i] and bear[i - 1] <= bull[i - 1]:
                    close_sig[i] = True

        elif self.mode == "signal":
            # Bull > signal AND bull > bear → long; bull < signal → close
            for i in range(1, n):
                if bull[i] > signal[i] and bull[i] > bear[i] and \
                   not (bull[i - 1] > signal[i - 1] and bull[i - 1] > bear[i - 1]):
                    buy[i] = True
                if bull[i] < signal[i] and bull[i - 1] >= signal[i - 1]:
                    close_sig[i] = True

        elif self.mode == "momentum":
            # Bull rising AND bull > bear → long; bull starts falling → close
            for i in range(1, n):
                bull_rising = bull[i] > bull[i - 1]
                was_not = not (bull[i - 1] > bear[i - 1] and (i < 2 or bull[i - 1] > bull[i - 2]))
                if bull_rising and bull[i] > bear[i] and was_not:
                    buy[i] = True
                if bull[i] < bull[i - 1] and bull[i - 1] >= bear[i - 1]:
                    close_sig[i] = True

        df["buy_signal"] = buy
        df["close_signal"] = close_sig
        df["sell_signal"] = False

        return df


# ╔══════════════════════════════════════════════════════════════╗
# ║               TRADE SIMULATION ENGINE                       ║
# ╚══════════════════════════════════════════════════════════════╝

def simulate_trades_sl_tp(df: pd.DataFrame, sl_atr_mult: float, rr_ratio: float,
                          direction: str = "both") -> list[dict]:
    """SL/TP based simulation for EMA Pullback style strategies."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    buy_sig = df["buy_signal"].values
    sell_sig = df["sell_signal"].values
    dates = df.index

    trades = []
    in_position = False
    side = ""
    entry_price = sl = tp = 0.0
    entry_date = None

    for i in range(len(close)):
        if in_position:
            if side == "long":
                if low[i] <= sl:
                    trades.append({"side": "long", "entry": entry_price, "exit": sl,
                                   "entry_date": entry_date, "exit_date": dates[i],
                                   "reason": "SL", "pnl_pct": (sl / entry_price - 1) * 100})
                    in_position = False
                elif high[i] >= tp:
                    trades.append({"side": "long", "entry": entry_price, "exit": tp,
                                   "entry_date": entry_date, "exit_date": dates[i],
                                   "reason": "TP", "pnl_pct": (tp / entry_price - 1) * 100})
                    in_position = False
            else:  # short
                if high[i] >= sl:
                    trades.append({"side": "short", "entry": entry_price, "exit": sl,
                                   "entry_date": entry_date, "exit_date": dates[i],
                                   "reason": "SL", "pnl_pct": (1 - sl / entry_price) * 100})
                    in_position = False
                elif low[i] <= tp:
                    trades.append({"side": "short", "entry": entry_price, "exit": tp,
                                   "entry_date": entry_date, "exit_date": dates[i],
                                   "reason": "TP", "pnl_pct": (1 - tp / entry_price) * 100})
                    in_position = False
            continue

        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        if buy_sig[i] and direction in ("both", "long"):
            sl_dist = a * sl_atr_mult
            entry_price = close[i]
            sl = entry_price - sl_dist
            tp = entry_price + sl_dist * rr_ratio
            entry_date = dates[i]
            side = "long"
            in_position = True
        elif sell_sig[i] and direction in ("both", "short"):
            sl_dist = a * sl_atr_mult
            entry_price = close[i]
            sl = entry_price + sl_dist
            tp = entry_price - sl_dist * rr_ratio
            entry_date = dates[i]
            side = "short"
            in_position = True

    # Close open position at last bar
    if in_position:
        pnl = (close[-1] / entry_price - 1) * 100 if side == "long" else (1 - close[-1] / entry_price) * 100
        trades.append({"side": side, "entry": entry_price, "exit": close[-1],
                       "entry_date": entry_date, "exit_date": dates[-1],
                       "reason": "OPEN", "pnl_pct": pnl})

    return trades


def simulate_trades_signal(df: pd.DataFrame) -> list[dict]:
    """Signal-based simulation for Gaussian Channel strategies (long only, close on signal)."""
    close = df["close"].values
    buy_sig = df["buy_signal"].values
    close_sig = df["close_signal"].values
    dates = df.index

    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None

    for i in range(len(close)):
        if in_position:
            if close_sig[i]:
                exit_price = close[i]
                pnl = (exit_price / entry_price - 1) * 100
                trades.append({"side": "long", "entry": entry_price, "exit": exit_price,
                               "entry_date": entry_date, "exit_date": dates[i],
                               "reason": "Signal", "pnl_pct": pnl})
                in_position = False
        else:
            if buy_sig[i]:
                entry_price = close[i]
                entry_date = dates[i]
                in_position = True

    # Close open position at last bar
    if in_position:
        pnl = (close[-1] / entry_price - 1) * 100
        trades.append({"side": "long", "entry": entry_price, "exit": close[-1],
                       "entry_date": entry_date, "exit_date": dates[-1],
                       "reason": "OPEN", "pnl_pct": pnl})

    return trades


# ╔══════════════════════════════════════════════════════════════╗
# ║              METRICS & EQUITY SIMULATION                    ║
# ╚══════════════════════════════════════════════════════════════╝

def calc_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
                "total_pnl": 0, "max_drawdown": 0, "expectancy": 0}

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    return {
        "total": len(pnls), "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins) / len(pnls) * 100, 1),
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "total_pnl": round(sum(pnls), 2),
        "max_drawdown": round(max_dd, 2),
        "expectancy": round(np.mean(pnls), 2),
    }


def simulate_equity(trades: list[dict], starting_capital: float,
                     commission_pct: float) -> dict:
    """
    Simulate compounding equity curve.
    Each trade uses 100% of current equity.
    Commission is applied on both entry and exit (round-trip = 2x).
    """
    equity = starting_capital
    peak = equity
    max_dd_pct = 0.0
    equity_curve = [equity]

    for t in trades:
        pnl_mult = 1.0 + (t["pnl_pct"] / 100.0)
        # Round-trip commission
        comm_mult = (1.0 - commission_pct / 100.0) ** 2
        equity *= pnl_mult * comm_mult
        equity_curve.append(equity)

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd

    return {
        "final_equity": round(equity, 2),
        "return_pct": round((equity / starting_capital - 1) * 100, 2),
        "return_x": round(equity / starting_capital, 2),
        "max_dd_equity_pct": round(max_dd_pct, 2),
        "equity_curve": equity_curve,
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║               RUN A STRATEGY                                ║
# ╚══════════════════════════════════════════════════════════════╝

def run_strategy(strategy: Strategy, df: pd.DataFrame, direction: str = "both",
                  starting_capital: float = 1000.0) -> dict:
    """Run a single strategy on a single symbol's data."""
    sig_df = strategy.generate_signals(df)

    if strategy.exit_mode == "sl_tp":
        # EMA Pullback uses SL/TP
        sl_mult = getattr(strategy, "sl_atr_mult", 2.0)
        rr = getattr(strategy, "rr_ratio", 2.0)
        trades = simulate_trades_sl_tp(sig_df, sl_mult, rr, direction)
    else:
        # Gaussian channel uses signal-based exit (long only)
        trades = simulate_trades_signal(sig_df)

    metrics = calc_metrics(trades)
    equity = simulate_equity(trades, starting_capital, strategy.commission_pct)

    return {
        "strategy": strategy.name,
        "config": strategy.config_label(),
        "trades": trades,
        "metrics": metrics,
        "equity": equity,
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║               OPTIMIZATION GRIDS                            ║
# ╚══════════════════════════════════════════════════════════════╝

def build_ema_optimization() -> list[EMAPullbackStrategy]:
    """Grid search for EMA Pullback."""
    configs = []
    grid = {
        "fast_len": [20, 50],
        "slow_len": [100, 200],
        "use_volume": [True, False],
        "use_candle_confirm": [True, False],
        "use_slope": [True, False],
        "use_rsi": [True, False],
        "sl_atr_mult": [1.5, 2.0, 3.0],
        "rr_ratio": [1.5, 2.0, 3.0],
    }
    keys = list(grid.keys())
    for combo in product(*grid.values()):
        kwargs = dict(zip(keys, combo))
        if kwargs["fast_len"] >= kwargs["slow_len"]:
            continue
        configs.append(EMAPullbackStrategy(**kwargs))
    return configs


def build_gaussian_optimization() -> list[GaussianChannelStrategy]:
    """Grid search for Gaussian Channel."""
    configs = []
    for poles in [3, 4, 5]:
        for period in [100, 144, 200]:
            for mult in [1.0, 1.414, 2.0]:
                for fast_resp in [False, True]:
                    configs.append(GaussianChannelStrategy(
                        poles=poles, period=period, mult=mult,
                        fast_response=fast_resp))
    return configs


def build_gaussian_stoch_optimization() -> list[GaussianChannelStochRSIStrategy]:
    """Grid search for Gaussian+StochRSI."""
    configs = []
    for poles in [3, 4, 5]:
        for period in [100, 144, 200]:
            for mult in [1.0, 1.414, 2.0]:
                for fast_resp in [False, True]:
                    configs.append(GaussianChannelStochRSIStrategy(
                        poles=poles, period=period, mult=mult,
                        fast_response=fast_resp))
    return configs


def build_andean_optimization() -> list[AndeanOscillatorStrategy]:
    """Grid search for Andean Oscillator."""
    configs = []
    for length in [20, 34, 50, 80]:
        for sig_length in [5, 9, 14, 21]:
            for mode in ["cross", "signal", "momentum"]:
                configs.append(AndeanOscillatorStrategy(
                    length=length, sig_length=sig_length, mode=mode))
    return configs


# ╔══════════════════════════════════════════════════════════════╗
# ║                    MAIN                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy Backtester — $1K to $1M Challenge")
    parser.add_argument("--api-key", default=os.environ.get("TWELVE_DATA_API_KEY"),
                        help="Twelve Data API key")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS)
    parser.add_argument("--direction", default="both", choices=["long", "short", "both"])
    parser.add_argument("--capital", type=float, default=1000.0,
                        help="Starting capital (default: $1,000)")
    parser.add_argument("--target", type=float, default=1_000_000.0,
                        help="Target equity (default: $1,000,000)")
    parser.add_argument("--optimize", action="store_true",
                        help="Grid search over strategy parameters (finds best config per strategy per symbol)")
    parser.add_argument("--strategy", nargs="+",
                        choices=["ema", "gaussian", "gaussian-stoch", "andean", "all"],
                        default=["all"],
                        help="Which strategies to test")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key. Set TWELVE_DATA_API_KEY or pass --api-key")
        sys.exit(1)

    td = TDClient(apikey=args.api_key)

    # ── Fetch data ──────────────────────────────────────────────
    datasets: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(args.symbols):
        try:
            print(f"  Fetching {sym} ({args.interval}, {args.bars} bars)...")
            datasets[sym] = fetch_ohlcv(td, sym, args.interval, args.bars)
            print(f"    → {len(datasets[sym])} bars loaded "
                  f"({datasets[sym].index[0]} to {datasets[sym].index[-1]})")
            if i < len(args.symbols) - 1:
                time.sleep(9)
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    if not datasets:
        print("\nNo data loaded. Exiting.")
        sys.exit(1)

    # ── Build strategy list ─────────────────────────────────────
    strat_names = set(args.strategy)
    if "all" in strat_names:
        strat_names = {"ema", "gaussian", "gaussian-stoch", "andean"}

    if args.optimize:
        strategies_map = {}
        if "ema" in strat_names:
            strategies_map["EMA Pullback"] = build_ema_optimization()
        if "gaussian" in strat_names:
            strategies_map["Gaussian Channel"] = build_gaussian_optimization()
        if "gaussian-stoch" in strat_names:
            strategies_map["Gaussian+StochRSI"] = build_gaussian_stoch_optimization()
        if "andean" in strat_names:
            strategies_map["Andean Oscillator"] = build_andean_optimization()

        total_configs = sum(len(v) for v in strategies_map.values())
        print(f"\n  Optimization mode: {total_configs} configs × {len(datasets)} symbols\n")
    else:
        # Default configs
        strategies_map = {}
        if "ema" in strat_names:
            strategies_map["EMA Pullback"] = [EMAPullbackStrategy()]
        if "gaussian" in strat_names:
            strategies_map["Gaussian Channel"] = [GaussianChannelStrategy()]
        if "gaussian-stoch" in strat_names:
            strategies_map["Gaussian+StochRSI"] = [GaussianChannelStochRSIStrategy()]
        if "andean" in strat_names:
            strategies_map["Andean Oscillator"] = [AndeanOscillatorStrategy()]

    # ── Run all strategies ──────────────────────────────────────
    all_results = []  # list of dicts for summary table

    for strat_group_name, strat_list in strategies_map.items():
        for sym, df in datasets.items():
            best_equity = -float("inf")
            best_result = None

            for strat in strat_list:
                result = run_strategy(strat, df, args.direction, args.capital)
                result["symbol"] = sym

                if args.optimize:
                    # Best by final equity (min 3 trades)
                    if result["metrics"]["total"] >= 3 and \
                       result["equity"]["final_equity"] > best_equity:
                        best_equity = result["equity"]["final_equity"]
                        best_result = result
                else:
                    all_results.append(result)

            if args.optimize and best_result:
                all_results.append(best_result)

    # ── Print results ───────────────────────────────────────────
    if not all_results:
        print("No trades generated. Exiting.")
        sys.exit(0)

    print("\n" + "=" * 130)
    print("  MULTI-STRATEGY BACKTESTER — RESULTS")
    print("=" * 130)
    print(f"  Interval: {args.interval}  |  Direction: {args.direction}  "
          f"|  Capital: ${args.capital:,.0f}  |  Target: ${args.target:,.0f}  "
          f"|  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if args.optimize:
        print("  Mode: OPTIMIZATION (best config per strategy per symbol)")
    print("=" * 130 + "\n")

    headers = ["Strategy", "Symbol", "Config", "Trades", "Win %", "PF",
               "Total PnL %", "Max DD %", "Expect %",
               f"${args.capital:,.0f} →", "Return X", "Equity DD %"]

    rows = []
    for r in all_results:
        m = r["metrics"]
        e = r["equity"]
        rows.append([
            r["strategy"], r["symbol"], r["config"],
            m["total"], f"{m['win_rate']}%", m["profit_factor"],
            f"{m['total_pnl']}%", f"{m['max_drawdown']}%", f"{m['expectancy']}%",
            f"${e['final_equity']:,.2f}", f"{e['return_x']}x", f"{e['max_dd_equity_pct']}%",
        ])

    # Sort by final equity descending
    rows.sort(key=lambda x: float(x[9].replace("$", "").replace(",", "")), reverse=True)
    print(tabulate(rows, headers=headers, tablefmt="pretty", stralign="right"))

    # ── $1K → $1M analysis ──────────────────────────────────────
    print(f"\n{'─' * 130}")
    print(f"  💰 $1K → $1M ANALYSIS (which strategies compound best?)")
    print(f"{'─' * 130}\n")

    # Group by strategy+symbol, show top performers
    sorted_results = sorted(all_results, key=lambda r: r["equity"]["final_equity"], reverse=True)

    for i, r in enumerate(sorted_results[:10]):
        e = r["equity"]
        m = r["metrics"]
        final = e["final_equity"]
        target_hit = "✅ TARGET HIT!" if final >= args.target else ""
        multiplier = final / args.capital

        # Estimate how many cycles of same performance needed to reach target
        if multiplier > 1.0:
            cycles_needed = np.log(args.target / args.capital) / np.log(multiplier)
            cycle_note = f"  ({cycles_needed:.1f} cycles of this data to reach ${args.target:,.0f})"
        else:
            cycle_note = "  (losing strategy — won't reach target)"

        print(f"  #{i+1}  {r['strategy']:20s} | {r['symbol']:8s} | "
              f"${args.capital:,.0f} → ${final:>14,.2f}  ({multiplier:>8,.2f}x)  "
              f"|  {m['total']} trades  |  DD {e['max_dd_equity_pct']}%  "
              f"{target_hit}{cycle_note}")

    # ── Legend ───────────────────────────────────────────────────
    print(f"\n{'─' * 130}")
    print("  LEGEND")
    print("  PF        = Profit Factor (gross profit / gross loss, > 1.5 is strong)")
    print("  Max DD %  = Max cumulative drawdown in PnL % terms")
    print("  Equity DD = Max peak-to-trough drawdown on the compound equity curve")
    print("  Expect %  = Average PnL per trade")
    print("  Return X  = Final equity / starting capital (compound)")
    print("  Cycles    = How many times you'd need to repeat this exact data to hit target")
    print(f"\n  ⚠️  Past performance ≠ future results. These are backtests, not guarantees.")
    print(f"  ⚠️  Slippage, fees, and liquidity may differ in live trading.\n")


if __name__ == "__main__":
    main()
