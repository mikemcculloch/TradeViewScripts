"""
EMA Pullback Strategy — Multi-Asset Backtester
================================================
Uses the Twelve Data API to pull OHLCV data for your assets and test
the same EMA pullback logic from the PineScript strategy.

Analyzes which filter combinations work best per asset and generates
a performance report (win rate, profit factor, max drawdown, etc.)

Usage:
    1.  pip install -r requirements.txt
    2.  set TWELVE_DATA_API_KEY=your_key_here       (or pass via --api-key)
    3.  python backtest.py
    4.  python backtest.py --symbols BTCUSD ETHUSD --interval 1h
    5.  python backtest.py --optimize              (grid search over filter combos)

Created By: Wooanaz
Created On: 3/6/2026
"""

import argparse
import os
import sys
import time
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
    "BTC/USD", "ETH/USD", "XRP/USD",   # crypto
    "GOOGL", "HIMS", "PLTR",            # stocks
]

DEFAULT_INTERVAL = "1day"
DEFAULT_BARS = 1000  # ~4 yrs of daily bars


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

    df = df.sort_index()  # oldest first
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Some Twelve Data tiers don't include volume (e.g. crypto)
    if "volume" not in df.columns:
        df["volume"] = 0
        df.attrs["has_volume"] = False
    else:
        df.attrs["has_volume"] = True
    df.dropna(subset=["close"], inplace=True)
    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║                    INDICATOR CALCULATIONS                   ║
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


# ╔══════════════════════════════════════════════════════════════╗
# ║                    SIGNAL GENERATION                        ║
# ╚══════════════════════════════════════════════════════════════╝

class FilterConfig:
    """Holds all filter toggle/param choices for a single backtest run."""

    def __init__(self, **kwargs):
        # EMA
        self.fast_len: int = kwargs.get("fast_len", 50)
        self.slow_len: int = kwargs.get("slow_len", 200)

        # Filters (core)
        self.use_cooldown: bool = kwargs.get("use_cooldown", True)
        self.cooldown_bars: int = kwargs.get("cooldown_bars", 3)
        self.filter_sideways: bool = kwargs.get("filter_sideways", True)
        self.sideways_threshold: float = kwargs.get("sideways_threshold", 0.005)
        self.use_volume: bool = kwargs.get("use_volume", True)
        self.use_candle_confirm: bool = kwargs.get("use_candle_confirm", True)
        self.use_slope: bool = kwargs.get("use_slope", True)
        self.slope_lookback: int = kwargs.get("slope_lookback", 10)

        # Optional
        self.use_rsi: bool = kwargs.get("use_rsi", False)
        self.rsi_overbought: float = kwargs.get("rsi_overbought", 70.0)
        self.rsi_oversold: float = kwargs.get("rsi_oversold", 30.0)

        # Risk management
        self.sl_atr_mult: float = kwargs.get("sl_atr_mult", 2.0)
        self.rr_ratio: float = kwargs.get("rr_ratio", 2.0)

        # Exit strategy: "sl_tp" (legacy SL/TP only), "death_cross", "fast_ema", "chandelier", "chandelier_fast_ema"
        self.exit_mode: str = kwargs.get("exit_mode", "sl_tp")
        self.chandelier_atr_len: int = kwargs.get("chandelier_atr_len", 22)
        self.chandelier_atr_mult: float = kwargs.get("chandelier_atr_mult", 3.0)
        self.exit_bars_below: int = kwargs.get("exit_bars_below", 1)

    def short_name(self) -> str:
        parts = [f"EMA({self.fast_len}/{self.slow_len})"]
        if self.use_cooldown:
            parts.append("CD")
        if self.filter_sideways:
            parts.append("SW")
        if self.use_volume:
            parts.append("VOL")
        if self.use_candle_confirm:
            parts.append("CC")
        if self.use_slope:
            parts.append("SLP")
        if self.use_rsi:
            parts.append("RSI")
        parts.append(f"SL{self.sl_atr_mult}x")
        parts.append(f"RR{self.rr_ratio}")
        exit_labels = {
            "sl_tp": "SL/TP",
            "death_cross": "DeathX",
            "fast_ema": "FastEMA",
            "chandelier": "Chand",
            "chandelier_fast_ema": "Chand+FE",
        }
        parts.append(exit_labels.get(self.exit_mode, self.exit_mode))
        return " ".join(parts)


def generate_signals(df: pd.DataFrame, cfg: FilterConfig) -> pd.DataFrame:
    """
    Apply the EMA pullback signal logic to a DataFrame.
    Returns a copy with signal columns added.
    """
    df = df.copy()

    # EMAs
    df["ema_fast"] = compute_ema(df["close"], cfg.fast_len)
    df["ema_slow"] = compute_ema(df["close"], cfg.slow_len)
    df["rsi"] = compute_rsi(df["close"], 14)
    df["atr"] = compute_atr(df, 14)
    has_volume = df.attrs.get("has_volume", df["volume"].sum() > 0)
    if has_volume:
        df["vol_sma"] = df["volume"].rolling(20).mean()
    else:
        df["vol_sma"] = 0

    # Trend
    df["bull_trend"] = df["ema_fast"] > df["ema_slow"]
    df["bear_trend"] = df["ema_fast"] < df["ema_slow"]

    # Sideways
    df["ema_spread"] = (df["ema_fast"] - df["ema_slow"]).abs() / df["ema_slow"]
    df["is_sideways"] = df["ema_spread"] < cfg.sideways_threshold

    # Price crosses slow EMA
    df["prev_below_slow"] = df["close"].shift(1) < df["ema_slow"].shift(1)
    df["prev_above_slow"] = df["close"].shift(1) > df["ema_slow"].shift(1)
    df["cross_up_slow"] = df["prev_below_slow"] & (df["close"] > df["ema_slow"])
    df["cross_down_slow"] = df["prev_above_slow"] & (df["close"] < df["ema_slow"])

    # Price crosses fast EMA
    df["prev_below_fast"] = df["close"].shift(1) < df["ema_fast"].shift(1)
    df["prev_above_fast"] = df["close"].shift(1) > df["ema_fast"].shift(1)
    df["cross_up_fast"] = df["prev_below_fast"] & (df["close"] > df["ema_fast"])
    df["cross_down_fast"] = df["prev_above_fast"] & (df["close"] < df["ema_fast"])

    # --- Filter conditions ---
    # Volume (skip if data has no volume)
    df["vol_ok"] = True
    if cfg.use_volume and has_volume:
        df["vol_ok"] = df["volume"] > df["vol_sma"]

    # RSI
    df["rsi_buy_ok"] = True
    df["rsi_sell_ok"] = True
    if cfg.use_rsi:
        df["rsi_buy_ok"] = df["rsi"] < cfg.rsi_overbought
        df["rsi_sell_ok"] = df["rsi"] > cfg.rsi_oversold

    # Candle confirm
    df["candle_buy_ok"] = True
    df["candle_sell_ok"] = True
    if cfg.use_candle_confirm:
        df["candle_buy_ok"] = df["close"] > df["open"]
        df["candle_sell_ok"] = df["close"] < df["open"]

    # Slope
    df["slope_buy_ok"] = True
    df["slope_sell_ok"] = True
    if cfg.use_slope:
        df["slope_buy_ok"] = df["ema_slow"] > df["ema_slow"].shift(cfg.slope_lookback)
        df["slope_sell_ok"] = df["ema_slow"] < df["ema_slow"].shift(cfg.slope_lookback)

    # Sideways
    df["sideways_ok"] = True
    if cfg.filter_sideways:
        df["sideways_ok"] = ~df["is_sideways"]

    # --- Raw signals (before cooldown) ---
    buy_filters = df["vol_ok"] & df["rsi_buy_ok"] & df["candle_buy_ok"] & df["slope_buy_ok"] & df["sideways_ok"]
    sell_filters = df["vol_ok"] & df["rsi_sell_ok"] & df["candle_sell_ok"] & df["slope_sell_ok"] & df["sideways_ok"]

    df["raw_buy_slow"] = df["bull_trend"] & df["cross_up_slow"] & buy_filters
    df["raw_buy_fast"] = df["bull_trend"] & df["cross_up_fast"] & buy_filters
    df["raw_sell_slow"] = df["bear_trend"] & df["cross_down_slow"] & sell_filters
    df["raw_sell_fast"] = df["bear_trend"] & df["cross_down_fast"] & sell_filters

    df["raw_buy"] = df["raw_buy_slow"] | df["raw_buy_fast"]
    df["raw_sell"] = df["raw_sell_slow"] | df["raw_sell_fast"]

    # --- Apply cooldown (vectorized) ---
    if cfg.use_cooldown:
        df["buy_signal"] = _apply_cooldown(df["raw_buy"].values, cfg.cooldown_bars)
        df["sell_signal"] = _apply_cooldown(df["raw_sell"].values, cfg.cooldown_bars)
    else:
        df["buy_signal"] = df["raw_buy"]
        df["sell_signal"] = df["raw_sell"]

    return df


def _apply_cooldown(signals: np.ndarray, cooldown: int) -> np.ndarray:
    """Fast cooldown using numpy arrays instead of row-by-row pandas."""
    result = np.zeros(len(signals), dtype=bool)
    last_idx = -999
    for i in range(len(signals)):
        if signals[i] and (i - last_idx > cooldown):
            result[i] = True
            last_idx = i
    return result


# ╔══════════════════════════════════════════════════════════════╗
# ║                    TRADE SIMULATION                         ║
# ╚══════════════════════════════════════════════════════════════╝

def simulate_trades(df: pd.DataFrame, cfg: FilterConfig, trade_direction: str = "both") -> list[dict]:
    """
    Simulate trades with configurable exit strategies:
      - sl_tp:              SL/TP only (legacy)
      - death_cross:        Exit when fast EMA crosses below slow EMA (+ SL)
      - fast_ema:           Exit on N consecutive closes below fast EMA (+ SL)
      - chandelier:         ATR trailing stop from highest high since entry (+ SL)
      - chandelier_fast_ema: First of chandelier or fast_ema to fire (+ SL)
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    ema_fast = df["ema_fast"].values
    ema_slow = df["ema_slow"].values
    buy_sig = df["buy_signal"].values
    sell_sig = df["sell_signal"].values
    dates = df.index

    # Pre-compute chandelier ATR (different length than entry ATR)
    chandelier_atr = compute_atr(df, cfg.chandelier_atr_len).values

    mode = cfg.exit_mode
    use_chandelier = mode in ("chandelier", "chandelier_fast_ema")
    use_fast_ema = mode in ("fast_ema", "chandelier_fast_ema")
    use_death_cross = mode == "death_cross"
    use_tp = mode == "sl_tp"  # TP only in legacy mode

    trades = []
    in_position = False
    side = ""
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    entry_date = None
    highest_since_entry = 0.0
    lowest_since_entry = float("inf")
    bars_below_fast = 0
    bars_above_fast = 0

    for i in range(len(close)):
        if in_position:
            # --- Tracking for dynamic exits ---
            if side == "long":
                highest_since_entry = max(highest_since_entry, high[i])
                if close[i] < ema_fast[i]:
                    bars_below_fast += 1
                else:
                    bars_below_fast = 0
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
                if close[i] > ema_fast[i]:
                    bars_above_fast += 1
                else:
                    bars_above_fast = 0

            # --- Check exits (priority: SL first, then strategy exit, then TP) ---
            exited = False

            if side == "long":
                # 1. Hard stop loss always active
                if low[i] <= sl:
                    _append_trade(trades, "long", entry_price, sl, entry_date, dates[i], "SL")
                    exited = True
                # 2. Take profit (only in sl_tp mode)
                elif use_tp and high[i] >= tp:
                    _append_trade(trades, "long", entry_price, tp, entry_date, dates[i], "TP")
                    exited = True
                # 3. Death cross
                elif use_death_cross and i > 0 and ema_fast[i] < ema_slow[i] and ema_fast[i - 1] >= ema_slow[i - 1]:
                    _append_trade(trades, "long", entry_price, close[i], entry_date, dates[i], "DeathX")
                    exited = True
                else:
                    # 4. Chandelier trailing stop
                    chand_exit = False
                    if use_chandelier and not np.isnan(chandelier_atr[i]):
                        chand_stop = highest_since_entry - chandelier_atr[i] * cfg.chandelier_atr_mult
                        if close[i] < chand_stop:
                            _append_trade(trades, "long", entry_price, close[i], entry_date, dates[i], "Chand")
                            exited = True
                            chand_exit = True
                    # 5. Close below fast EMA
                    if not chand_exit and use_fast_ema and bars_below_fast >= cfg.exit_bars_below:
                        _append_trade(trades, "long", entry_price, close[i], entry_date, dates[i], "FastEMA")
                        exited = True
            else:  # short
                if high[i] >= sl:
                    _append_trade(trades, "short", entry_price, sl, entry_date, dates[i], "SL")
                    exited = True
                elif use_tp and low[i] <= tp:
                    _append_trade(trades, "short", entry_price, tp, entry_date, dates[i], "TP")
                    exited = True
                elif use_death_cross and i > 0 and ema_fast[i] > ema_slow[i] and ema_fast[i - 1] <= ema_slow[i - 1]:
                    _append_trade(trades, "short", entry_price, close[i], entry_date, dates[i], "GoldenX")
                    exited = True
                else:
                    chand_exit = False
                    if use_chandelier and not np.isnan(chandelier_atr[i]):
                        chand_stop = lowest_since_entry + chandelier_atr[i] * cfg.chandelier_atr_mult
                        if close[i] > chand_stop:
                            _append_trade(trades, "short", entry_price, close[i], entry_date, dates[i], "Chand")
                            exited = True
                            chand_exit = True
                    if not chand_exit and use_fast_ema and bars_above_fast >= cfg.exit_bars_below:
                        _append_trade(trades, "short", entry_price, close[i], entry_date, dates[i], "FastEMA")
                        exited = True

            if exited:
                in_position = False
            continue

        # --- Entry logic ---
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        if buy_sig[i] and trade_direction in ("both", "long"):
            sl_dist = a * cfg.sl_atr_mult
            entry_price = close[i]
            sl = entry_price - sl_dist
            tp = entry_price + sl_dist * cfg.rr_ratio
            entry_date = dates[i]
            side = "long"
            in_position = True
            highest_since_entry = high[i]
            bars_below_fast = 0
        elif sell_sig[i] and trade_direction in ("both", "short"):
            sl_dist = a * cfg.sl_atr_mult
            entry_price = close[i]
            sl = entry_price + sl_dist
            tp = entry_price - sl_dist * cfg.rr_ratio
            entry_date = dates[i]
            side = "short"
            in_position = True
            lowest_since_entry = low[i]
            bars_above_fast = 0

    if in_position:
        pnl = (close[-1] / entry_price - 1) * 100 if side == "long" else (1 - close[-1] / entry_price) * 100
        trades.append({"side": side, "entry_price": entry_price, "entry_date": entry_date,
                       "exit_price": close[-1], "exit_date": dates[-1], "exit_reason": "OPEN",
                       "pnl_pct": pnl})

    return trades


def _append_trade(trades: list, side: str, entry_price: float, exit_price: float,
                  entry_date, exit_date, reason: str):
    """Helper to append a trade with correct PnL calculation."""
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
# ║                    PERFORMANCE METRICS                      ║
# ╚══════════════════════════════════════════════════════════════╝

def calc_metrics(trades: list[dict]) -> dict:
    """Calculate performance metrics from a list of trades."""
    if not trades:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
            "total_pnl": 0, "max_drawdown": 0, "expectancy": 0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    # Max drawdown (on cumulative PnL curve)
    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    total = len(pnls)
    n_wins = len(wins)
    n_losses = len(losses)

    return {
        "total": total,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": round(n_wins / total * 100, 1) if total > 0 else 0,
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "total_pnl": round(sum(pnls), 2),
        "max_drawdown": round(max_dd, 2),
        "expectancy": round(np.mean(pnls), 2) if pnls else 0,
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║                    OPTIMIZATION GRID                        ║
# ╚══════════════════════════════════════════════════════════════╝

def build_optimization_configs() -> list[FilterConfig]:
    """Generate a grid of filter configurations to test."""
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
        # Skip nonsensical: fast must be < slow
        if kwargs["fast_len"] >= kwargs["slow_len"]:
            continue
        configs.append(FilterConfig(**kwargs))

    return configs


# ╔══════════════════════════════════════════════════════════════╗
# ║                    MAIN                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="EMA Pullback Strategy Backtester")
    parser.add_argument("--api-key", default=os.environ.get("TWELVE_DATA_API_KEY"),
                        help="Twelve Data API key (or set TWELVE_DATA_API_KEY env var)")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Symbols to test (e.g. BTC/USD GOOGL PLTR)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL,
                        help="Candle interval (1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week)")
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS,
                        help="Number of bars to fetch per symbol")
    parser.add_argument("--optimize", action="store_true",
                        help="Run grid search over filter/parameter combinations")
    parser.add_argument("--direction", default="both", choices=["long", "short", "both"],
                        help="Trade direction")
    parser.add_argument("--exit-mode", default="sl_tp",
                        choices=["sl_tp", "death_cross", "fast_ema", "chandelier", "chandelier_fast_ema"],
                        help="Exit strategy (sl_tp=legacy SL/TP, death_cross, fast_ema, chandelier, chandelier_fast_ema)")
    parser.add_argument("--compare-exits", action="store_true",
                        help="Run all 5 exit strategies side-by-side for comparison")
    parser.add_argument("--chandelier-len", type=int, default=22,
                        help="ATR length for Chandelier exit (default: 22)")
    parser.add_argument("--chandelier-mult", type=float, default=3.0,
                        help="ATR multiplier for Chandelier exit (default: 3.0)")
    parser.add_argument("--exit-bars", type=int, default=1,
                        help="Consecutive bars below fast EMA to trigger exit (default: 1)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key provided.")
        print("  Set environment variable:  set TWELVE_DATA_API_KEY=your_key")
        print("  Or pass via CLI:           python backtest.py --api-key your_key")
        sys.exit(1)

    td = TDClient(apikey=args.api_key)

    # ── Fetch data ──────────────────────────────────────────────
    datasets: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(args.symbols):
        try:
            print(f"  Fetching {sym} ({args.interval}, {args.bars} bars)...")
            datasets[sym] = fetch_ohlcv(td, sym, args.interval, args.bars)
            print(f"    → {len(datasets[sym])} bars loaded ({datasets[sym].index[0]} to {datasets[sym].index[-1]})")
            # Rate limit: Twelve Data free tier = 8 calls/min
            if i < len(args.symbols) - 1:
                time.sleep(9)  # ~6.6 calls/min to stay safely under 8
        except Exception as e:
            print(f"    ✗ Failed to fetch {sym}: {e}")

    if not datasets:
        print("\nNo data loaded. Exiting.")
        sys.exit(1)

    # ── Choose configs ──────────────────────────────────────────
    exit_modes_to_test = ["sl_tp", "death_cross", "fast_ema", "chandelier", "chandelier_fast_ema"] if args.compare_exits else [args.exit_mode]

    if args.optimize:
        base_configs = build_optimization_configs()
        print(f"\n  Optimization mode: testing {len(base_configs)} configurations per symbol\n")
    else:
        base_configs = [FilterConfig()]

    # Inject exit params into all configs
    configs_per_exit = {}
    for em in exit_modes_to_test:
        cfgs = []
        for bc in base_configs:
            cfg = FilterConfig(
                fast_len=bc.fast_len, slow_len=bc.slow_len,
                use_cooldown=bc.use_cooldown, cooldown_bars=bc.cooldown_bars,
                filter_sideways=bc.filter_sideways, sideways_threshold=bc.sideways_threshold,
                use_volume=bc.use_volume, use_candle_confirm=bc.use_candle_confirm,
                use_slope=bc.use_slope, slope_lookback=bc.slope_lookback,
                use_rsi=bc.use_rsi, rsi_overbought=bc.rsi_overbought, rsi_oversold=bc.rsi_oversold,
                sl_atr_mult=bc.sl_atr_mult, rr_ratio=bc.rr_ratio,
                exit_mode=em,
                chandelier_atr_len=args.chandelier_len,
                chandelier_atr_mult=args.chandelier_mult,
                exit_bars_below=args.exit_bars,
            )
            cfgs.append(cfg)
        configs_per_exit[em] = cfgs

    if not args.optimize and not args.compare_exits:
        print(f"\n  Single run with config: {configs_per_exit[exit_modes_to_test[0]][0].short_name()}\n")
    elif args.compare_exits:
        print(f"\n  Comparing {len(exit_modes_to_test)} exit strategies per symbol\n")

    # ── Run backtest ────────────────────────────────────────────
    all_results = []

    for sym, df in datasets.items():
        for em, configs in configs_per_exit.items():
            best_pf = -1
            best_result = None

            for cfg in configs:
                sig_df = generate_signals(df, cfg)
                trades = simulate_trades(sig_df, cfg, args.direction)
                metrics = calc_metrics(trades)
                metrics["symbol"] = sym
                metrics["config"] = cfg.short_name()

                if args.optimize:
                    if metrics["total"] >= 5 and metrics["profit_factor"] > best_pf:
                        best_pf = metrics["profit_factor"]
                        best_result = metrics
                else:
                    all_results.append(metrics)

            if args.optimize and best_result:
                all_results.append(best_result)

    # ── Print results ───────────────────────────────────────────
    if not all_results:
        print("No trades generated across all symbols/configs.")
        sys.exit(0)

    print("\n" + "=" * 100)
    print("  EMA PULLBACK STRATEGY — BACKTEST RESULTS")
    print("=" * 100)
    print(f"  Interval: {args.interval}  |  Direction: {args.direction}  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if args.optimize:
        print("  Mode: OPTIMIZATION (showing best config per symbol)")
    print("=" * 100 + "\n")

    headers = ["Symbol", "Config", "Trades", "Wins", "Losses", "Win %",
               "Avg Win %", "Avg Loss %", "PF", "Total PnL %", "Max DD %", "Expect %"]

    rows = []
    for r in all_results:
        rows.append([
            r["symbol"], r["config"], r["total"], r["wins"], r["losses"],
            f"{r['win_rate']}%", f"{r['avg_win']}%", f"{r['avg_loss']}%",
            r["profit_factor"], f"{r['total_pnl']}%", f"{r['max_drawdown']}%",
            f"{r['expectancy']}%",
        ])

    print(tabulate(rows, headers=headers, tablefmt="pretty", stralign="right"))

    # ── Summary ─────────────────────────────────────────────────
    print("\n--- LEGEND ---")
    print("  PF     = Profit Factor (gross profit / gross loss, > 1.5 is strong)")
    print("  Max DD = Max cumulative drawdown in % PnL terms")
    print("  Expect = Average PnL per trade (positive = edge)")
    print("  CD=Cooldown  SW=Sideways  VOL=Volume  CC=Candle Confirm  SLP=Slope  RSI=RSI Filter")
    print()

    if args.optimize:
        print("  TIP: Take the best config for each asset and set those filters in TradingView.")
        print("       Different assets may prefer different filter combinations!\n")


if __name__ == "__main__":
    main()
