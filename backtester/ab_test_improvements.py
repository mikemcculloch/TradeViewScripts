"""
A/B Test Framework — Systematically test risk management improvements
=====================================================================
Tests each proposed improvement against baseline for all 3 strategies
across all 9 assets. Keeps only what improves profitability / reduces DD.

Assets:
  Crypto (Bitunix): BTC/USD, ETH/USD, XRP/USD, ALGO/USD
  Non-Crypto:       HIMS, GOOGL, GLD (Gold), SLV (Silver), USO (Oil)

Strategies:
  1. EMA Pullback         (SL/TP exit)
  2. Gaussian+StochRSI    (signal exit)
  3. Andean Oscillator    (signal exit)

Improvements tested (one at a time):
  A. ATR Stop Loss        — add 2x ATR SL to signal strategies
  B. ATR Take Profit      — add 3x ATR TP
  C. Trailing Stop        — 1.5x ATR trail
  D. Position Sizing      — 50% and 75% instead of 100%
  E. Re-entry Cooldown    — min 3 bars between entries
  F. HTF Trend Filter     — requires price > 200 EMA

Created By: Wooanaz
Created On: 3/9/2026
"""

import os
import sys
import time
import copy
import numpy as np
import pandas as pd
from tabulate import tabulate

# Import everything from multi_backtest
sys.path.insert(0, os.path.dirname(__file__))
from multi_backtest import (
    TDClient, fetch_ohlcv,
    compute_ema, compute_atr, compute_rsi,
    EMAPullbackStrategy, GaussianChannelStochRSIStrategy, AndeanOscillatorStrategy,
    simulate_trades_sl_tp, simulate_trades_signal,
    calc_metrics, simulate_equity, run_strategy,
)

# ── Assets ──────────────────────────────────────────────────────
SYMBOLS = [
    "BTC/USD", "ETH/USD", "XRP/USD", "ALGO/USD",
    "HIMS", "GOOGL", "GLD", "SLV", "USO",
]

INTERVAL = "1day"
BARS = 1000
CAPITAL = 1000.0


# ╔══════════════════════════════════════════════════════════════╗
# ║           ENHANCED SIMULATION: SL + TP + TRAILING           ║
# ╚══════════════════════════════════════════════════════════════╝

def simulate_trades_signal_with_risk(df: pd.DataFrame,
                                      use_sl: bool = False, sl_atr_mult: float = 2.0,
                                      use_tp: bool = False, tp_atr_mult: float = 3.0,
                                      use_trail: bool = False, trail_atr_mult: float = 1.5,
                                      ) -> list[dict]:
    """
    Signal-based entry with optional ATR stop loss, take profit, and trailing stop.
    Long only. Exits on close_signal OR SL/TP/trail hit (whichever comes first).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    buy_sig = df["buy_signal"].values
    close_sig = df["close_signal"].values
    dates = df.index

    # Pre-compute ATR if needed
    atr = compute_atr(df, 14).values if (use_sl or use_tp or use_trail) else None

    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    sl_price = 0.0
    tp_price = 0.0
    trail_price = 0.0

    for i in range(len(close)):
        if in_position:
            exit_price = None
            reason = None

            # Check SL first (worst case)
            if use_sl and low[i] <= sl_price:
                exit_price = sl_price
                reason = "SL"
            # Check TP
            elif use_tp and high[i] >= tp_price:
                exit_price = tp_price
                reason = "TP"
            # Check trailing stop
            elif use_trail:
                # Update trail: highest close since entry minus trail distance
                current_atr = atr[i] if not np.isnan(atr[i]) else atr[i-1]
                new_trail = high[i] - current_atr * trail_atr_mult
                if new_trail > trail_price:
                    trail_price = new_trail
                if low[i] <= trail_price:
                    exit_price = trail_price
                    reason = "Trail"

            # Check signal exit (if no SL/TP/trail hit)
            if exit_price is None and close_sig[i]:
                exit_price = close[i]
                reason = "Signal"

            if exit_price is not None:
                # Clamp exit price to reasonable range
                exit_price = max(exit_price, low[i])
                exit_price = min(exit_price, high[i])
                pnl = (exit_price / entry_price - 1) * 100
                trades.append({"side": "long", "entry": entry_price, "exit": exit_price,
                               "entry_date": entry_date, "exit_date": dates[i],
                               "reason": reason, "pnl_pct": pnl})
                in_position = False

        else:
            if buy_sig[i]:
                entry_price = close[i]
                entry_date = dates[i]
                in_position = True

                a = atr[i] if atr is not None and not np.isnan(atr[i]) else 0
                sl_price = entry_price - a * sl_atr_mult if use_sl and a > 0 else 0
                tp_price = entry_price + a * tp_atr_mult if use_tp and a > 0 else float('inf')
                trail_price = entry_price - a * trail_atr_mult if use_trail and a > 0 else 0

    # Close open position
    if in_position:
        pnl = (close[-1] / entry_price - 1) * 100
        trades.append({"side": "long", "entry": entry_price, "exit": close[-1],
                       "entry_date": entry_date, "exit_date": dates[-1],
                       "reason": "OPEN", "pnl_pct": pnl})

    return trades


# ╔══════════════════════════════════════════════════════════════╗
# ║           ENHANCED RUN WITH RISK OPTIONS                    ║
# ╚══════════════════════════════════════════════════════════════╝

def run_with_risk(strategy, df: pd.DataFrame,
                  use_sl=False, sl_atr_mult=2.0,
                  use_tp=False, tp_atr_mult=3.0,
                  use_trail=False, trail_atr_mult=1.5,
                  position_pct=100.0,
                  capital=CAPITAL) -> dict:
    """Run a strategy with optional risk management overlays."""
    sig_df = strategy.generate_signals(df)

    if strategy.exit_mode == "sl_tp":
        # EMA Pullback already has SL/TP built in; modify its params
        trades = simulate_trades_sl_tp(sig_df, strategy.sl_atr_mult, strategy.rr_ratio, "both")
    else:
        # Signal-based strategies with optional risk overlays
        trades = simulate_trades_signal_with_risk(
            sig_df, use_sl=use_sl, sl_atr_mult=sl_atr_mult,
            use_tp=use_tp, tp_atr_mult=tp_atr_mult,
            use_trail=use_trail, trail_atr_mult=trail_atr_mult,
        )

    metrics = calc_metrics(trades)

    # Position sizing: scale PnL proportionally
    if position_pct < 100.0:
        scale = position_pct / 100.0
        scaled_trades = []
        for t in trades:
            t2 = dict(t)
            t2["pnl_pct"] = t["pnl_pct"] * scale
            scaled_trades.append(t2)
        trades = scaled_trades
        metrics = calc_metrics(trades)

    equity = simulate_equity(trades, capital, strategy.commission_pct)

    return {
        "strategy": strategy.name,
        "config": strategy.config_label(),
        "trades": trades,
        "metrics": metrics,
        "equity": equity,
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║               ADD COOLDOWN TO SIGNALS                       ║
# ╚══════════════════════════════════════════════════════════════╝

def apply_cooldown_to_df(df: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    """Apply minimum bar spacing to buy signals."""
    df = df.copy()
    buy = df["buy_signal"].values.copy()
    last = -999
    for i in range(len(buy)):
        if buy[i]:
            if i - last <= cooldown_bars:
                buy[i] = False
            else:
                last = i
    df["buy_signal"] = buy
    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║               ADD HTF TREND FILTER                          ║
# ╚══════════════════════════════════════════════════════════════╝

def apply_htf_filter(df: pd.DataFrame, ema_length: int = 200) -> pd.DataFrame:
    """Only allow buy signals when close > EMA(ema_length)."""
    df = df.copy()
    ema200 = compute_ema(df["close"], ema_length)
    mask = df["close"] > ema200
    df["buy_signal"] = df["buy_signal"] & mask
    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║               DEFAULT STRATEGY CONFIGS                      ║
# ╚══════════════════════════════════════════════════════════════╝

def get_default_strategies():
    """Return default config for each of the 3 strategies."""
    return {
        "EMA Pullback": EMAPullbackStrategy(
            fast_len=50, slow_len=200, use_cooldown=True, use_volume=True,
            use_candle_confirm=True, use_slope=True, use_rsi=True,
            sl_atr_mult=2.0, rr_ratio=2.0
        ),
        "Gaussian+StochRSI": GaussianChannelStochRSIStrategy(),
        "Andean Oscillator": AndeanOscillatorStrategy(length=50, sig_length=9, mode="cross"),
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║                    MAIN A/B TESTING                         ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    api_key = os.environ.get("TWELVE_DATA_API_KEY")
    if not api_key:
        print("ERROR: Set TWELVE_DATA_API_KEY environment variable")
        sys.exit(1)

    td = TDClient(apikey=api_key)

    # ── Fetch all data ──────────────────────────────────────────
    print("=" * 100)
    print("  A/B TEST FRAMEWORK — Systematic Improvement Testing")
    print("=" * 100)
    print(f"\n  Assets: {', '.join(SYMBOLS)}")
    print(f"  Strategies: EMA Pullback, Gaussian+StochRSI, Andean Oscillator")
    print(f"  Capital: ${CAPITAL:,.0f}  |  Interval: {INTERVAL}  |  Bars: {BARS}\n")

    datasets = {}
    for i, sym in enumerate(SYMBOLS):
        try:
            print(f"  Fetching {sym}...")
            datasets[sym] = fetch_ohlcv(td, sym, INTERVAL, BARS)
            print(f"    → {len(datasets[sym])} bars")
            if i < len(SYMBOLS) - 1:
                time.sleep(9)
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    if not datasets:
        print("No data. Exiting.")
        sys.exit(1)

    strategies = get_default_strategies()

    # ══════════════════════════════════════════════════════════════
    #  TEST A: BASELINE (no risk management on signal strategies)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST A: BASELINE")
    print("=" * 100 + "\n")

    baseline = {}  # {(strat_name, symbol): result}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            result = run_with_risk(strat, df, capital=CAPITAL)
            result["symbol"] = sym
            baseline[(sname, sym)] = result

    print_comparison("BASELINE", baseline, None, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  TEST B: ATR STOP LOSS (2x ATR)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST B: + ATR STOP LOSS (2x ATR)")
    print("=" * 100 + "\n")

    test_sl = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            result = run_with_risk(strat, df, use_sl=True, sl_atr_mult=2.0, capital=CAPITAL)
            result["symbol"] = sym
            test_sl[(sname, sym)] = result

    sl_verdict = print_comparison("+ SL 2x ATR", test_sl, baseline, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  TEST B2: ATR STOP LOSS (1.5x ATR) — tighter
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST B2: + ATR STOP LOSS (1.5x ATR)")
    print("=" * 100 + "\n")

    test_sl15 = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            result = run_with_risk(strat, df, use_sl=True, sl_atr_mult=1.5, capital=CAPITAL)
            result["symbol"] = sym
            test_sl15[(sname, sym)] = result

    sl15_verdict = print_comparison("+ SL 1.5x ATR", test_sl15, baseline, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  TEST C: ATR STOP LOSS + TAKE PROFIT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST C: + SL 2x ATR + TP 3x ATR")
    print("=" * 100 + "\n")

    test_sltp = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            result = run_with_risk(strat, df, use_sl=True, sl_atr_mult=2.0,
                                   use_tp=True, tp_atr_mult=3.0, capital=CAPITAL)
            result["symbol"] = sym
            test_sltp[(sname, sym)] = result

    sltp_verdict = print_comparison("+ SL 2x + TP 3x", test_sltp, baseline, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  TEST D: TRAILING STOP (1.5x ATR)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST D: + TRAILING STOP (1.5x ATR)")
    print("=" * 100 + "\n")

    test_trail = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            result = run_with_risk(strat, df, use_trail=True, trail_atr_mult=1.5, capital=CAPITAL)
            result["symbol"] = sym
            test_trail[(sname, sym)] = result

    trail_verdict = print_comparison("+ Trail 1.5x ATR", test_trail, baseline, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  TEST E: SL + TRAILING (combined)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST E: + SL 2x ATR + TRAILING 1.5x ATR")
    print("=" * 100 + "\n")

    test_sl_trail = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            result = run_with_risk(strat, df, use_sl=True, sl_atr_mult=2.0,
                                   use_trail=True, trail_atr_mult=1.5, capital=CAPITAL)
            result["symbol"] = sym
            test_sl_trail[(sname, sym)] = result

    sl_trail_verdict = print_comparison("+ SL + Trail", test_sl_trail, baseline, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  TEST F: POSITION SIZING (75%)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST F: POSITION SIZING — 75% of equity")
    print("=" * 100 + "\n")

    test_pos75 = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            result = run_with_risk(strat, df, position_pct=75.0, capital=CAPITAL)
            result["symbol"] = sym
            test_pos75[(sname, sym)] = result

    pos75_verdict = print_comparison("75% Position", test_pos75, baseline, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  TEST G: POSITION SIZING (50%)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST G: POSITION SIZING — 50% of equity")
    print("=" * 100 + "\n")

    test_pos50 = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            result = run_with_risk(strat, df, position_pct=50.0, capital=CAPITAL)
            result["symbol"] = sym
            test_pos50[(sname, sym)] = result

    pos50_verdict = print_comparison("50% Position", test_pos50, baseline, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  TEST H: RE-ENTRY COOLDOWN (3 bars)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST H: RE-ENTRY COOLDOWN (3 bar minimum)")
    print("=" * 100 + "\n")

    test_cooldown = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            sig_df = strat.generate_signals(df)
            sig_df = apply_cooldown_to_df(sig_df, 3)
            if strat.exit_mode == "sl_tp":
                trades = simulate_trades_sl_tp(sig_df, strat.sl_atr_mult, strat.rr_ratio, "both")
            else:
                trades = simulate_trades_signal(sig_df)
            metrics = calc_metrics(trades)
            equity = simulate_equity(trades, CAPITAL, strat.commission_pct)
            result = {"strategy": sname, "config": strat.config_label(),
                      "trades": trades, "metrics": metrics, "equity": equity, "symbol": sym}
            test_cooldown[(sname, sym)] = result

    cooldown_verdict = print_comparison("+ Cooldown 3 bars", test_cooldown, baseline, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  TEST I: HTF TREND FILTER (200 EMA)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST I: HTF TREND FILTER (close > 200 EMA)")
    print("=" * 100 + "\n")

    test_htf = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            sig_df = strat.generate_signals(df)
            sig_df = apply_htf_filter(sig_df, 200)
            if strat.exit_mode == "sl_tp":
                trades = simulate_trades_sl_tp(sig_df, strat.sl_atr_mult, strat.rr_ratio, "both")
            else:
                trades = simulate_trades_signal(sig_df)
            metrics = calc_metrics(trades)
            equity = simulate_equity(trades, CAPITAL, strat.commission_pct)
            result = {"strategy": sname, "config": strat.config_label(),
                      "trades": trades, "metrics": metrics, "equity": equity, "symbol": sym}
            test_htf[(sname, sym)] = result

    htf_verdict = print_comparison("+ 200 EMA Filter", test_htf, baseline, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  FINAL SCORECARD
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  📊  FINAL SCORECARD — Which improvements to KEEP?")
    print("=" * 100 + "\n")

    all_verdicts = [
        ("B: SL 2x ATR",        sl_verdict),
        ("B2: SL 1.5x ATR",     sl15_verdict),
        ("C: SL 2x + TP 3x",    sltp_verdict),
        ("D: Trail 1.5x ATR",   trail_verdict),
        ("E: SL + Trail",       sl_trail_verdict),
        ("F: 75% Position",     pos75_verdict),
        ("G: 50% Position",     pos50_verdict),
        ("H: Cooldown 3 bars",  cooldown_verdict),
        ("I: 200 EMA Filter",   htf_verdict),
    ]

    scorecard_rows = []
    for name, v in all_verdicts:
        emoji = "✅ KEEP" if v["keep"] else "❌ REMOVE"
        scorecard_rows.append([
            name,
            f"{v['avg_equity_delta']:+.1f}%",
            f"{v['avg_dd_delta']:+.1f}%",
            f"{v['avg_winrate_delta']:+.1f}%",
            f"{v['improved']}/{v['total']}",
            f"{v['dd_improved']}/{v['total']}",
            emoji,
            v["reason"],
        ])

    print(tabulate(scorecard_rows,
                   headers=["Improvement", "Avg Equity Δ", "Avg DD Δ", "Avg WR Δ",
                            "Equity ↑", "DD ↓", "Verdict", "Reason"],
                   tablefmt="pretty"))

    print("\n  Legend:")
    print("    Equity Δ  = Average change in final equity (positive = more money)")
    print("    DD Δ      = Average change in max drawdown (negative = less drawdown)")
    print("    WR Δ      = Average change in win rate")
    print("    Equity ↑  = How many strategy×asset combos improved equity")
    print("    DD ↓      = How many strategy×asset combos reduced drawdown\n")


# ╔══════════════════════════════════════════════════════════════╗
# ║           COMPARISON & VERDICT LOGIC                        ║
# ╚══════════════════════════════════════════════════════════════╝

def print_comparison(label: str, test_results: dict, baseline_results: dict | None,
                     strategies: dict, datasets: dict) -> dict:
    """Print comparison table and return verdict dict."""
    rows = []
    equity_deltas = []
    dd_deltas = []
    wr_deltas = []
    improved_count = 0
    dd_improved_count = 0
    total = 0

    for sname in strategies:
        for sym in datasets:
            key = (sname, sym)
            if key not in test_results:
                continue
            r = test_results[key]
            m = r["metrics"]
            e = r["equity"]

            row = [sname, sym, m["total"], f"{m['win_rate']}%", m["profit_factor"],
                   f"${e['final_equity']:,.2f}", f"{e['return_x']}x", f"{e['max_dd_equity_pct']}%"]

            if baseline_results and key in baseline_results:
                b = baseline_results[key]
                bm = b["metrics"]
                be = b["equity"]

                eq_delta = ((e["final_equity"] - be["final_equity"]) / max(be["final_equity"], 1)) * 100
                dd_delta = e["max_dd_equity_pct"] - be["max_dd_equity_pct"]
                wr_delta = m["win_rate"] - bm["win_rate"]

                equity_deltas.append(eq_delta)
                dd_deltas.append(dd_delta)
                wr_deltas.append(wr_delta)
                total += 1

                if e["final_equity"] > be["final_equity"]:
                    improved_count += 1
                if e["max_dd_equity_pct"] < be["max_dd_equity_pct"]:
                    dd_improved_count += 1

                eq_arrow = "↑" if eq_delta > 0 else "↓" if eq_delta < 0 else "="
                dd_arrow = "↓" if dd_delta < 0 else "↑" if dd_delta > 0 else "="
                row.append(f"{eq_arrow} {eq_delta:+.1f}%")
                row.append(f"{dd_arrow} {dd_delta:+.1f}%")
            else:
                row.extend(["—", "—"])

            rows.append(row)

    headers = ["Strategy", "Symbol", "Trades", "Win %", "PF",
               "Final $", "Return", "DD %"]
    if baseline_results:
        headers.extend(["Equity Δ", "DD Δ"])

    rows.sort(key=lambda x: float(x[5].replace("$", "").replace(",", "")), reverse=True)
    print(tabulate(rows, headers=headers, tablefmt="pretty", stralign="right"))

    # Build verdict
    if not baseline_results or total == 0:
        return {"keep": False, "reason": "baseline", "avg_equity_delta": 0,
                "avg_dd_delta": 0, "avg_winrate_delta": 0,
                "improved": 0, "dd_improved": 0, "total": total}

    avg_eq = np.mean(equity_deltas)
    avg_dd = np.mean(dd_deltas)
    avg_wr = np.mean(wr_deltas)

    # Decision logic:
    # KEEP if: equity improved on >50% of combos OR avg equity up AND dd not worse
    keep = False
    reason = ""

    if avg_eq > 5 and improved_count > total * 0.5:
        keep = True
        reason = f"Avg equity +{avg_eq:.1f}%, {improved_count}/{total} improved"
    elif avg_eq > 0 and avg_dd < -1:
        keep = True
        reason = f"Lower DD ({avg_dd:.1f}%) with positive equity ({avg_eq:+.1f}%)"
    elif avg_dd < -3 and avg_eq > -5:
        keep = True
        reason = f"Significantly lower DD ({avg_dd:.1f}%) with minor equity impact"
    elif avg_eq <= 0 and avg_dd >= 0:
        reason = f"Worse equity ({avg_eq:+.1f}%) and DD ({avg_dd:+.1f}%)"
    elif avg_eq < -10:
        reason = f"Large equity loss ({avg_eq:+.1f}%)"
    else:
        reason = f"Mixed: equity {avg_eq:+.1f}%, DD {avg_dd:+.1f}%, {improved_count}/{total} better"

    verdict = {"keep": keep, "reason": reason, "avg_equity_delta": avg_eq,
               "avg_dd_delta": avg_dd, "avg_winrate_delta": avg_wr,
               "improved": improved_count, "dd_improved": dd_improved_count, "total": total}

    verdict_str = "✅ KEEP" if keep else "❌ REMOVE"
    print(f"\n  Verdict: {verdict_str} — {reason}\n")

    return verdict


if __name__ == "__main__":
    main()
