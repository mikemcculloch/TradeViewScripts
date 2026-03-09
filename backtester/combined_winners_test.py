"""
Combined Winners Test — Stack B2 + F + I and compare vs baseline
================================================================
Winners from A/B testing:
  B2: SL 1.5x ATR       → +0.6% equity, -1.0% DD
  F:  75% Position Sizing → -3.1% equity, -5.8% DD
  I:  200 EMA Trend Filter → -4.3% equity, -3.4% DD

This script tests all 3 stacked together to verify they compound
positively without cancelling each other out.

Created By: Wooanaz
Created On: 3/9/2026
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(__file__))
from multi_backtest import (
    TDClient, fetch_ohlcv,
    compute_ema, compute_atr,
    EMAPullbackStrategy, GaussianChannelStochRSIStrategy, AndeanOscillatorStrategy,
    simulate_trades_sl_tp, simulate_trades_signal,
    calc_metrics, simulate_equity,
)
from ab_test_improvements import (
    simulate_trades_signal_with_risk,
    apply_htf_filter,
    run_with_risk,
    SYMBOLS, INTERVAL, BARS, CAPITAL,
    get_default_strategies,
    print_comparison,
)

# ╔══════════════════════════════════════════════════════════════╗
# ║          COMBINED: SL 1.5x ATR + 75% Position + 200 EMA    ║
# ╚══════════════════════════════════════════════════════════════╝

def run_combined(strategy, df: pd.DataFrame, capital=CAPITAL) -> dict:
    """
    Run strategy with all 3 winning improvements stacked:
      1. 200 EMA trend filter (only buy when close > 200 EMA)
      2. SL at 1.5x ATR (for signal-exit strategies)
      3. 75% position sizing
    """
    sig_df = strategy.generate_signals(df)

    # 1. Apply 200 EMA trend filter
    sig_df = apply_htf_filter(sig_df, 200)

    # 2. Generate trades with SL 1.5x ATR (signal strategies) or native SL (EMA)
    if strategy.exit_mode == "sl_tp":
        # EMA Pullback: already has SL/TP; just apply the EMA filter above
        trades = simulate_trades_sl_tp(sig_df, strategy.sl_atr_mult, strategy.rr_ratio, "both")
    else:
        # Signal-exit strategies: add 1.5x ATR stop loss
        trades = simulate_trades_signal_with_risk(
            sig_df, use_sl=True, sl_atr_mult=1.5,
            use_tp=False, use_trail=False,
        )

    # 3. Scale PnL to 75% position sizing
    scaled_trades = []
    for t in trades:
        t2 = dict(t)
        t2["pnl_pct"] = t["pnl_pct"] * 0.75
        scaled_trades.append(t2)

    metrics = calc_metrics(scaled_trades)
    equity = simulate_equity(scaled_trades, capital, strategy.commission_pct)

    return {
        "strategy": strategy.name,
        "config": strategy.config_label(),
        "trades": scaled_trades,
        "metrics": metrics,
        "equity": equity,
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║      INDIVIDUAL WINNER TESTS (for layered comparison)       ║
# ╚══════════════════════════════════════════════════════════════╝

def run_b2_only(strategy, df, capital=CAPITAL):
    """SL 1.5x ATR only."""
    return run_with_risk(strategy, df, use_sl=True, sl_atr_mult=1.5, capital=capital)

def run_f_only(strategy, df, capital=CAPITAL):
    """75% position sizing only."""
    return run_with_risk(strategy, df, position_pct=75.0, capital=capital)

def run_i_only(strategy, df, capital=CAPITAL):
    """200 EMA filter only."""
    sig_df = strategy.generate_signals(df)
    sig_df = apply_htf_filter(sig_df, 200)
    if strategy.exit_mode == "sl_tp":
        trades = simulate_trades_sl_tp(sig_df, strategy.sl_atr_mult, strategy.rr_ratio, "both")
    else:
        trades = simulate_trades_signal(sig_df)
    metrics = calc_metrics(trades)
    equity = simulate_equity(trades, capital, strategy.commission_pct)
    return {"strategy": strategy.name, "config": strategy.config_label(),
            "trades": trades, "metrics": metrics, "equity": equity}


def main():
    api_key = os.environ.get("TWELVE_DATA_API_KEY")
    if not api_key:
        print("ERROR: Set TWELVE_DATA_API_KEY environment variable")
        sys.exit(1)

    td = TDClient(apikey=api_key)

    # ── Fetch all data ──────────────────────────────────────────
    print("=" * 100)
    print("  COMBINED WINNERS TEST — B2 (SL 1.5 ATR) + F (75% Pos) + I (200 EMA)")
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
    #  BASELINE (no improvements)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  BASELINE — No improvements")
    print("=" * 100 + "\n")

    baseline = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            result = run_with_risk(strat, df, capital=CAPITAL)
            result["symbol"] = sym
            baseline[(sname, sym)] = result

    print_comparison("BASELINE", baseline, None, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  COMBINED: B2 + F + I  (all 3 winners stacked)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  COMBINED: SL 1.5x ATR + 75% Position + 200 EMA Filter")
    print("=" * 100 + "\n")

    combined = {}
    for sname, strat in strategies.items():
        for sym, df in datasets.items():
            result = run_combined(strat, df, capital=CAPITAL)
            result["symbol"] = sym
            combined[(sname, sym)] = result

    combined_verdict = print_comparison("COMBINED (B2+F+I)", combined, baseline, strategies, datasets)

    # ══════════════════════════════════════════════════════════════
    #  PER-STRATEGY BREAKDOWN
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  📊  PER-STRATEGY BREAKDOWN — Combined vs Baseline")
    print("=" * 100)

    for strat_name in strategies:
        strat_rows = []
        total_base_eq = 0
        total_comb_eq = 0
        base_dds = []
        comb_dds = []

        for sym in datasets:
            key = (strat_name, sym)
            if key not in baseline or key not in combined:
                continue

            b = baseline[key]
            c = combined[key]
            be = b["equity"]
            ce = c["equity"]

            eq_delta = ((ce["final_equity"] - be["final_equity"]) / max(be["final_equity"], 1)) * 100
            dd_delta = ce["max_dd_equity_pct"] - be["max_dd_equity_pct"]

            total_base_eq += be["final_equity"]
            total_comb_eq += ce["final_equity"]
            base_dds.append(be["max_dd_equity_pct"])
            comb_dds.append(ce["max_dd_equity_pct"])

            strat_rows.append([
                sym,
                f"${be['final_equity']:,.2f}", f"{be['max_dd_equity_pct']:.1f}%",
                f"${ce['final_equity']:,.2f}", f"{ce['max_dd_equity_pct']:.1f}%",
                f"{eq_delta:+.1f}%", f"{dd_delta:+.1f}%",
            ])

        if strat_rows:
            avg_base_dd = np.mean(base_dds)
            avg_comb_dd = np.mean(comb_dds)
            total_eq_delta = ((total_comb_eq - total_base_eq) / max(total_base_eq, 1)) * 100

            strat_rows.append([
                "── TOTAL ──",
                f"${total_base_eq:,.2f}", f"{avg_base_dd:.1f}%",
                f"${total_comb_eq:,.2f}", f"{avg_comb_dd:.1f}%",
                f"{total_eq_delta:+.1f}%", f"{avg_comb_dd - avg_base_dd:+.1f}%",
            ])

            print(f"\n  {strat_name}:")
            print(tabulate(strat_rows,
                           headers=["Symbol", "Base $", "Base DD", "Combined $", "Comb DD",
                                    "Equity Δ", "DD Δ"],
                           tablefmt="pretty", stralign="right"))

    # ══════════════════════════════════════════════════════════════
    #  TOP 10 BEST COMBINED RESULTS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  🏆  TOP 10 BEST COMBINED RESULTS")
    print("=" * 100 + "\n")

    top_rows = []
    for (sname, sym), r in combined.items():
        e = r["equity"]
        m = r["metrics"]
        b = baseline[(sname, sym)]["equity"]
        eq_delta = ((e["final_equity"] - b["final_equity"]) / max(b["final_equity"], 1)) * 100
        dd_delta = e["max_dd_equity_pct"] - b["max_dd_equity_pct"]
        top_rows.append([
            sname, sym, m["total"], f"{m['win_rate']}%", m["profit_factor"],
            f"${e['final_equity']:,.2f}", f"{e['return_x']}x",
            f"{e['max_dd_equity_pct']:.1f}%",
            f"${b['final_equity']:,.2f}",
            f"{eq_delta:+.1f}%", f"{dd_delta:+.1f}%",
        ])

    top_rows.sort(key=lambda x: float(x[5].replace("$", "").replace(",", "")), reverse=True)
    print(tabulate(top_rows[:10],
                   headers=["Strategy", "Symbol", "Trades", "Win %", "PF",
                            "Combined $", "Return", "DD %", "Baseline $",
                            "Equity Δ", "DD Δ"],
                   tablefmt="pretty", stralign="right"))

    # ══════════════════════════════════════════════════════════════
    #  FINAL VERDICT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  📋  FINAL VERDICT")
    print("=" * 100)

    total_base = sum(baseline[k]["equity"]["final_equity"] for k in baseline)
    total_comb = sum(combined[k]["equity"]["final_equity"] for k in combined)
    total_delta = ((total_comb - total_base) / total_base) * 100

    base_dds = [baseline[k]["equity"]["max_dd_equity_pct"] for k in baseline]
    comb_dds = [combined[k]["equity"]["max_dd_equity_pct"] for k in combined]
    avg_base_dd = np.mean(base_dds)
    avg_comb_dd = np.mean(comb_dds)

    improved = sum(1 for k in combined if combined[k]["equity"]["final_equity"] > baseline[k]["equity"]["final_equity"])
    dd_better = sum(1 for k in combined if combined[k]["equity"]["max_dd_equity_pct"] < baseline[k]["equity"]["max_dd_equity_pct"])
    total_combos = len(combined)

    print(f"\n  Total Baseline Portfolio:  ${total_base:,.2f}")
    print(f"  Total Combined Portfolio: ${total_comb:,.2f}  ({total_delta:+.1f}%)")
    print(f"\n  Avg Baseline Drawdown:  {avg_base_dd:.1f}%")
    print(f"  Avg Combined Drawdown:  {avg_comb_dd:.1f}%  ({avg_comb_dd - avg_base_dd:+.1f}%)")
    print(f"\n  Equity improved: {improved}/{total_combos} combos")
    print(f"  Drawdown improved: {dd_better}/{total_combos} combos")

    if total_comb > total_base and avg_comb_dd < avg_base_dd:
        print("\n  🎯 VERDICT: ✅ WINNERS STACK WELL — More money AND less risk!")
    elif total_comb > total_base:
        print("\n  🎯 VERDICT: ✅ STACK APPROVED — More total equity")
    elif avg_comb_dd < avg_base_dd - 3:
        print(f"\n  🎯 VERDICT: ✅ STACK APPROVED — Significantly lower DD ({avg_comb_dd - avg_base_dd:+.1f}%)")
        print(f"     Equity trade-off: {total_delta:+.1f}% is acceptable for {avg_comb_dd - avg_base_dd:+.1f}% DD reduction")
    else:
        print("\n  🎯 VERDICT: ❌ WINNERS DON'T STACK — Individual improvements cancel each other out")
        print("     Consider using only the best 2, or applying per-strategy/per-asset")

    print()


if __name__ == "__main__":
    main()
