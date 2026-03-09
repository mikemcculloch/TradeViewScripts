"""Head-to-head: 'verygood' (no ATR SL, 100% eq) vs 'v3.1' (ATR SL, 75% eq)"""
from gaussian_backtest import *
from tabulate import tabulate
from datetime import datetime

td = TDClient(apikey="e07e612b5bc5440ba6ff1fb9441d1b80")
print("  Fetching BTC/USD (1day, 1500 bars)...")
raw_df = fetch_ohlcv(td, "BTC/USD", "1day", 1500)
print(f"    -> {len(raw_df)} bars loaded ({raw_df.index[0]} to {raw_df.index[-1]})")

configs = [
    # 'verygood' script: StochRSI entry, no ATR SL, 100% equity
    GaussianConfig(strategy_version="v3.1", use_atr_sl=False, equity_pct=1.0),
    # 'v3.0 Updated' script: StochRSI entry + ATR SL, 100% equity (same sizing)
    GaussianConfig(strategy_version="v3.1", use_atr_sl=True, equity_pct=1.0),
]

all_results = []
all_trades = {}
for cfg in configs:
    df = compute_gaussian_channel(
        raw_df, src_type=cfg.src_type, poles=cfg.poles,
        period=cfg.period, mult=cfg.mult,
        reduced_lag=cfg.reduced_lag, fast_response=cfg.fast_response,
    )
    trades = simulate_trades(df, cfg)
    metrics = calc_metrics(trades, 1000.0, cfg.equity_pct)
    metrics["symbol"] = "BTC/USD"
    metrics["config"] = cfg.short_name()
    key = f"BTC/USD [{cfg.short_name()}]"
    all_results.append(metrics)
    all_trades[key] = trades
    print_trade_log(trades, key)

print()
print("=" * 130)
print("  GAUSSIAN CHANNEL — HEAD-TO-HEAD: 'verygood' vs 'v3.1 (ATR SL)'")
print("=" * 130)
print(f"  Interval: 1day  |  Capital: $1,000  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 130)

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

save_results(all_results, all_trades, "1day", 1000.0, "verygood_vs_v31")
