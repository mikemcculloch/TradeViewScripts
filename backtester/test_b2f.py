"""Quick test: B2+F combo (SL 1.5x ATR + 75% position) vs baseline."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from multi_backtest import *
from ab_test_improvements import *

td = TDClient(apikey=os.environ["TWELVE_DATA_API_KEY"])

datasets = {}
for i, sym in enumerate(SYMBOLS):
    print(f"  Fetching {sym}...")
    datasets[sym] = fetch_ohlcv(td, sym, INTERVAL, BARS)
    if i < len(SYMBOLS) - 1:
        time.sleep(9)

strategies = get_default_strategies()

baseline = {}
for sname, strat in strategies.items():
    for sym, df in datasets.items():
        r = run_with_risk(strat, df, capital=CAPITAL)
        r["symbol"] = sym
        baseline[(sname, sym)] = r

b2f = {}
for sname, strat in strategies.items():
    for sym, df in datasets.items():
        r = run_with_risk(strat, df, use_sl=True, sl_atr_mult=1.5, position_pct=75.0, capital=CAPITAL)
        r["symbol"] = sym
        b2f[(sname, sym)] = r

print()
print("=" * 80)
print("  B2+F: SL 1.5x ATR + 75% Position (NO 200 EMA)")
print("=" * 80)
print()
v = print_comparison("B2+F", b2f, baseline, strategies, datasets)

for sn in strategies:
    base_total = sum(baseline[(sn, s)]["equity"]["final_equity"] for s in datasets)
    comb_total = sum(b2f[(sn, s)]["equity"]["final_equity"] for s in datasets)
    base_dd = np.mean([baseline[(sn, s)]["equity"]["max_dd_equity_pct"] for s in datasets])
    comb_dd = np.mean([b2f[(sn, s)]["equity"]["max_dd_equity_pct"] for s in datasets])
    delta = ((comb_total - base_total) / base_total) * 100
    print(f"  {sn:25s}: base ${base_total:,.0f} -> combo ${comb_total:,.0f} ({delta:+.1f}%)  DD: {base_dd:.1f}% -> {comb_dd:.1f}% ({comb_dd - base_dd:+.1f}%)")

total_b = sum(r["equity"]["final_equity"] for r in baseline.values())
total_c = sum(r["equity"]["final_equity"] for r in b2f.values())
dd_b = np.mean([r["equity"]["max_dd_equity_pct"] for r in baseline.values()])
dd_c = np.mean([r["equity"]["max_dd_equity_pct"] for r in b2f.values()])
imp_eq = sum(1 for k in b2f if b2f[k]["equity"]["final_equity"] > baseline[k]["equity"]["final_equity"])
imp_dd = sum(1 for k in b2f if b2f[k]["equity"]["max_dd_equity_pct"] < baseline[k]["equity"]["max_dd_equity_pct"])
print(f"\n  PORTFOLIO: ${total_b:,.0f} -> ${total_c:,.0f} ({((total_c - total_b) / total_b) * 100:+.1f}%)")
print(f"  AVG DD:    {dd_b:.1f}% -> {dd_c:.1f}% ({dd_c - dd_b:+.1f}%)")
print(f"  Equity UP: {imp_eq}/27  |  DD DOWN: {imp_dd}/27")
