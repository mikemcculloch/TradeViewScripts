# EMA Pullback Strategy — Backtester

A Python backtester that uses the **Twelve Data API** to test the same EMA pullback logic from the PineScript strategy across multiple assets.

## Setup

```bash
cd backtester
pip install -r requirements.txt
```

## Usage

### Set your API key
```bash
# Windows
set TWELVE_DATA_API_KEY=your_key_here

# or pass directly
python backtest.py --api-key your_key_here
```

### Run default backtest (all 6 assets, daily)
```bash
python backtest.py
```

### Test specific symbols
```bash
python backtest.py --symbols BTC/USD ETH/USD --interval 4h --bars 2000
```

### Optimization mode (grid search)
```bash
python backtest.py --optimize
```
Tests **576 filter combinations** per symbol across:
- EMA lengths: 20/100, 20/200, 50/100, 50/200
- Volume filter: on/off
- Candle confirmation: on/off
- EMA slope filter: on/off
- RSI filter: on/off
- SL multiplier: 1.5x, 2.0x, 3.0x ATR
- Risk:Reward ratio: 1.5, 2.0, 3.0

Shows the **best configuration per asset** — use those settings in TradingView.

### Trade direction
```bash
python backtest.py --direction long     # longs only
python backtest.py --direction short    # shorts only
```

## Output

```
==================================================================================================
  EMA PULLBACK STRATEGY — BACKTEST RESULTS
==================================================================================================
  Symbol | Config               | Trades | Wins | Win % | PF   | Total PnL % | Max DD %
  BTC/USD| EMA(50/200) CD SW... |     42 |   24 | 57.1% | 1.82 |      34.5%  |   12.3%
  GOOGL  | EMA(50/200) CD SW... |     38 |   20 | 52.6% | 1.45 |      18.2%  |    8.7%
  ...
```

## Intervals
`1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`

## API Limits
Twelve Data free tier: 8 API calls/minute, 800/day. The backtester makes 1 call per symbol.
