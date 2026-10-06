# NIFTY-100 Deep Statistical Arbitrage — Benchmark Comparison Report

**Period:** 2 January 2019 – 23 September 2026  
**Benchmark:** Yahoo Finance `^CNX100` (NIFTY 100 price index)  
**Strategy:** Completed out-of-sample backtest, seed 42

## Executive conclusion

The strategy generated a positive gross return with exceptionally low market risk. It compounded by **36.19%**, versus **123.10%** for NIFTY 100. Annualized volatility was **3.61%** versus **17.28%**, maximum drawdown was **−2.85%** versus **−38.10%**, and full-period beta to NIFTY 100 was **0.0053** with correlation **0.0254**.

The appropriate interpretation is therefore **not** that the strategy beat the equity index. It did not. Rather, the result is encouraging evidence of a low-beta, market-neutral statistical-arbitrage strategy producing positive gross returns.

## 1. Data and alignment

The benchmark was downloaded from Yahoo Finance using symbol `^CNX100`. It contains 1,899 observations from 2019-01-02 through 2026-09-23. The strategy contains 1,911 observations. Return-based statistics use exact common-date alignment and do not interpolate missing benchmark observations.

The benchmark is a **price index**, not NIFTY 100 TRI. Dividends are therefore excluded from this comparison. The official NSE TRI series should be used for the final investor-return comparison.

## 2. Headline results

| Metric | Strategy | NIFTY 100 |
|---|---:|---:|
| Total compounded return | **36.19%** | **123.10%** |
| Annualized return (daily mean × 252) | **4.35%** | **12.17%** |
| Annualized volatility | **3.61%** | **17.28%** |
| Sharpe | **1.20** | **0.70** |
| Maximum drawdown | **−2.85%** | **−38.10%** |
| Correlation | **0.0254** | — |
| Beta to NIFTY 100 | **0.0053** | — |

## 3. Cumulative wealth

₹1 grew to approximately **₹1.36** in the strategy and **₹2.23** in NIFTY 100. Thus the strategy did not outperform the broad equity market in absolute wealth creation.

That comparison is not sufficient to judge a market-neutral strategy because NIFTY 100 carries substantial long equity exposure that the statistical-arbitrage strategy is intended to remove.

## 4. Drawdown

The strategy's maximum drawdown was only **−2.85%**, compared with **−38.10%** for NIFTY 100. This large difference is consistent with the intended residual/statistical-arbitrage construction.

## 5. Rolling Sharpe

The rolling 252-trading-day Sharpe analysis is more informative than a single full-period number because it exposes regime dependence. The backtest already showed meaningful variation across retraining blocks, so persistence must be examined before treating the headline Sharpe as stable structural alpha.

## 6. Rolling beta and market neutrality

The full-period beta is **0.0053** and correlation is **0.0254**. This is strong evidence that the strategy's daily return variation is largely independent of broad NIFTY 100 movements.

The result is consistent with the intended PCA residualization and cross-sectional positioning, but it is not by itself proof of alpha.

## 7. Relative performance

The strategy underperformed a passive NIFTY 100 investment by approximately **38.95 percentage points of terminal wealth** on the normalized comparison.

This is expected when comparing a market-neutral strategy with a long-only equity index during a strongly appreciating equity period. The economically relevant question is whether the strategy provides attractive **absolute** return for the much lower market and drawdown risk.

## 8. Caveats

1. This benchmark is NIFTY 100 **price index**, not TRI.
2. The strategy result is currently **gross of transaction costs**.
3. Daily stock weights were not saved in the current run, so realized turnover has not yet been calculated.
4. Genuine factor-adjusted alpha has not yet been established.
5. Statistical significance and robustness to model selection require further testing.

## 9. Recommended next steps

1. Obtain and archive official NSE NIFTY 100 TRI and repeat the comparison.
2. Implement the paper's exact transaction-cost model and save daily weights.
3. Run factor regressions against NIFTY 100/NIFTY 50, momentum, value, size and sector factors.
4. Report alpha, t-statistic and R².
5. Perform walk-forward sensitivity and sub-period robustness tests.
6. Compare against simple residual/statistical-arbitrage baselines.

## 10. Reproducibility record

- PCA factors: 5
- PCA window: 252 days
- Beta window: 60 days
- Signal lookback: 30 days
- Training window: 1,000 days
- Retraining frequency: 125 days
- Batch: 125 days
- Epochs: 100
- Seed: 42
- OOS period: 2019-01-02 → 2026-09-23
