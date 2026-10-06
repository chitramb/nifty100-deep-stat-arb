# Data provenance

## Price data

Primary price source for the reproducible free run: `ancap97/NSEDATA4ME`, data branch commit `a13d8112c71528183fff44ac8b27a1458bea3da4`, whose snapshot contains NSE EOD data through 2026-09-23.

The source keeps raw NSE prices and an `adj_factor`. Its adjusted prices correct splits, bonuses, consolidations and demergers and resolve symbol changes through ISIN. Ordinary dividends are **not** automatically applied. Therefore this experiment is a split/corporate-action-adjusted **price-return** study, not a fully dividend-reinvested total-return study.

## Index membership

Point-in-time NIFTY-100 membership comes from `aditya-jha/nse-historical-membership`, using its `index_membership_history.csv`. Membership is applied by effective date; the backtest never replaces historical membership with today's constituents.

## Paper correspondence

The reference paper uses daily adjusted US equity returns, a rolling 252-day PCA correlation window followed by 60-day loading estimation, a 30-day signal lookback, a 1,000-day neural-network training window and 125-day re-estimation frequency. The CNN+Transformer benchmark uses 8 filters, kernel size 2, 4 attention heads, one transformer layer, 0.25 dropout, learning rate 0.001 and 100 epochs.

The NIFTY-100 experiment follows those structural settings where the Indian data permits. It is not a claim of byte-for-byte replication of the original US CRSP/Compustat experiment.

## Reproducibility rule

Never report a performance number unless it is present in a generated result artifact from a completed GitHub Actions run. Failed, partial or preliminary runs are not treated as results.
