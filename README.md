# NIFTY-100 Deep Statistical Arbitrage

Reproduction of Guijarro-Ordonez, Pelger & Zanotti (arXiv:2106.04028), adapted to NIFTY-100.

## Research principles

- Point-in-time NIFTY-100 membership; no survivorship-biased universe.
- Raw market data stays out of Git.
- Corporate-action handling is explicit.
- Rolling factor estimation uses only information available at each date.
- Walk-forward out-of-sample evaluation.
- Transaction costs and portfolio constraints are evaluated separately.
- Every experiment records its configuration and code version.

## Initial target

PCA residual model + CNN + Transformer + neural allocation, followed by IPCA and PatchTST comparisons.

## Status

Scaffold initialized. Data acquisition and validation are next.
