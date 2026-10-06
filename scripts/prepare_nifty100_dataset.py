from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nse-code", required=True)
    ap.add_argument("--membership", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2026-09-23")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, args.nse_code)
    from loader import load_data

    m = pd.read_csv(args.membership, parse_dates=["valid_from", "valid_to"])
    m = m[m["index_name"].str.lower().eq("nifty 100")].copy()
    symbols = sorted(m["symbol"].dropna().astype(str).str.upper().unique())

    df = load_data(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        include_delisted=True,
        adjusted=True,
        columns=["date", "symbol", "key", "series", "close"],
    )
    if df.empty:
        raise RuntimeError("NSEDATA4ME returned no price rows for the NIFTY-100 symbols.")

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["series"].eq("EQ")].copy()
    df = df.rename(columns={"close": "close_adjusted"})
    df = df[["date", "symbol", "key", "close_adjusted"]]
    df = df.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "nifty100_adjusted_prices.parquet", index=False)

    manifest = {
        "source": "ancap97/NSEDATA4ME",
        "source_data_branch": "data",
        "source_data_commit": "a13d8112c71528183fff44ac8b27a1458bea3da4",
        "source_snapshot": "NSE EOD data through 2026-09-23",
        "price_adjustment": "split/bonus/consolidation/demerger adjusted; dividends not automatically adjusted by source",
        "membership_source": "aditya-jha/nse-historical-membership",
        "membership_index": "Nifty 100",
        "start": args.start,
        "end": args.end,
        "n_rows": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "first_date": str(df["date"].min().date()),
        "last_date": str(df["date"].max().date()),
    }
    (out / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
