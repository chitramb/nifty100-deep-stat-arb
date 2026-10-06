import pandas as pd
from data.constituents import membership_on_date
from data.returns import compute_log_returns

def test_membership_on_date():
    h = pd.DataFrame({"effective_date": ["2020-01-01", "2020-06-01"], "symbol": ["AAA", "BBB"], "index_name": ["NIFTY100", "NIFTY100"], "in_index": [True, True]})
    assert membership_on_date(h, "2020-03-01") == {"AAA"}

def test_log_return_resets_by_symbol():
    p = pd.DataFrame({"date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"], "symbol": ["AAA", "AAA", "BBB", "BBB"], "close": [100, 101, 50, 55]})
    out = compute_log_returns(p)
    assert out.loc[out["symbol"] == "AAA", "log_return"].isna().sum() == 1
