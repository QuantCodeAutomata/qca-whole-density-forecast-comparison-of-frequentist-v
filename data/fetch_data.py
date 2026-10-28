"""Fetch S&P 500 and Nikkei 225 daily closing prices and compute log returns.

Primary source: Massive REST API (massive library).
Fallback source: Yahoo Finance via yfinance, using actual index tickers
  ^GSPC (S&P 500) and ^N225 (Nikkei 225), invoked automatically if the
  Massive API key is absent or returns an authentication error.

All returns are in decimal form: y_t = log(P_t / P_{t-1}).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Map logical index names to (massive_ticker, yfinance_ticker)
_INDEX_MAP: dict[str, tuple[str, str]] = {
    'sp500': ('SPY', '^GSPC'),
    'nikkei': ('EWJ', '^N225'),
}


def _fetch_via_massive(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str],
) -> pd.Series:
    """Fetch daily log returns using the Massive REST API."""
    from massive import RESTClient

    key = api_key or os.getenv('MASSIVE_API_KEY') or os.getenv('MASSIVE_TOKEN')
    client = RESTClient(api_key=key)

    aggs: list[dict] = []
    for a in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan='day',
        from_=start_date,
        to=end_date,
        limit=50000,
        adjusted=True,
    ):
        aggs.append(
            {
                'date': pd.Timestamp(a.timestamp, unit='ms', tz='UTC').date(),
                'close': a.close,
            }
        )

    if not aggs:
        raise ValueError(f'No data returned for {ticker}')

    df = pd.DataFrame(aggs).sort_values('date').drop_duplicates('date').set_index('date')
    prices = df['close']
    return np.log(prices / prices.shift(1)).dropna()


def _fetch_via_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Fetch daily log returns using Yahoo Finance (fallback).

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker (e.g. '^GSPC', '^N225').
    start_date, end_date:
        Inclusive date range strings 'YYYY-MM-DD'.
    """
    import yfinance as yf

    # yfinance end date is exclusive — add one day
    end_dt = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_dt, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f'yfinance returned no data for {ticker}')
    prices = df['Close'].squeeze()
    prices.index = prices.index.date
    return np.log(prices / prices.shift(1)).dropna()


def fetch_returns(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    yf_fallback_ticker: Optional[str] = None,
) -> pd.Series:
    """Fetch daily log returns, trying Massive API then Yahoo Finance fallback.

    Returns log returns y_t = log(P_t / P_{t-1}) in decimal form, sorted
    ascending, with NaN values dropped.

    Parameters
    ----------
    ticker:
        Massive API ticker symbol (e.g. 'SPY').
    start_date, end_date:
        Inclusive date range strings 'YYYY-MM-DD'.
    api_key:
        Massive API key. Falls back to MASSIVE_API_KEY / MASSIVE_TOKEN env vars.
    yf_fallback_ticker:
        Yahoo Finance ticker to use if Massive fails (e.g. '^GSPC'). When
        None, no fallback is attempted and the Massive error is re-raised.

    Returns
    -------
    pd.Series
        Daily log returns indexed by ``datetime.date``.
    """
    try:
        return _fetch_via_massive(ticker, start_date, end_date, api_key)
    except Exception as exc:
        if yf_fallback_ticker is None:
            raise
        logger.warning(
            'Massive API failed for %s (%s); falling back to yfinance ticker %s.',
            ticker, exc, yf_fallback_ticker,
        )
        return _fetch_via_yfinance(yf_fallback_ticker, start_date, end_date)


def fetch_index_data(
    index_name: str,
    start_date: str = '1997-01-01',
    end_date: str = '2010-04-20',
    api_key: Optional[str] = None,
) -> pd.Series:
    """Fetch daily log returns for S&P 500 or Nikkei 225.

    Tries the Massive REST API first (SPY / EWJ proxies).  If authentication
    fails, falls back to Yahoo Finance using the actual index tickers
    (^GSPC / ^N225), which cover the full history required by the experiment.

    Parameters
    ----------
    index_name:
        ``'sp500'`` or ``'nikkei'``.
    start_date, end_date:
        Date range strings 'YYYY-MM-DD'.
    api_key:
        Optional Massive API key override.

    Returns
    -------
    pd.Series
        Daily log returns in decimal form.
    """
    if index_name not in _INDEX_MAP:
        raise KeyError(f"Unknown index '{index_name}'. Choose from: {list(_INDEX_MAP)}")
    massive_ticker, yf_ticker = _INDEX_MAP[index_name]
    return fetch_returns(massive_ticker, start_date, end_date, api_key, yf_ticker)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for name in ('sp500', 'nikkei'):
        label = 'S&P 500 (^GSPC)' if name == 'sp500' else 'Nikkei 225 (^N225)'
        print(f'Fetching {label} ...')
        ret = fetch_index_data(name, '1997-01-01', '2010-04-20')
        print(f'  {len(ret)} observations  [{ret.index[0]}  →  {ret.index[-1]}]')
