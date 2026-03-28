"""
Quiver Quantitative Congressional Trades Ingester
===================================================
Optional — only runs if QUIVER_API_KEY is set.
Falls back to Capitol Trades API (no key) if Quiver is unavailable.

Data: Congressional stock trades under the STOCK Act.
"""

import logging
from typing import Optional, List, Dict

import requests
import pandas as pd

logger = logging.getLogger(__name__)

QUIVER_BASE = "https://api.quiverquant.com/beta"
CAPITOL_TRADES_BASE = "https://api.capitoltrades.com/trades"


class QuiverIngester:
    """Fetches congressional trading data from Quiver or Capitol Trades."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.use_quiver = bool(api_key)

    def fetch_congressional_trades(
        self,
        ticker: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch congressional trades. Tries Quiver first, falls back to Capitol Trades.
        Returns DataFrame with columns:
            trader_name, ticker, trade_date, trade_value_min, trade_value_max,
            trade_value, direction, trader_type
        """
        records: List[Dict] = []

        if self.use_quiver:
            records = self._fetch_quiver(ticker, date_from, date_to)
        if not records:
            logger.info("Using Capitol Trades fallback for congressional data.")
            records = self._fetch_capitol_trades(ticker, date_from, date_to)

        if not records:
            logger.warning("No congressional trade data found.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["trader_type"] = "politician"
        return df

    def _fetch_quiver(
        self, ticker: Optional[str], date_from: Optional[str], date_to: Optional[str]
    ) -> List[Dict]:
        try:
            headers = {"Authorization": f"Token {self.api_key}"}
            endpoint = f"{QUIVER_BASE}/live/congresstrading"
            if ticker:
                endpoint = f"{QUIVER_BASE}/live/congresstrading/{ticker.upper()}"

            resp = requests.get(endpoint, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            records = []
            for r in data:
                trade_date = r.get("Date") or r.get("TransactionDate", "")
                if date_from and trade_date < date_from:
                    continue
                if date_to and trade_date > date_to:
                    continue

                amount_str = r.get("Amount", "$1,001-$15,000")
                val_min, val_max = self._parse_amount_range(amount_str)
                records.append({
                    "trader_name": r.get("Representative", ""),
                    "ticker": r.get("Ticker", ticker or ""),
                    "trade_date": trade_date,
                    "trade_value_min": val_min,
                    "trade_value_max": val_max,
                    "trade_value": (val_min + val_max) / 2,
                    "direction": r.get("Transaction", "buy").lower(),
                })
            return records
        except Exception as e:
            logger.warning(f"Quiver API failed: {e}")
            return []

    def _fetch_capitol_trades(
        self, ticker: Optional[str], date_from: Optional[str], date_to: Optional[str]
    ) -> List[Dict]:
        """Capitol Trades — free, no key, good congressional data."""
        try:
            params: Dict = {"pageSize": 500}
            if ticker:
                params["ticker"] = ticker.upper()
            if date_from:
                params["startDate"] = date_from
            if date_to:
                params["endDate"] = date_to

            resp = requests.get(CAPITOL_TRADES_BASE, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            trades = data.get("data", [])

            records = []
            for r in trades:
                val = float(r.get("amount", 0))
                records.append({
                    "trader_name": r.get("politician", {}).get("name", ""),
                    "ticker": r.get("asset", {}).get("ticker", ticker or ""),
                    "trade_date": r.get("filedAt", "")[:10],
                    "trade_value_min": val * 0.8,
                    "trade_value_max": val * 1.2,
                    "trade_value": val,
                    "direction": r.get("type", "buy").lower(),
                })
            return records
        except Exception as e:
            logger.warning(f"Capitol Trades fallback failed: {e}")
            return []

    @staticmethod
    def _parse_amount_range(amount_str: str) -> tuple:
        """Parse Quiver amount strings like '$1,001-$15,000' → (1001, 15000)."""
        try:
            clean = amount_str.replace("$", "").replace(",", "")
            if "-" in clean:
                parts = clean.split("-")
                return float(parts[0].strip()), float(parts[1].strip())
            elif "+" in clean:
                val = float(clean.replace("+", "").strip())
                return val, val * 2
            else:
                val = float(clean.strip())
                return val, val
        except Exception:
            return 0.0, 0.0
