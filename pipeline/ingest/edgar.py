"""
SEC EDGAR Ingester — Form-4 insider trades
============================================
Uses the SEC EDGAR public API (no key required).
CIK is the canonical ID for all entities.

Usage:
    from pipeline.ingest.edgar import EdgarIngester
    ingester = EdgarIngester()
    df = ingester.fetch_insider_trades("NVDA", "0001045810")  # NVDA CIK
"""

import logging
import time
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://data.sec.gov"
HEADERS = {"User-Agent": "AlphaNexus raunak@alphanexus.ai"}

# Known CIK mappings for demo tickers (pre-loaded for hackathon speed)
DEMO_CIK_MAP = {
    "NVDA": "0001045810",
    "AMD":  "0000002488",
    "INTC": "0000050863",
    "TSMC": "0001046179",
    "MSFT": "0000789019",
    "AAPL": "0000320193",
}


class EdgarIngester:
    """Fetches SEC Form-4 insider trading data from EDGAR."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def lookup_cik_by_ticker(self, ticker: str) -> Optional[str]:
        """Lookup CIK for a ticker symbol via EDGAR company facts endpoint."""
        # First check demo map
        if ticker.upper() in DEMO_CIK_MAP:
            return DEMO_CIK_MAP[ticker.upper()]

        try:
            resp = self.session.get(
                f"{BASE_URL}/submissions/",
                params={"action": "getcompany", "company": ticker,
                        "type": "10-K", "dateb": "", "owner": "include",
                        "count": "1", "search_text": "", "output": "atom"},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"CIK lookup failed for {ticker}: {e}")
        return None

    def fetch_entity_submissions(self, cik: str) -> Dict[str, Any]:
        """Fetch all submission metadata for a CIK."""
        padded_cik = cik.lstrip("0").zfill(10)
        url = f"{BASE_URL}/submissions/CIK{padded_cik}.json"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
            return {}

    def fetch_insider_trades(
        self,
        ticker: str,
        cik: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch Form-4 insider trades for a ticker.

        Returns a DataFrame with columns:
            cik, trader_name, trader_title, ticker, trade_date,
            exact_value, shares, price_per_share, direction, filing_date
        """
        if cik is None:
            cik = self.lookup_cik_by_ticker(ticker)
        if cik is None:
            logger.warning(f"No CIK found for {ticker}, skipping EDGAR ingest.")
            return pd.DataFrame()

        submissions = self.fetch_entity_submissions(cik)
        if not submissions:
            return pd.DataFrame()

        # EDGAR submissions contain filings index — extract Form-4s
        filings = submissions.get("filings", {}).get("recent", {})
        form_types = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])
        accession_nos = filings.get("accessionNumber", [])

        records: List[Dict] = []
        deadline = time.time() + 8  # hard 8-second budget
        count = 0
        for i, form_type in enumerate(form_types):
            if form_type != "4":
                continue
            if count >= 20 or time.time() > deadline:  # cap at 20 filings
                break
            filing_date_str = filing_dates[i] if i < len(filing_dates) else None
            accession = accession_nos[i] if i < len(accession_nos) else None

            # Date filter
            if date_from and filing_date_str and filing_date_str < date_from:
                continue
            if date_to and filing_date_str and filing_date_str > date_to:
                continue

            detail = self._fetch_form4_detail(cik, accession, ticker, filing_date_str)
            records.extend(detail)
            count += 1
            time.sleep(0.1)  # Respect SEC rate limits

        if not records:
            logger.info(f"No Form-4 records found for {ticker} (CIK {cik})")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["cik"] = cik
        df["ticker"] = ticker.upper()
        return df

    def _fetch_form4_detail(
        self, cik: str, accession: Optional[str], ticker: str, filing_date: Optional[str]
    ) -> List[Dict]:
        """Parse a single Form-4 filing for transaction details."""
        if not accession:
            return []
        accession_clean = accession.replace("-", "")
        padded_cik = cik.lstrip("0").zfill(10)
        url = f"{BASE_URL}/Archives/edgar/data/{padded_cik}/{accession_clean}/form4.xml"

        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                return []
            return self._parse_form4_xml(resp.text, filing_date)
        except Exception as e:
            logger.debug(f"Could not parse Form-4 {accession}: {e}")
            return []

    def _parse_form4_xml(self, xml_text: str, filing_date: Optional[str]) -> List[Dict]:
        """Minimal XML parsing for Form-4 transaction data."""
        import xml.etree.ElementTree as ET
        records = []
        try:
            root = ET.fromstring(xml_text)
            # Reporter (insider)
            reporter_name = ""
            reporter_title = ""
            name_el = root.find(".//reportingOwner/reportingOwnerRelationship/officerTitle")
            if name_el is not None:
                reporter_title = name_el.text or ""
            name_el = root.find(".//reportingOwner/reportingOwnerId/rptOwnerName")
            if name_el is not None:
                reporter_name = name_el.text or ""

            # Transactions
            for tx in root.findall(".//nonDerivativeTransaction"):
                trans_date_el = tx.find(".//transactionDate/value")
                shares_el = tx.find(".//transactionAmounts/transactionShares/value")
                price_el = tx.find(".//transactionAmounts/transactionPricePerShare/value")
                code_el = tx.find(".//transactionCoding/transactionCode")

                trans_date = trans_date_el.text if trans_date_el is not None else filing_date
                shares_str = shares_el.text if shares_el is not None else "0"
                price_str = price_el.text if price_el is not None else "0"
                code = code_el.text if code_el is not None else "U"

                try:
                    shares = float(shares_str)
                    price = float(price_str)
                except ValueError:
                    shares, price = 0.0, 0.0

                direction = "buy" if code in ("P", "A") else "sell"
                records.append({
                    "trader_name": reporter_name,
                    "trader_title": reporter_title,
                    "trade_date": trans_date,
                    "shares": shares,
                    "price_per_share": price,
                    "exact_value": round(shares * price, 2),
                    "direction": direction,
                    "filing_date": filing_date,
                })
        except ET.ParseError as e:
            logger.debug(f"XML parse error: {e}")
        return records
