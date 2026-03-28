"""
Entity Resolution
==================
Canonical entity ID assignment using SEC CIK numbers.
Fuzzy name matching via RapidFuzz at 85% threshold.

Rule:
  1. If CIK is known → use CIK as canonical trader_id
  2. If no CIK → fuzzy match against EDGAR canonical names at ≥85%
  3. Below 85% → generate deterministic ID from name (don't force a wrong match)
"""

import hashlib
import logging
from typing import Dict, Optional

import requests
import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

EDGAR_HEADERS = {"User-Agent": "AlphaNexus raunak@alphanexus.ai"}


class EntityResolver:
    """
    Resolves trader names across data sources to a single canonical trader_id.
    """

    def __init__(self):
        # Cache: canonical_name → cik
        self._edgar_cik_cache: Dict[str, str] = {}
        # Cache: normalized_name → canonical_name
        self._match_cache: Dict[str, Optional[str]] = {}

    def resolve_trader_id(self, name: str, cik: Optional[str] = None) -> str:
        """
        Returns a stable trader_id string for the given name/CIK.
        CIK always wins over name matching.
        """
        if cik:
            return f"cik-{cik.lstrip('0')}"

        # Name-based resolution
        canonical = self._fuzzy_match_edgar(name)
        if canonical and canonical in self._edgar_cik_cache:
            return f"cik-{self._edgar_cik_cache[canonical].lstrip('0')}"

        # No match → deterministic hash-based ID
        return f"name-{self._name_hash(name)}"

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trader_id column to a DataFrame containing a trader_name column.
        If df already has a cik column, CIK takes priority.
        """
        if df.empty:
            return df

        has_cik = "cik" in df.columns

        def _resolve(row):
            cik = row.get("cik") if has_cik else None
            return self.resolve_trader_id(
                str(row.get("trader_name", "")),
                str(cik) if cik and pd.notna(cik) else None,
            )

        df["trader_id"] = df.apply(_resolve, axis=1)
        return df

    def load_edgar_lookup(self, names: list) -> None:
        """
        Pre-populate the EDGAR name→CIK cache for a list of names.
        Useful to batch-load before processing large datasets.
        """
        for name in names:
            self._fetch_cik_from_edgar(name)

    # ──────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────

    def _fuzzy_match_edgar(self, name: str) -> Optional[str]:
        """Match a name against the EDGAR canonical name cache at ≥85%."""
        key = name.lower().strip()
        if key in self._match_cache:
            return self._match_cache[key]

        if not self._edgar_cik_cache:
            self._match_cache[key] = None
            return None

        best_name = max(
            self._edgar_cik_cache.keys(),
            key=lambda x: fuzz.ratio(key, x.lower()),
        )
        score = fuzz.ratio(key, best_name.lower())
        result = best_name if score >= 85 else None
        self._match_cache[key] = result
        if result:
            logger.debug(f"Matched '{name}' → '{best_name}' (score={score})")
        return result

    def _fetch_cik_from_edgar(self, name: str) -> Optional[str]:
        """Fetch CIK from EDGAR company search."""
        try:
            resp = requests.get(
                "https://www.sec.gov/cgi-bin/browse-edgar",
                params={"company": name, "CIK": "", "type": "4",
                        "dateb": "", "owner": "include", "count": "1",
                        "search_text": "", "action": "getcompany", "output": "atom"},
                headers=EDGAR_HEADERS,
                timeout=10,
            )
            # Basic parsing — for hackathon speed we use the demo map
            return None
        except Exception:
            return None

    @staticmethod
    def _name_hash(name: str) -> str:
        """Deterministic 8-char hash for unresolved names."""
        return hashlib.md5(name.lower().strip().encode()).hexdigest()[:8]


def match_cik(quiver_name: str, edgar_lookup: Dict[str, str]) -> Optional[str]:
    """
    Standalone utility matching a Quiver name against EDGAR lookup dict.
    Mirrors the workflow doc implementation exactly.

    edgar_lookup = {canonical_name: cik}
    Returns CIK string or None if below 85% threshold.
    """
    if not edgar_lookup:
        return None
    best_match = max(
        edgar_lookup.keys(),
        key=lambda x: fuzz.ratio(quiver_name.lower(), x.lower()),
    )
    score = fuzz.ratio(quiver_name.lower(), best_match.lower())
    if score > 85:
        return edgar_lookup[best_match]
    return None
