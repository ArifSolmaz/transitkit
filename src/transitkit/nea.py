"""
NASA Exoplanet Archive (NEA) TAP helper (stdlib only).

Fetches planet parameters (period, duration, depth, TIC ID) using TAP sync.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def _escape_adql_string(s: str) -> str:
    # ADQL string literal escaping: single quote doubled
    return s.replace("'", "''")


def tap_sync_json(adql: str, timeout: int = 60) -> list[dict]:
    params = {"query": adql, "format": "json"}
    url = TAP_SYNC + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def lookup_planet(query_text: str, default_only: bool = True, limit: int = 20) -> list[dict]:
    """
    Search the NEA 'ps' table by planet name or hostname.

    Returns rows with:
      pl_name, hostname, tic_id, pl_orbper (days), pl_tranmid (JD),
      pl_trandur (hours), pl_trandep (%), pl_ratror (Rp/Rs), default_flag
    """
    q = _escape_adql_string(query_text.strip())
    where = (
        f"(UPPER(pl_name) LIKE UPPER('%{q}%') OR UPPER(hostname) LIKE UPPER('%{q}%'))"
    )
    if default_only:
        where += " AND default_flag = 1"

    adql = f"""
    SELECT TOP {int(limit)}
      pl_name,
      hostname,
      tic_id,
      pl_orbper,
      pl_tranmid,
      pl_trandur,
      pl_trandep,
      pl_ratror,
      default_flag
    FROM ps
    WHERE {where}
    ORDER BY default_flag DESC, pl_name
    """
    return tap_sync_json(" ".join(adql.split()))
