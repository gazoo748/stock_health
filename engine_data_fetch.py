# engine_data_fetch.py

from __future__ import annotations

import re
import requests
from functools import lru_cache
from typing import Optional, Tuple

import pandas as pd
import pandas_ta as ta
import yfinance as yf

from config import SPY_TICKER, FMP_BASE_URL, get_fmp_api_key
from models import (
    InstrumentInfo,
    FundamentalResult,
    ValuationSnapshot,
    MarketRegimeResult,
    TechnicalSnapshot,
    StockCheckup,
)

class FMPUnavailableError(Exception):
    """Raised when FMP cannot be used (no key, quota, or API error)."""
    pass



# ---------- Info / classification helpers ----------



def infer_underlying_ticker(symbol: str, info: dict) -> Optional[str]:
    """
    Best-effort attempt to infer the underlying common-equity ticker for
    preferreds, warrants, rights, units, etc.

    Strategy:
      1. If info contains something like `underlyingSymbol`, use that.
      2. Otherwise, strip common suffix patterns from the ticker.
      3. Verify that the inferred ticker looks real (has price history).
    """
    s = symbol.upper()

    # 1) Metadata-based guesses
    for key in ("underlyingSymbol", "underlyingExchangeSymbol", "underlyingTicker"):
        base = info.get(key)
        if isinstance(base, str) and base.strip():
            base = base.strip().upper()
            # sanity check: does it have any history?
            base_t = yf.Ticker(base)
            hist = base_t.history(period="1mo")
            if not hist.empty:
                return base

    # 2) Pattern-based guesses on the symbol itself

    # Common preferred / warrant suffix patterns (US-style & some variations)
    patterns = [
        r"(?P<base>.+)-P[A-Z]?$",      # XYZ-PA, XYZ-P, XYZ-PR?
        r"(?P<base>.+)\.P[A-Z]?$",     # XYZ.PA
        r"(?P<base>.+)\.PR[A-Z]?$",    # XYZ.PRA
        r"(?P<base>.+)-PR[A-Z]?$",     # XYZ-PRA
        r"(?P<base>.+)\.WS[A-Z]?$",    # XYZ.WS, XYZ.WSA
        r"(?P<base>.+)-WS[A-Z]?$",     # XYZ-WSA
        r"(?P<base>.+)\+$",            # XYZ+, sometimes warrants
        r"(?P<base>.+)\.U$",           # XYZ.U (units)
    ]

    candidates: list[str] = []

    for pat in patterns:
        m = re.match(pat, s)
        if m:
            base = m.group("base")
            if base:
                candidates.append(base)

    # If no pattern matched, try simple trailing "-A"/".A" type
    if not candidates:
        simple_suffixes = ["-A", "-B", "-C", ".A", ".B", ".C"]
        for suf in simple_suffixes:
            if s.endswith(suf):
                base = s[: -len(suf)]
                if base:
                    candidates.append(base)

    # Verify candidates by asking yfinance for some price history
    for base in candidates:
        try:
            base_t = yf.Ticker(base)
            hist = base_t.history(period="1mo")
            if not hist.empty:
                return base
        except Exception:
            continue

    # No good candidate found
    return None


def safe_info(ticker_obj: yf.Ticker) -> dict:
    """Safely fetch ticker.info (it can be slow / error)."""
    try:
        info = ticker_obj.info or {}
        if not isinstance(info, dict):
            return {}
        return info
    except Exception:
        return {}


def detect_is_fund(symbol: str, info: dict) -> bool:
    """
    Determine if this is a fund/ETF/index using quoteType/category plus heuristics.
    """
    s = symbol.upper()
    qt = str(info.get("quoteType", "")).lower()
    category = str(info.get("category", "")).lower()

    if qt in {"etf", "mutualfund", "index", "fund"}:
        return True
    if "fund" in category:
        return True

    if s.endswith("X"):
        return True

    etf_like = {
        "SPY", "GLD", "QQQ", "VTI", "VOO",
        "IWM", "XLK", "XLF", "XLV", "XLE", "XLU",
    }
    if s in etf_like:
        return True

    return False


def should_skip_fundamentals(info: dict) -> bool:
    """
    Identify instruments where company-level fundamentals don't make sense
    (preferred stock, warrants, bonds, etc.).
    """
    qt = str(info.get("quoteType", "")).lower()
    name = (info.get("longName") or info.get("shortName") or "").lower()

    if "preferred" in name or "pref" in name:
        return True
    if qt in {
        "preferred_stock",
        "warrant",
        "right",
        "structuredproduct",
        "unit",
        "bond",
    }:
        return True

    return False


def get_instrument_info(symbol: str, ticker_obj: yf.Ticker) -> InstrumentInfo:
    info = safe_info(ticker_obj)
    is_fund_like = detect_is_fund(symbol, info)
    fundamentals_skipped = should_skip_fundamentals(info) or is_fund_like
    return InstrumentInfo(
        symbol=symbol,
        short_name=info.get("shortName"),
        long_name=info.get("longName"),
        quote_type=info.get("quoteType"),
        exchange=info.get("exchange") or info.get("fullExchangeName"),
        is_fund_like=is_fund_like,
        fundamentals_skipped=fundamentals_skipped,
    )


# ---------- Fundamentals ----------

def fmp_get_json(path: str, params: Optional[dict] = None) -> list:
    """
    Soft-fail FMP helper.

    - If no key is configured or the API errors out (401/403/429/etc),
      we raise FMPUnavailableError so the caller can gracefully skip FMP
      and just treat fundamentals as "unavailable" instead of crashing
      or surfacing noisy technical messages to the user.
    """
    key = get_fmp_api_key()
    if not key:
        raise FMPUnavailableError("FMP not configured (no API key).")

    params = dict(params or {})
    params["apikey"] = key
    url = f"{FMP_BASE_URL}{path}"

    try:
        resp = requests.get(url, params=params, timeout=15)
        # If the key is wrong or quota exceeded, often 401/403/429
        if resp.status_code in (401, 403, 429):
            raise FMPUnavailableError(
                f"FMP unavailable (HTTP {resp.status_code} – auth/quota issue)."
            )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            # Treat unexpected format as "FMP currently not usable"
            raise FMPUnavailableError("FMP returned an unexpected response format.")
        return data
    except FMPUnavailableError:
        # Bubble up our soft error as-is
        raise
    except Exception as e:
        # Any other network/JSON errors → treat as "FMP unavailable"
        raise FMPUnavailableError(f"FMP request error: {e}")



def get_fundamentals_yf(ticker_obj: yf.Ticker) -> Tuple[float, float, Optional[float], str]:
    inc = ticker_obj.quarterly_income_stmt
    if inc is None or inc.empty:
        raise RuntimeError("yfinance: quarterly income statement empty.")

    net_income_row_candidates = [
        "Net Income",
        "NetIncome",
        "Net Income Applicable To Common Shares",
    ]
    net_income_series = None
    for label in net_income_row_candidates:
        if label in inc.index:
            net_income_series = inc.loc[label]
            break
    if net_income_series is None:
        raise RuntimeError("yfinance: net income row not found.")

    ttm_net_income = float(net_income_series.iloc[:4].sum())

    bs = ticker_obj.quarterly_balance_sheet
    if bs is None or bs.empty:
        raise RuntimeError("yfinance: quarterly balance sheet empty.")

    assets_row_candidates = ["Total Assets", "TotalAssets"]
    total_assets = None
    for label in assets_row_candidates:
        if label in bs.index:
            total_assets = float(bs.loc[label].iloc[0])
            break

    debt_row_candidates = ["Total Debt", "TotalDebt"]
    total_debt = None
    for label in debt_row_candidates:
        if label in bs.index:
            total_debt = float(bs.loc[label].iloc[0])
            break

    if total_debt is None:
        st_labels = ["Short Long Term Debt", "Short Term Debt"]
        lt_labels = ["Long Term Debt", "LongTermDebt"]
        short_debt = next(
            (float(bs.loc[l].iloc[0]) for l in st_labels if l in bs.index),
            0.0,
        )
        long_debt = next(
            (float(bs.loc[l].iloc[0]) for l in lt_labels if l in bs.index),
            0.0,
        )
        total_debt = short_debt + long_debt

    reason = "Fundamentals from yfinance quarterly statements."
    return ttm_net_income, total_debt, total_assets, reason


def get_fundamentals_fmp(symbol: str) -> Tuple[float, float, Optional[float], str]:
    sym = symbol.upper()
    inc_data = fmp_get_json(
        f"/income-statement/{sym}",
        params={"period": "quarter", "limit": 4},
    )
    if not inc_data:
        raise RuntimeError("FMP: income statement data empty.")
    ttm_net_income = 0.0
    for item in inc_data[:4]:
        ni = item.get("netIncome")
        if ni is None:
            raise RuntimeError("FMP: 'netIncome' missing.")
        ttm_net_income += float(ni)

    bs_data = fmp_get_json(
        f"/balance-sheet-statement/{sym}",
        params={"period": "quarter", "limit": 1},
    )
    if not bs_data:
        raise RuntimeError("FMP: balance sheet data empty.")
    bs0 = bs_data[0]
    total_assets = bs0.get("totalAssets")
    total_assets = float(total_assets) if total_assets is not None else None

    total_debt = bs0.get("totalDebt")
    if total_debt is not None:
        total_debt = float(total_debt)
    else:
        short_debt = float(bs0.get("shortTermDebt", 0.0) or 0.0)
        long_debt = float(bs0.get("longTermDebt", 0.0) or 0.0)
        total_debt = short_debt + long_debt

    reason = "Fundamentals from Financial Modeling Prep (FMP)."
    return ttm_net_income, total_debt, total_assets, reason


def compute_fundamentals(symbol: str, identity: InstrumentInfo, ticker_obj: yf.Ticker) -> FundamentalResult:
    """
    Try yfinance first. If that fails:
      - If FMP is available, use it.
      - If FMP is unavailable (no key, quota, etc.), just report
        "fundamentals unavailable" without throwing a noisy error.

    For non-common-equity instruments (preferreds, warrants, units), we try to
    infer an underlying common-equity ticker and pull fundamentals from there.
    """
    info = safe_info(ticker_obj)

if identity.fundamentals_skipped:
    # Attempt to infer underlying
    underlying = infer_underlying_ticker(symbol, info)

    if underlying:
        try:
            underlying_t = yf.Ticker(underlying)
            ttm_net_income, total_debt, total_assets, src = get_fundamentals_yf(underlying_t)
        except Exception:
            # Could not retrieve fundamentals even after inference
            return FundamentalResult(
                ttm_net_income=None,
                total_debt=None,
                total_assets=None,
                debt_asset_ratio=None,
                source=f"Underlying {underlying}",
                reason=(
                    f"Instrument is non-common equity. "
                    f"Underlying '{underlying}' inferred heuristically, "
                    f"but fundamentals could not be retrieved."
                ),
                used_underlying=True,
                underlying_symbol=underlying,
            )

        # Compute ratio
        debt_asset_ratio = (
            total_debt / total_assets
            if total_assets and total_assets != 0
            else None
        )

        return FundamentalResult(
            ttm_net_income=ttm_net_income,
            total_debt=total_debt,
            total_assets=total_assets,
            debt_asset_ratio=debt_asset_ratio,
            source=f"{src} (via underlying {underlying})",
            reason=(
                "Instrument appears to be preferred/warrant/unit. "
                f"Fundamentals were derived heuristically from underlying "
                f"common equity '{underlying}'."
            ),
            used_underlying=True,
            underlying_symbol=underlying,
        )

    # No underlying was discovered → totally transparent skip
    return FundamentalResult(
        ttm_net_income=None,
        total_debt=None,
        total_assets=None,
        debt_asset_ratio=None,
        source="None",
        reason=(
            "Non-common-equity instrument. "
            "Fundamentals skipped; no reliable underlying ticker "
            "could be inferred."
        ),
        used_underlying=False,
        underlying_symbol=None,
    )

    # Try yfinance
    try:
        ttm_net_income, total_debt, total_assets, src = get_fundamentals_yf(ticker_obj)
    except Exception as e_yf:
        # Try FMP as a soft fallback
        try:
            ttm_net_income, total_debt, total_assets, src = get_fundamentals_fmp(symbol)
        except FMPUnavailableError as e_fmp_unavail:
            # Soft failure: no FMP, so we simply mark fundamentals as unavailable
            return FundamentalResult(
                ttm_net_income=None,
                total_debt=None,
                total_assets=None,
                debt_asset_ratio=None,
                source="None",
                reason=(
                    "Fundamentals unavailable: yfinance failed, and FMP fallback "
                    f"was not usable ({e_fmp_unavail})."
                ),
            )
        except Exception as e_fmp:
            # Hard data error from FMP itself (e.g. symbol truly missing)
            return FundamentalResult(
                ttm_net_income=None,
                total_debt=None,
                total_assets=None,
                debt_asset_ratio=None,
                source="None",
                reason=(
                    "Fundamentals unavailable: yfinance failed "
                    f"({e_yf}) and FMP could not provide data ({e_fmp})."
                ),
            )

    # We have some numbers from either yf or FMP
    if total_assets is None or total_assets == 0:
        debt_asset_ratio = None
    else:
        debt_asset_ratio = total_debt / total_assets

    return FundamentalResult(
        ttm_net_income=ttm_net_income,
        total_debt=total_debt,
        total_assets=total_assets,
        debt_asset_ratio=debt_asset_ratio,
        source=src,
        reason="TTM net income and balance sheet computed.",
    )


# ---------- Valuation (P/E) ----------

def compute_valuation(identity: InstrumentInfo, ticker_obj: yf.Ticker) -> ValuationSnapshot:
    info = safe_info(ticker_obj)

    if identity.is_fund_like or identity.fundamentals_skipped:
        return ValuationSnapshot(
            pe_ratio=None,
            source="None",
            reason="Funds/ETFs or non-common equity: P/E is not a primary yardstick; skipped.",
        )

    pe = info.get("trailingPE")
    src = "Yahoo Finance trailingPE."
    if pe is None or not isinstance(pe, (int, float)) or pe <= 0:
        fpe = info.get("forwardPE")
        if fpe is not None and isinstance(fpe, (int, float)) and fpe > 0:
            pe = fpe
            src = "Yahoo Finance forwardPE (used when trailingPE unavailable)."
        else:
            return ValuationSnapshot(
                pe_ratio=None,
                source="None",
                reason="P/E not available or not meaningful (negative earnings or missing data).",
            )

    return ValuationSnapshot(
        pe_ratio=float(pe),
        source=src,
        reason="P/E approximates how 'fat' (expensive) or 'skinny' (cheap) the stock is vs its earnings.",
    )


# ---------- Market regime (SPY) ----------

@lru_cache(maxsize=1)
def get_market_regime() -> MarketRegimeResult:
    spy = yf.Ticker(SPY_TICKER)
    hist = spy.history(period="1y")
    if hist.empty or "Close" not in hist:
        raise RuntimeError("Failed to fetch SPY history.")
    closes = hist["Close"]
    sma_200 = closes.rolling(200).mean().iloc[-1]
    last_close = float(closes.iloc[-1])
    spy_trend_ok = last_close > sma_200
    return MarketRegimeResult(
        spy_last_close=last_close,
        spy_sma_200=float(sma_200),
        spy_trend_ok=spy_trend_ok,
    )


# ---------- Technical snapshot ----------

def compute_technical_snapshot(ticker_obj: yf.Ticker) -> TechnicalSnapshot:
    hist = ticker_obj.history(period="1y")
    if hist.empty or "Close" not in hist:
        return TechnicalSnapshot(
            last_close=float("nan"),
            rsi_14=None,
            bb_lower_20=None,
            bb_mid_20=None,
            bb_upper_20=None,
            wk52_low=None,
            wk52_high=None,
            wk52_position_pct=None,
            reason="No price history.",
        )

    df = hist.copy()
    closes = df["Close"].dropna()
    last_close = float(closes.iloc[-1])

    wk52_low = float(closes.min())
    wk52_high = float(closes.max())
    if wk52_high > wk52_low:
        wk52_position_pct = (last_close - wk52_low) / (wk52_high - wk52_low) * 100.0
    else:
        wk52_position_pct = None

    df["RSI_14"] = ta.rsi(df["Close"], length=14)
    rsi_14 = float(df["RSI_14"].iloc[-1]) if pd.notna(df["RSI_14"].iloc[-1]) else None

    bb = ta.bbands(df["Close"], length=20, std=2)
    if bb is None or bb.empty:
        return TechnicalSnapshot(
            last_close=last_close,
            rsi_14=rsi_14,
            bb_lower_20=None,
            bb_mid_20=None,
            bb_upper_20=None,
            wk52_low=wk52_low,
            wk52_high=wk52_high,
            wk52_position_pct=wk52_position_pct,
            reason="Failed to compute Bollinger Bands.",
        )

    lower_cols = [c for c in bb.columns if c.startswith("BBL_20")]
    mid_cols = [c for c in bb.columns if c.startswith("BBM_20")]
    upper_cols = [c for c in bb.columns if c.startswith("BBU_20")]

    if not lower_cols or not mid_cols or not upper_cols:
        return TechnicalSnapshot(
            last_close=last_close,
            rsi_14=rsi_14,
            bb_lower_20=None,
            bb_mid_20=None,
            bb_upper_20=None,
            wk52_low=wk52_low,
            wk52_high=wk52_high,
            wk52_position_pct=wk52_position_pct,
            reason="One or more Bollinger Band columns not found.",
        )

    df["BBL_20"] = bb[lower_cols[0]]
    df["BBM_20"] = bb[mid_cols[0]]
    df["BBU_20"] = bb[upper_cols[0]]

    bb_lower_20 = float(df["BBL_20"].iloc[-1]) if pd.notna(df["BBL_20"].iloc[-1]) else None
    bb_mid_20 = float(df["BBM_20"].iloc[-1]) if pd.notna(df["BBM_20"].iloc[-1]) else None
    bb_upper_20 = float(df["BBU_20"].iloc[-1]) if pd.notna(df["BBU_20"].iloc[-1]) else None

    return TechnicalSnapshot(
        last_close=last_close,
        rsi_14=rsi_14,
        bb_lower_20=bb_lower_20,
        bb_mid_20=bb_mid_20,
        bb_upper_20=bb_upper_20,
        wk52_low=wk52_low,
        wk52_high=wk52_high,
        wk52_position_pct=wk52_position_pct,
        reason="Technical indicators and 52-week range calculated.",
    )


# ---------- One-stop checkup ----------

def get_full_checkup(symbol: str) -> StockCheckup:
    symbol = symbol.upper()
    ticker_obj = yf.Ticker(symbol)

    try:
        identity = get_instrument_info(symbol, ticker_obj)
        fundamentals = compute_fundamentals(symbol, identity, ticker_obj)
        valuation = compute_valuation(identity, ticker_obj)
        market = get_market_regime()
        technical = compute_technical_snapshot(ticker_obj)
        return StockCheckup(
            identity=identity,
            fundamentals=fundamentals,
            valuation=valuation,
            market=market,
            technical=technical,
            error=None,
        )
    except Exception as e:
        # Fallback with partial info
        try:
            identity = get_instrument_info(symbol, ticker_obj)
        except Exception:
            identity = InstrumentInfo(
                symbol=symbol,
                short_name=None,
                long_name=None,
                quote_type=None,
                exchange=None,
                is_fund_like=False,
                fundamentals_skipped=False,
            )
        dummy_fund = FundamentalResult(
            ttm_net_income=None,
            total_debt=None,
            total_assets=None,
            debt_asset_ratio=None,
            source="None",
            reason=f"Error computing fundamentals: {e}",
        )
        dummy_val = ValuationSnapshot(
            pe_ratio=None,
            source="None",
            reason="Valuation not computed due to error.",
        )
        dummy_market = MarketRegimeResult(
            spy_last_close=float("nan"),
            spy_sma_200=float("nan"),
            spy_trend_ok=False,
        )
        dummy_tech = TechnicalSnapshot(
            last_close=float("nan"),
            rsi_14=None,
            bb_lower_20=None,
            bb_mid_20=None,
            bb_upper_20=None,
            wk52_low=None,
            wk52_high=None,
            wk52_position_pct=None,
            reason="Technical not evaluated due to error.",
        )
        return StockCheckup(
            identity=identity,
            fundamentals=dummy_fund,
            valuation=dummy_val,
            market=dummy_market,
            technical=dummy_tech,
            error=str(e),
        )

