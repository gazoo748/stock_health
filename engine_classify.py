# engine_classify.py

from typing import Optional, Tuple

import pandas as pd

from models import FundamentalResult, ValuationSnapshot, MarketRegimeResult, TechnicalSnapshot


# ---------- Market ----------

def summarize_market_health(market: MarketRegimeResult) -> str:
    if pd.isna(market.spy_last_close) or pd.isna(market.spy_sma_200):
        return "⚪ Unknown (SPY data unavailable)"
    if market.spy_trend_ok:
        return f"🟢 Healthy environment: SPY {market.spy_last_close:.1f} > 200d SMA {market.spy_sma_200:.1f}"
    return f"🔴 Risk-off environment: SPY {market.spy_last_close:.1f} ≤ 200d SMA {market.spy_sma_200:.1f}"


# ---------- Company health ----------

def summarize_company_health(fund: FundamentalResult) -> str:
    if fund.ttm_net_income is None:
        if "skipped" in (fund.reason or "").lower():
            return "🟡 N/A (fund / non-common equity; profitability not evaluated)"
        return "⚪ Unknown (no net income data)."
    if fund.ttm_net_income > 0:
        return "🟢 Profitable (TTM net income > 0)."
    return "🔴 Not profitable (TTM net income ≤ 0)."


def summarize_balance_sheet(fund: FundamentalResult) -> str:
    if fund.debt_asset_ratio is None:
        return "⚪ Debt/Assets unknown."
    if fund.debt_asset_ratio < 0.3:
        return f"🟢 Conservative balance sheet (Debt/Assets {fund.debt_asset_ratio:.2f})."
    if fund.debt_asset_ratio < 0.5:
        return f"🟡 Moderate leverage (Debt/Assets {fund.debt_asset_ratio:.2f})."
    return f"🔴 Highly leveraged (Debt/Assets {fund.debt_asset_ratio:.2f})."


# ---------- Valuation (P/E) ----------

def classify_pe_zone(pe: Optional[float], is_fund_like: bool) -> Tuple[str, str]:
    if is_fund_like:
        return (
            "N/A for funds",
            "For funds/ETFs, P/E is usually not the main diagnostic tool."
        )
    if pe is None or pd.isna(pe):
        return "Unknown", "P/E not available or not meaningful."

    if pe < 10:
        return (
            "Very lean",
            f"P/E {pe:.1f} < 10 — very cheap; could be deep value or market expects trouble."
        )
    if pe < 15:
        return (
            "Lean",
            f"P/E {pe:.1f} in 10–15 — classic value territory for many sectors."
        )
    if pe < 25:
        return (
            "Normal",
            f"P/E {pe:.1f} in 15–25 — fairly typical for quality businesses."
        )
    if pe < 40:
        return (
            "Rich",
            f"P/E {pe:.1f} in 25–40 — paying up; market expects strong growth or high quality."
        )
    return (
        "Very rich",
        f"P/E {pe:.1f} ≥ 40 — extremely high multiple; expectations are sky-high."
    )


# ---------- RSI / bands / 52w range ----------

def classify_rsi_zone(rsi: Optional[float]) -> Tuple[str, str]:
    if rsi is None or pd.isna(rsi):
        return "Unknown", "RSI not available."
    if rsi < 30:
        return "Oversold (slow reflexes)", f"RSI {rsi:.1f} < 30 — washed out / oversold; reflexes sluggish."
    if rsi < 45:
        return "Value zone", f"RSI {rsi:.1f} in 30–45 — mild value region; reflexes waking up."
    if rsi < 60:
        return "Neutral", f"RSI {rsi:.1f} in 45–60 — normal reflex range."
    if rsi < 70:
        return "Momentum", f"RSI {rsi:.1f} in 60–70 — moving strong; reflexes very active."
    return "Overbought (hyper)", f"RSI {rsi:.1f} ≥ 70 — extended; reflexes twitchy."


def classify_price_zone(
    last_close: float,
    bb_lower: Optional[float],
    bb_mid: Optional[float],
    bb_upper: Optional[float],
) -> Tuple[str, str]:
    if any(v is None or pd.isna(v) for v in (last_close, bb_lower, bb_mid, bb_upper)):
        return "Unknown", "Bands not available to position price."
    if last_close < bb_lower:
        return (
            "Below lower band",
            f"Close {last_close:.2f} < Lower {bb_lower:.2f} — lying on the mat, extremely beaten down."
        )
    if last_close < bb_mid:
        return (
            "Value band (L–M)",
            f"Close {last_close:.2f} between Lower {bb_lower:.2f} and Mid {bb_mid:.2f} — crouched in the value zone."
        )
    if last_close < bb_upper:
        return (
            "Neutral band (M–U)",
            f"Close {last_close:.2f} between Mid {bb_mid:.2f} and Upper {bb_upper:.2f} — standing up, neutral posture."
        )
    return (
        "Breakout / extended",
        f"Close {last_close:.2f} ≥ Upper {bb_upper:.2f} — jumping above the mat; breakout / extended."
    )


def classify_range_zone(
    wk52_low: Optional[float],
    wk52_high: Optional[float],
    wk52_pct: Optional[float],
) -> Tuple[str, str]:
    if (
        wk52_low is None or pd.isna(wk52_low)
        or wk52_high is None or pd.isna(wk52_high)
        or wk52_pct is None or pd.isna(wk52_pct)
    ):
        return "Unknown", "52-week range not available."

    if wk52_pct < 20:
        return (
            "Near 52w lows (very cold)",
            f"Price is at {wk52_pct:.1f}% of its 52-week range — shivering near the lows."
        )
    if wk52_pct < 40:
        return (
            "Below mid-range (cool)",
            f"Price is at {wk52_pct:.1f}% of its 52-week range — still on the cool side."
        )
    if wk52_pct < 60:
        return (
            "Mid-range (mild)",
            f"Price is at {wk52_pct:.1f}% of its 52-week range — normal temperature."
        )
    if wk52_pct < 80:
        return (
            "Above mid-range (warm)",
            f"Price is at {wk52_pct:.1f}% of its 52-week range — getting warm."
        )
    return (
        "Near 52w highs (hot)",
        f"Price is at {wk52_pct:.1f}% of its 52-week range — running a fever near the highs."
    )

