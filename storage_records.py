# engine_screener.py

from typing import List

import pandas as pd

from engine_data_fetch import get_full_checkup
from models import ScreenerResult, StockCheckup


def evaluate_screener(checkup: StockCheckup, style: str = "value") -> ScreenerResult:
    """
    Apply simple screener rules on top of a StockCheckup.

    style:
        - "value": your value rules
        - "momentum": breakout rules
    """
    style = style.lower()
    if style not in {"value", "momentum"}:
        style = "value"

    ident = checkup.identity
    fund = checkup.fundamentals
    tech = checkup.technical
    mkt = checkup.market

    # Market rule
    market_ok = bool(mkt.spy_trend_ok)

    # Fundamental rule (stocks only)
    if ident.fundamentals_skipped:
        fundamentals_ok = True  # not applicable
    else:
        fundamentals_ok = (
            fund.ttm_net_income is not None
            and fund.ttm_net_income > 0
            and fund.debt_asset_ratio is not None
            and fund.debt_asset_ratio < 0.5
        )

    technical_ok = False
    notes: list[str] = []

    # Funds / ETFs
    if ident.is_fund_like:
        if tech.rsi_14 is None:
            technical_ok = False
            notes.append("Fund: RSI not available.")
        else:
            if style == "value":
                technical_ok = tech.rsi_14 < 30
                notes.append(f"Fund value rule: RSI < 30 ({tech.rsi_14:.1f}).")
            else:  # momentum
                technical_ok = tech.rsi_14 > 55
                notes.append(f"Fund momentum rule: RSI > 55 ({tech.rsi_14:.1f}).")
    else:
        # Stocks
        if style == "value":
            # Your value rules:
            #  - RSI between 25 and 45
            #  - Price < mid BB
            #  - Price > lower BB
            if tech.rsi_14 is None or any(
                x is None for x in (tech.bb_lower_20, tech.bb_mid_20)
            ):
                technical_ok = False
                notes.append("Stock: insufficient data for value rules (RSI/BB).")
            else:
                rsi_ok = 25 <= tech.rsi_14 <= 45
                discounted = tech.last_close < tech.bb_mid_20
                not_crashing = tech.last_close > tech.bb_lower_20
                technical_ok = rsi_ok and discounted and not_crashing
                notes.append(
                    f"Value rule: RSI {tech.rsi_14:.1f}, "
                    f"Close {tech.last_close:.2f} vs BB "
                    f"L={tech.bb_lower_20:.2f}, M={tech.bb_mid_20:.2f}."
                )
        else:
            # Momentum rule: breakout above upper band + reasonably strong RSI
            if tech.rsi_14 is None or tech.bb_upper_20 is None:
                technical_ok = False
                notes.append("Stock: insufficient data for momentum rules.")
            else:
                breakout = tech.last_close > tech.bb_upper_20
                strong_rsi = tech.rsi_14 >= 55
                technical_ok = breakout and strong_rsi
                notes.append(
                    f"Momentum rule: Close {tech.last_close:.2f} vs U={tech.bb_upper_20:.2f}, "
                    f"RSI {tech.rsi_14:.1f}."
                )

    if ident.is_fund_like:
        overall_ok = market_ok and technical_ok
    else:
        overall_ok = fundamentals_ok and market_ok and technical_ok

    reason = " ".join(notes)
    return ScreenerResult(
        checkup=checkup,
        style=style,
        fundamentals_ok=fundamentals_ok,
        market_ok=market_ok,
        technical_ok=technical_ok,
        overall_ok=overall_ok,
        reason=reason,
    )


def screen_single(symbol: str, style: str = "value") -> ScreenerResult:
    chk = get_full_checkup(symbol)
    return evaluate_screener(chk, style=style)


def screen_batch(symbols: List[str], style: str = "value") -> pd.DataFrame:
    rows = []
    for sym in symbols:
        res = screen_single(sym, style=style)
        chk = res.checkup

        rows.append(
            {
                "Symbol": chk.identity.symbol,
                "Style": res.style,
                "Type": "Fund" if chk.identity.is_fund_like else "Stock",
                "OverallPass": res.overall_ok,
                "FundamentalsOK": res.fundamentals_ok,
                "MarketOK": res.market_ok,
                "TechnicalOK": res.technical_ok,
                "Reason": res.reason,
                "TTMNetIncome": chk.fundamentals.ttm_net_income,
                "DebtAssetsRatio": chk.fundamentals.debt_asset_ratio,
                "RSI14": chk.technical.rsi_14,
                "LastClose": chk.technical.last_close,
            }
        )

    return pd.DataFrame(rows)

