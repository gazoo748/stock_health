# models.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class InstrumentInfo:
    symbol: str
    short_name: Optional[str]
    long_name: Optional[str]
    quote_type: Optional[str]
    exchange: Optional[str]
    is_fund_like: bool
    fundamentals_skipped: bool


@dataclass
class FundamentalResult:
    ttm_net_income: Optional[float]
    total_debt: Optional[float]
    total_assets: Optional[float]
    debt_asset_ratio: Optional[float]
    source: str
    reason: str
    used_underlying: bool = False
    underlying_symbol: Optional[str] = None


@dataclass
class ValuationSnapshot:
    pe_ratio: Optional[float]
    source: str
    reason: str


@dataclass
class MarketRegimeResult:
    spy_last_close: float
    spy_sma_200: float
    spy_trend_ok: bool


@dataclass
class TechnicalSnapshot:
    last_close: float
    rsi_14: Optional[float]
    bb_lower_20: Optional[float]
    bb_mid_20: Optional[float]
    bb_upper_20: Optional[float]
    wk52_low: Optional[float]
    wk52_high: Optional[float]
    wk52_position_pct: Optional[float]  # 0–100% within 52w range
    reason: str  # notes about computation success/failure


@dataclass
class StockCheckup:
    identity: InstrumentInfo
    fundamentals: FundamentalResult
    valuation: ValuationSnapshot
    market: MarketRegimeResult
    technical: TechnicalSnapshot
    error: Optional[str] = None


@dataclass
class ScreenerResult:
    checkup: StockCheckup
    style: str  # "value" or "momentum"
    fundamentals_ok: bool
    market_ok: bool
    technical_ok: bool
    overall_ok: bool
    reason: str

