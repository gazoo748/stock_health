# apps/screener_app.py

import os
import sys

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from engine_screener import screen_single, screen_batch  # noqa: E402
from engine_classify import summarize_market_health  # noqa: E402
from engine_data_fetch import get_market_regime  # noqa: E402


# ---------- helpers for dashboard visuals ----------


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Simple RSI implementation so this file doesn't depend on pandas_ta.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_price_history_with_indicators(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetch recent history and compute:
      - 20d SMA
      - 20d Bollinger Bands (± 2 std)
      - RSI(14)
    """
    hist = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if hist.empty or "Close" not in hist:
        return hist

    close = hist["Close"]
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    hist["SMA20"] = sma20
    hist["BB_upper"] = sma20 + 2 * std20
    hist["BB_lower"] = sma20 - 2 * std20
    hist["RSI14"] = compute_rsi(close, window=14)

    return hist


def make_price_bb_chart(hist: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["Close"],
            name="Close",
            mode="lines",
        )
    )

    if "SMA20" in hist:
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["SMA20"],
                name="SMA20",
                mode="lines",
                line=dict(dash="dot"),
            )
        )

    if "BB_upper" in hist and "BB_lower" in hist:
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["BB_upper"],
                name="BB Upper",
                mode="lines",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["BB_lower"],
                name="BB Lower",
                mode="lines",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title=f"{symbol} price with Bollinger Bands (20d, ±2σ)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    return fig


def make_rsi_chart(hist: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["RSI14"],
            name="RSI(14)",
            mode="lines",
        )
    )

    # Add "oversold" / "overbought" bands
    fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="lightgray", opacity=0.2)
    fig.add_hline(y=30, line=dict(dash="dash"))
    fig.add_hline(y=70, line=dict(dash="dash"))

    fig.update_layout(
        title=f"{symbol} RSI(14)",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=10, r=10, t=40, b=10),
        height=260,
    )
    return fig


def make_price_temperature_gauge(hist: pd.DataFrame, last_close: float) -> go.Figure:
    """
    Gauge: where is the current price in the last-52-week range?
    0% = 52w low, 100% = 52w high.
    """
    if hist.empty:
        value = 0
        low_52 = high_52 = last_close
    else:
        low_52 = float(hist["Close"].min())
        high_52 = float(hist["Close"].max())
        if high_52 == low_52:
            value = 0
        else:
            value = (last_close - low_52) / (high_52 - low_52) * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Price temperature\n(52-week range %)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_debt_pressure_gauge(debt_asset_ratio: float | None) -> go.Figure:
    """
    Gauge: how 'pressured' is the balance sheet?
    0 = no debt, 0.5 = your cutoff, >0.5 = above comfort zone.
    We'll map 0–1.0 to 0–100%.
    """
    if debt_asset_ratio is None:
        value = 0.0
    else:
        # Cap at 1.0+
        value = max(0.0, min(1.0, float(debt_asset_ratio))) * 100.0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Debt pressure\n(Debt / Assets %)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 25], "color": "green"},
                    {"range": [25, 50], "color": "yellow"},
                    # > 50% is above your cutoff
                    {"range": [50, 100], "color": "red"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------- main app ----------


def main():
    st.title("Value & Momentum Screener (Doctor’s Triage)")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Style")
        style = st.radio(
            "Select screener style",
            options=["value", "momentum"],
            index=0,
        )

        st.markdown("---")
        user_fmp_key = st.text_input(
            "FMP API key (optional)",
            type="password",
            help=(
                "Paste your own FMP key to use your personal 250-request quota. "
                "If left blank, the app will use its default key (if configured) "
                "or skip FMP fallback."
            ),
        )
        if user_fmp_key:
            st.session_state["fmp_api_key_user"] = user_fmp_key
        else:
            if "fmp_api_key_user" in st.session_state:
                del st.session_state["fmp_api_key_user"]

    # ---------------- Tabs ----------------
    tab_single, tab_batch = st.tabs(["Single ticker", "Batch"])

    # ---------------- Single Ticker ----------------
    with tab_single:
        ticker = st.text_input("Ticker symbol", value="LYB")

        if st.button("Run single screen", type="primary", key="single_btn"):
            if not ticker.strip():
                st.warning("Please enter a ticker.")
            else:
                res = screen_single(ticker.strip(), style=style)
                chk = res.checkup

                # --- transparency panel ---
                if chk.fundamentals.used_underlying:
                    st.info(
                        f"ℹ️ Fundamentals were derived from an inferred underlying common stock: "
                        f"**{chk.fundamentals.underlying_symbol}**. "
                        "This mapping is heuristic and may not be perfect."
                    )

                # --- overall header ---
                st.subheader(
                    f"Result for {chk.identity.symbol} "
                    f"({'Fund' if chk.identity.is_fund_like else 'Stock'})"
                )

                overall_label = "👍 PASS" if res.overall_ok else "👎 DO NOT BUY"
                st.subheader(f"Overall: {overall_label}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Style", res.style)
                with col2:
                    st.metric("Fundamentals OK?", "Yes" if res.fundamentals_ok else "No")
                with col3:
                    st.metric("Market OK?", "Yes" if res.market_ok else "No")
                with col4:
                    st.metric("Technical OK?", "Yes" if res.technical_ok else "No")

                # ---------------- Dashboard visuals ----------------
                hist = get_price_history_with_indicators(chk.identity.symbol, period="6mo")

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    if not hist.empty:
                        st.plotly_chart(
                            make_price_bb_chart(hist, chk.identity.symbol),
                            use_container_width=True,
                        )
                    else:
                        st.info("No price history available for Bollinger chart.")

                with chart_col2:
                    if not hist.empty and "RSI14" in hist:
                        st.plotly_chart(
                            make_rsi_chart(hist, chk.identity.symbol),
                            use_container_width=True,
                        )
                    else:
                        st.info("No RSI data available for this symbol.")

                gauge_col1, gauge_col2 = st.columns(2)
                with gauge_col1:
                    if not hist.empty:
                        st.plotly_chart(
                            make_price_temperature_gauge(hist, chk.technical.last_close),
                            use_container_width=True,
                        )
                    else:
                        st.info("Not enough history to compute price temperature.")

                with gauge_col2:
                    st.plotly_chart(
                        make_debt_pressure_gauge(chk.fundamentals.debt_asset_ratio),
                        use_container_width=True,
                    )

                # --- fundamentals block ---
                st.markdown("### Fundamentals")

                fund_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Source",
                            "TTM Net Income",
                            "Debt/Assets",
                            "Fundamentals OK?",
                            "Used underlying?",
                            "Underlying symbol",
                            "Notes",
                        ],
                        "Value": [
                            chk.fundamentals.source,
                            chk.fundamentals.ttm_net_income,
                            chk.fundamentals.debt_asset_ratio,
                            res.fundamentals_ok,
                            chk.fundamentals.used_underlying,
                            chk.fundamentals.underlying_symbol,
                            chk.fundamentals.reason,
                        ],
                    }
                )
                fund_df["Value"] = fund_df["Value"].astype(str)
                st.table(fund_df)

                # --- market ---
                st.markdown("### Market health (SPY)")
                st.write(summarize_market_health(chk.market))

                # --- notes ---
                st.markdown("### Notes")
                st.write(res.reason)

    # ---------------- Batch ----------------
    with tab_batch:
        st.write("Enter one ticker per line.")
        raw = st.text_area("Tickers", value="AAPL\nMSFT\nQQQ")

        if st.button("Run batch screen", type="primary", key="batch_btn"):
            symbols = [t.strip().upper() for t in raw.splitlines() if t.strip()]
            if not symbols:
                st.warning("Please enter at least one ticker.")
            else:
                df = screen_batch(symbols, style=style)
                st.subheader("Batch results")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"screen_results_{style}.csv",
                    mime="text/csv",
                )

        # Quick snapshot of environment
        st.markdown("### Current SPY environment")
        try:
            mkt = get_market_regime()
            st.write(summarize_market_health(mkt))
        except Exception as e:
            st.error(f"Failed to fetch SPY market regime: {e}")


if __name__ == "__main__":
    main()

