# apps/stock_checkup_app.py

import os
import sys

import pandas as pd
import streamlit as st

# Ensure project root on path so we can import modules
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from engine_data_fetch import get_full_checkup  # noqa: E402
from engine_classify import (  # noqa: E402
    summarize_market_health,
    summarize_company_health,
    summarize_balance_sheet,
    classify_pe_zone,
    classify_range_zone,
    classify_rsi_zone,
    classify_price_zone,
)
from storage_records import save_checkup, load_history  # noqa: E402


def render_legend():
    st.info(
        "- **Environment (market health)**: SPY vs its 200-day SMA.\n"
        "- **Company health**:\n"
        "  • Profitability (TTM net income).\n"
        "  • Balance sheet (debt/assets).\n"
        "- **Body fat (valuation)**: P/E ratio.\n"
        "- **Temperature**: price position within 52-week low/high.\n"
        "- **Reflexes & posture (technicals)**:\n"
        "  • RSI(14) = reflexes.\n"
        "  • Price vs Bollinger bands = posture on the mat."
    )


def main():
    st.title("Stock Checkup – Health, Body Fat, Temperature & Reflexes")
    render_legend()

    # Sidebar: optional user-specific FMP key
    with st.sidebar:
        st.header("Session settings")

        user_fmp_key = st.text_input(
            "FMP API key (optional)",
            type="password",
            help=(
                "If you have your own free FMP key, paste it here. "
                "It will be used ONLY for this session to fetch fundamentals "
                "when Yahoo data is missing."
            ),
        )
        if user_fmp_key:
            # Per-session override; config.get_fmp_api_key() will pick this up
            st.session_state["fmp_api_key_user"] = user_fmp_key
        else:
            # Clear any previous key if the field is empty
            if "fmp_api_key_user" in st.session_state:
                del st.session_state["fmp_api_key_user"]

    ticker = st.text_input("Ticker symbol", value="LYB")

    if st.button("Run Checkup", type="primary"):
        if not ticker.strip():
            st.warning("Please enter a ticker.")
            return

        chk = get_full_checkup(ticker.strip())
        save_checkup(chk)

        if chk.error:
            st.error(f"Error during checkup: {chk.error}")

        # Identity
        st.markdown("### Identity")
        kind = "Fund/ETF" if chk.identity.is_fund_like else "Stock / Other"
        id_df = pd.DataFrame(
            {
                "Field": ["Symbol", "Name", "Quote Type", "Exchange", "Instrument class"],
                "Value": [
                    chk.identity.symbol,
                    chk.identity.long_name or chk.identity.short_name,
                    chk.identity.quote_type,
                    chk.identity.exchange,
                    kind,
                ],
            }
        )
        id_df["Value"] = id_df["Value"].astype(str)
        st.table(id_df)

        # Core health
        st.markdown("### Core health check")
        col_mkt, col_co, col_bs = st.columns(3)

        with col_mkt:
            st.write("**Environment (SPY)**")
            st.write(summarize_market_health(chk.market))

        with col_co:
            st.write("**Company profitability**")
            st.write(summarize_company_health(chk.fundamentals))

        with col_bs:
            st.write("**Balance sheet**")
            st.write(summarize_balance_sheet(chk.fundamentals))

        # Body fat & temperature
        st.markdown("### Body fat & temperature")

        pe_label, pe_desc = classify_pe_zone(
            chk.valuation.pe_ratio,
            chk.identity.is_fund_like,
        )
        range_label, range_desc = classify_range_zone(
            chk.technical.wk52_low,
            chk.technical.wk52_high,
            chk.technical.wk52_position_pct,
        )

        col_val, col_temp = st.columns(2)
        with col_val:
            st.write("**Body fat (P/E ratio)**")
            st.write(f"Zone: `{pe_label}`")
            extra = ""
            if chk.valuation.pe_ratio is not None and not pd.isna(chk.valuation.pe_ratio):
                extra = f"\n\nCurrent P/E ~ **{chk.valuation.pe_ratio:.1f}**."
            st.caption(pe_desc + extra)
        with col_temp:
            st.write("**Temperature (52-week range)**")
            st.write(f"Zone: `{range_label}`")
            extra = ""
            if (
                chk.technical.wk52_low is not None
                and chk.technical.wk52_high is not None
                and not pd.isna(chk.technical.wk52_low)
                and not pd.isna(chk.technical.wk52_high)
            ):
                extra += (
                    f"\n\n52w low: **{chk.technical.wk52_low:.2f}**, "
                    f"52w high: **{chk.technical.wk52_high:.2f}**."
                )
            if chk.technical.wk52_position_pct is not None and not pd.isna(
                chk.technical.wk52_position_pct
            ):
                extra += f" Current position: **{chk.technical.wk52_position_pct:.1f}%** of that range."
            st.caption(range_desc + extra)

        # Reflexes & posture
        st.markdown("### Reflexes & posture (technicals)")

        rsi_zone, rsi_desc = classify_rsi_zone(chk.technical.rsi_14)
        price_zone, price_desc = classify_price_zone(
            chk.technical.last_close,
            chk.technical.bb_lower_20,
            chk.technical.bb_mid_20,
            chk.technical.bb_upper_20,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.write("**RSI(14) – reflexes**")
            st.write(f"Zone: `{rsi_zone}`")
            extra_rsi = ""
            if chk.technical.rsi_14 is not None and not pd.isna(chk.technical.rsi_14):
                extra_rsi = f"\n\nRSI(14): **{chk.technical.rsi_14:.1f}**."
            st.caption(rsi_desc + extra_rsi)
        with c2:
            st.write("**Price vs 20-day Bollinger bands – posture**")
            st.write(f"Zone: `{price_zone}`")
            details = ""
            if not pd.isna(chk.technical.last_close):
                details = f"\n\nLast close: **{chk.technical.last_close:.2f}**."
            if (
                chk.technical.bb_lower_20 is not None
                and chk.technical.bb_mid_20 is not None
                and chk.technical.bb_upper_20 is not None
                and not pd.isna(chk.technical.bb_lower_20)
                and not pd.isna(chk.technical.bb_mid_20)
                and not pd.isna(chk.technical.bb_upper_20)
            ):
                details += (
                    f" Bands — Lower: **{chk.technical.bb_lower_20:.2f}**, "
                    f"Mid: **{chk.technical.bb_mid_20:.2f}**, "
                    f"Upper: **{chk.technical.bb_upper_20:.2f}**."
                )
            st.caption(price_desc + details)

        # History / medical record
        st.markdown("### Medical record (history)")
        hist = load_history(chk.identity.symbol)
        if hist.empty:
            st.caption("No historical records yet for this ticker.")
        else:
            st.line_chart(
                hist.set_index("ts")[["rsi"]],
                height=200,
                use_container_width=True,
            )
            st.line_chart(
                hist.set_index("ts")[["pe", "wk52_pct"]],
                height=200,
                use_container_width=True,
            )
            with st.expander("Raw history data"):
                st.dataframe(hist)


if __name__ == "__main__":
    main()

