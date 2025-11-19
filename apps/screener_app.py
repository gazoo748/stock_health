# apps/screener_app.py

import os
import sys

import pandas as pd
import streamlit as st

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from engine_screener import screen_single, screen_batch  # noqa: E402
from engine_classify import summarize_market_health  # noqa: E402
from engine_data_fetch import get_market_regime  # noqa: E402


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

