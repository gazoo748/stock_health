# config.py

from __future__ import annotations
import os

SPY_TICKER = "SPY"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


def get_fmp_api_key() -> str | None:
    """
    Priority order:
      1. User-provided key in this Streamlit session (st.session_state["fmp_api_key_user"])
      2. Streamlit secrets (FMP_API_KEY)
      3. Environment variable FMP_API_KEY

    This lets you run the app publicly:
      - If the user pastes a key, THEIR key is used.
      - Otherwise, your app-wide key (secrets/env) is used if present.
      - If nothing is set, FMP simply won't be called.
    """
    # 1) User-provided key (per session)
    try:
        import streamlit as st  # type: ignore[import-not-found]
        user_key = st.session_state.get("fmp_api_key_user")
        if user_key:
            return user_key.strip()
    except Exception:
        # Not running under Streamlit, or session_state unavailable
        pass

    # 2) App-level secret key (Streamlit Cloud / local .streamlit/secrets.toml)
    try:
        import streamlit as st  # type: ignore[import-not-found]
        secret_key = st.secrets.get("FMP_API_KEY", None)
        if secret_key:
            return secret_key.strip()
    except Exception:
        pass

    # 3) Environment variable
    env_key = os.environ.get("FMP_API_KEY")
    if env_key:
        return env_key.strip()

    # Nothing configured
    return None

