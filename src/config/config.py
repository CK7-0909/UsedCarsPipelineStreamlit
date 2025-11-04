import os
from pathlib import Path

try:
    import streamlit as st  # type: ignore
except ImportError:  # Streamlit not available outside the app
    st = None  # type: ignore[assignment]


def _load_local_env() -> None:
    """Populate os.environ with values from a repository-level .env (if present)."""
    if getattr(_load_local_env, "_loaded", False):
        return

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip("\"'")  # tolerate quoted values
            os.environ.setdefault(key, value)

    _load_local_env._loaded = True  # type: ignore[attr-defined]


def _get_secret(name: str) -> str:
    """
    Resolve a configuration value in priority order:
    1. Streamlit secrets (Streamlit Cloud or local `.streamlit/secrets.toml`)
    2. Environment variables (e.g. set by Jenkins, shell export, Docker, etc.)
    3. Values loaded from a repository-level `.env` file for local development
    """
    _load_local_env()

    if st is not None:
        try:
            value = st.secrets[name]  # type: ignore[index]
            if value:
                return str(value)
        except Exception:
            pass

    value = os.getenv(name)
    if value:
        return value

    raise RuntimeError(
        f"Required configuration '{name}' not found. "
        "Set it via Streamlit secrets, environment variables, or a local .env file."
    )


# SNOWFLAKE_USER = _get_secret("SNOWFLAKE_USER")
# SNOWFLAKE_PASSWORD = _get_secret("SNOWFLAKE_PASSWORD")
# SNOWFLAKE_ACCOUNT = _get_secret("SNOWFLAKE_ACCOUNT")
# SNOWFLAKE_WH = _get_secret("SNOWFLAKE_WH")
# SNOWFLAKE_DB = _get_secret("SNOWFLAKE_DB")
# SNOWFLAKE_SCHEMA = _get_secret("SNOWFLAKE_SCHEMA")

SNOWFLAKE_USER = "c0kalv01"
SNOWFLAKE_PASSWORD = "Lamborgini@0909"
SNOWFLAKE_ACCOUNT = "PHHTHAA-YSB95590"
SNOWFLAKE_WH = "COMPUTE_WH"
SNOWFLAKE_DB = "USED_VEHICLE_ANALYTICS"
SNOWFLAKE_SCHEMA = "DEV_SCHEMA_MART"
SNOWFLAKE_ROLE = "ACCOUNTADMIN"