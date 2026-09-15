"""
Central configuration for the app.

Loads API keys from .env and decides whether each external service should
run in MOCK MODE (no key present -> use realistic fake data so the whole
pipeline still runs end-to-end for demos) or LIVE MODE (real API calls).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys -----------------------------------------------------------
SENTINEL_HUB_CLIENT_ID = os.getenv("SENTINEL_HUB_CLIENT_ID", "")
SENTINEL_HUB_CLIENT_SECRET = os.getenv("SENTINEL_HUB_CLIENT_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Mode flags -----------------------------------------------------------
# TODO: insert API key — once these are set in .env, the flags below flip
# to False automatically and the real API calls in agents/*.py kick in.
SENTINEL_HUB_MOCK = not (SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET)
GEMINI_MOCK = not GEMINI_API_KEY

# --- Geography ------------------------------------------------------------
# Rough bounding box for Maharashtra state (lon_min, lat_min, lon_max, lat_max)
MAHARASHTRA_BBOX = (72.6, 15.6, 80.9, 22.1)

# Max area (in km^2) a user can select in one query — keeps Sentinel Hub
# requests small and fast for a demo / free-tier account.
MAX_AREA_KM2 = 2500

# Max acceptable cloud cover percentage before the Input Agent triggers a
# re-query with a shifted date range.
MAX_CLOUD_COVER_PCT = 30

# Default date range width (days) to search for a clear-sky image.
DATE_RANGE_DAYS = 14
