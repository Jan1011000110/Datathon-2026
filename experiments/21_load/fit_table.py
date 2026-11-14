"""Fit table for the final agent: every train hour with a valid meter reading (6,187 h), not only
the 5,993 h of the official table (which drops hours where solar is missing, e.g. 17-24 August,
the days right before the August test week). Not a runtime module (imports src)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_hourly_full  # noqa: E402
from src.hard import METEO_MAP  # noqa: E402


def load_fit_table():
    return load_hourly_full().rename(columns=METEO_MAP)[["load_kw"] + list(METEO_MAP.values())]
