from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ============================================================
# FUTURES CURVE TERMINAL — STREAMLIT SINGLE-FILE EDITION
# Bloomberg-style terminal aesthetic for education / prototyping
# ============================================================

APP_TZ = "Europe/Paris"
DEFAULT_MONTHS = 18
DEFAULT_RF = 4.00  # user-adjustable in sidebar
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

COMMODITY_UNIVERSE: Dict[str, Dict[str, str]] = {
    "WTI Crude": {"sector": "energy", "ticker": "CL"},
    "Brent Crude": {"sector": "energy", "ticker": "CO"},
    "Natural Gas": {"sector": "energy", "ticker": "NG"},
    "RBOB Gasoline": {"sector": "energy", "ticker": "RB"},
    "Heating Oil": {"sector": "energy", "ticker": "HO"},
    "Gold": {"sector": "metals", "ticker": "GC"},
    "Silver": {"sector": "metals", "ticker": "SI"},
    "Copper": {"sector": "metals", "ticker": "HG"},
    "Platinum": {"sector": "metals", "ticker": "PL"},
    "Palladium": {"sector": "metals", "ticker": "PA"},
    "Wheat": {"sector": "agriculture", "ticker": "ZW"},
    "Corn": {"sector": "agriculture", "ticker": "ZC"},
    "Soybeans": {"sector": "agriculture", "ticker": "ZS"},
    "Sugar": {"sector": "agriculture", "ticker": "SB"},
    "Coffee": {"sector": "agriculture", "ticker": "KC"},
    "Cotton": {"sector": "agriculture", "ticker": "CT"},
}

MARKET_META: Dict[str, Dict[str, Any]] = {
    "WTI Crude": {"symbol": "$", "unit": "$/bbl", "color": "#f6ad55", "storage_cost": 3.5, "base_spot": 78.40, "base_vol": 0.015},
    "Brent Crude": {"symbol": "$", "unit": "$/bbl", "color": "#ed8936", "storage_cost": 3.2, "base_spot": 82.10, "base_vol": 0.014},
    "Natural Gas": {"symbol": "$", "unit": "$/MMBtu", "color": "#4299e1", "storage_cost": 8.0, "base_spot": 2.18, "base_vol": 0.035},
    "RBOB Gasoline": {"symbol": "$", "unit": "$/gal", "color": "#e53e3e", "storage_cost": 4.0, "base_spot": 2.41, "base_vol": 0.018},
    "Heating Oil": {"symbol": "$", "unit": "$/gal", "color": "#dd6b20", "storage_cost": 3.8, "base_spot": 2.58, "base_vol": 0.016},
    "Gold": {"symbol": "$", "unit": "$/oz", "color": "#f0b429", "storage_cost": 0.5, "base_spot": 2341.0, "base_vol": 0.006},
    "Silver": {"symbol": "$", "unit": "$/oz", "color": "#a0aec0", "storage_cost": 1.2, "base_spot": 27.80, "base_vol": 0.012},
    "Copper": {"symbol": "$", "unit": "$/lb", "color": "#fc8181", "storage_cost": 2.8, "base_spot": 4.12, "base_vol": 0.010},
    "Platinum": {"symbol": "$", "unit": "$/oz", "color": "#b794f4", "storage_cost": 1.0, "base_spot": 968.0, "base_vol": 0.009},
    "Palladium": {"symbol": "$", "unit": "$/oz", "color": "#4fd1c5", "storage_cost": 1.1, "base_spot": 1042.0, "base_vol": 0.015},
    "Wheat": {"symbol": "¢", "unit": "¢/bu", "color": "#48bb78", "storage_cost": 5.5, "base_spot": 542.0, "base_vol": 0.012},
    "Corn": {"symbol": "¢", "unit": "¢/bu", "color": "#68d391", "storage_cost": 4.2, "base_spot": 438.0, "base_vol": 0.010},
    "Soybeans": {"symbol": "¢", "unit": "¢/bu", "color": "#9ae6b4", "storage_cost": 4.8, "base_spot": 1148.0, "base_vol": 0.009},
    "Sugar": {"symbol": "$", "unit": "¢/lb", "color": "#f687b3", "storage_cost": 6.0, "base_spot": 21.4, "base_vol": 0.018},
    "Coffee": {"symbol": "$", "unit": "¢/lb", "color": "#b7791f", "storage_cost": 7.0, "base_spot": 198.5, "base_vol": 0.020},
    "Cotton": {"symbol": "¢", "unit": "¢/lb", "color": "#fefcbf", "storage_cost": 5.2, "base_spot": 84.3, "base_vol": 0.011},
}

SCENARIO_PARAMS: Dict[str, Dict[str, float]] = {
    "Current Market": {"spot_shock": 0.00, "curve_tilt": 0.000, "vol_mult": 1.0, "convenience_boost": 0.0},
    "Supply Shock (Backwardation)": {"spot_shock": 0.08, "curve_tilt": -0.012, "vol_mult": 1.6, "convenience_boost": 4.0},
    "Demand Collapse (Contango)": {"spot_shock": -0.10, "curve_tilt": 0.008, "vol_mult": 1.4, "convenience_boost": -2.0},
    "Geopolitical Spike": {"spot_shock": 0.15, "curve_tilt": -0.018, "vol_mult": 2.2, "convenience_boost": 6.0},
    "Normalising Curve": {"spot_shock": 0.00, "curve_tilt": 0.003, "vol_mult": 0.8, "convenience_boost": -1.0},
}

CURVE_PROFILES: Dict[str, Dict[str, Any]] = {
    "WTI Crude": {"base_slope": -0.004, "convenience_yield": 3.8, "mean_reversion": 0.15, "noise": 0.002, "structural": "backwardation"},
    "Brent Crude": {"base_slope": -0.003, "convenience_yield": 3.2, "mean_reversion": 0.14, "noise": 0.002, "structural": "backwardation"},
    "Natural Gas": {"base_slope": 0.010, "convenience_yield": 1.2, "mean_reversion": 0.25, "noise": 0.008, "structural": "seasonal"},
    "RBOB Gasoline": {"base_slope": -0.006, "convenience_yield": 4.5, "mean_reversion": 0.20, "noise": 0.003, "structural": "seasonal"},
    "Heating Oil": {"base_slope": -0.005, "convenience_yield": 4.0, "mean_reversion": 0.18, "noise": 0.003, "structural": "seasonal"},
    "Gold": {"base_slope": 0.004, "convenience_yield": 0.3, "mean_reversion": 0.05, "noise": 0.001, "structural": "contango"},
    "Silver": {"base_slope": 0.004, "convenience_yield": 0.5, "mean_reversion": 0.06, "noise": 0.002, "structural": "contango"},
    "Copper": {"base_slope": -0.002, "convenience_yield": 2.1, "mean_reversion": 0.12, "noise": 0.003, "structural": "mixed"},
    "Platinum": {"base_slope": 0.002, "convenience_yield": 0.8, "mean_reversion": 0.08, "noise": 0.003, "structural": "contango"},
    "Palladium": {"base_slope": -0.005, "convenience_yield": 3.5, "mean_reversion": 0.15, "noise": 0.004, "structural": "backwardation"},
    "Wheat": {"base_slope": 0.003, "convenience_yield": 2.8, "mean_reversion": 0.20, "noise": 0.004, "structural": "seasonal"},
    "Corn": {"base_slope": 0.002, "convenience_yield": 2.2, "mean_reversion": 0.18, "noise": 0.003, "structural": "seasonal"},
    "Soybeans": {"base_slope": -0.001, "convenience_yield": 2.5, "mean_reversion": 0.16, "noise": 0.003, "structural": "seasonal"},
    "Sugar": {"base_slope": 0.005, "convenience_yield": 1.8, "mean_reversion": 0.22, "noise": 0.005, "structural": "contango"},
    "Coffee": {"base_slope": 0.006, "convenience_yield": 1.5, "mean_reversion": 0.25, "noise": 0.006, "structural": "contango"},
    "Cotton": {"base_slope": 0.003, "convenience_yield": 2.0, "mean_reversion": 0.20, "noise": 0.004, "structural": "mixed"},
}

SEASONAL_DRIVERS: Dict[str, Dict[str, Any]] = {
    "WTI Crude": {"text": "Summer driving season tightens front contracts; shoulder months often soften as refinery runs shift.", "peaks": [5, 6, 7, 10, 11], "troughs": [1, 2, 8, 9], "calendar": "Long M1/Short M3 into driving season; reassess after Labour Day."},
    "Brent Crude": {"text": "Brent tends to hold firmer backwardation than WTI because of seaborne demand pull and tighter prompt barrels.", "peaks": [4, 5, 6, 10, 11], "troughs": [1, 2, 8], "calendar": "Long Brent/Short WTI in strong Q2 tightening phases."},
    "Natural Gas": {"text": "Injection season often builds contango; winter withdrawal can create sharp prompt backwardation.", "peaks": [10, 11, 0, 1, 2], "troughs": [3, 4, 5, 6, 7, 8, 9], "calendar": "Long winter gas / short shoulder months when storage starts tightening."},
    "RBOB Gasoline": {"text": "Summer blend and driving season usually support front spreads in late winter and spring.", "peaks": [1, 2, 3, 4, 5], "troughs": [8, 9, 10], "calendar": "Long spring gasoline strength versus later months."},
    "Heating Oil": {"text": "Winter demand often tightens front contracts; storage rebuild months soften the structure.", "peaks": [9, 10, 11, 0, 1], "troughs": [3, 4, 5, 6], "calendar": "Long heating oil versus gasoline into colder months."},
    "Gold": {"text": "Gold usually shows mild financing-led contango with occasional physical demand pulses.", "peaks": [8, 9, 10, 11], "troughs": [2, 3, 6, 7], "calendar": "Curve trade is usually secondary to macro rate and USD regime."},
    "Silver": {"text": "Silver blends industrial and monetary demand, so seasonality is weaker than in agriculture or energy.", "peaks": [0, 1, 4, 8, 9], "troughs": [5, 6, 7], "calendar": "Ratio and carry trades tend to dominate pure seasonal curve trades."},
    "Copper": {"text": "Chinese restocking and construction activity often shape copper curve tightness around spring periods.", "peaks": [1, 2, 3, 4, 8, 9], "troughs": [5, 6, 11, 0], "calendar": "Watch prompt copper for restocking-led backwardation."},
    "Platinum": {"text": "Seasonality is modest; supply disruptions and automotive demand matter more.", "peaks": [0, 1, 2, 9, 10], "troughs": [5, 6, 7], "calendar": "Relative-value spread versus palladium is often cleaner than outright seasonality."},
    "Palladium": {"text": "Physical tightness can generate prompt squeezes and lease-rate spikes.", "peaks": [0, 1, 9, 10, 11], "troughs": [5, 6, 7], "calendar": "Watch lease rates and prompt spreads for squeeze dynamics."},
    "Wheat": {"text": "Pre-harvest weather risk can tighten the curve; harvest pressure often rebuilds carry.", "peaks": [1, 2, 3, 4, 9, 10], "troughs": [5, 6, 7, 8], "calendar": "Pre-harvest weather premium trades can be attractive when crop risk rises."},
    "Corn": {"text": "Planting, pollination, and harvest windows dominate curve behaviour.", "peaks": [1, 2, 3, 4, 5, 6], "troughs": [8, 9, 10], "calendar": "Old-crop/new-crop spread is usually the key seasonal trade."},
    "Soybeans": {"text": "US and Brazilian crop cycles create alternating seasonal tightness and carry phases.", "peaks": [0, 1, 3, 4, 5, 6], "troughs": [8, 9, 10], "calendar": "Calendar spreads often reflect origin risk and harvest timing."},
    "Sugar": {"text": "Brazilian crop timing and energy linkage through ethanol shape the curve.", "peaks": [0, 1, 2, 3, 9, 10, 11], "troughs": [5, 6, 7, 8], "calendar": "Pre-harvest risk premium often matters more than passive carry alone."},
    "Coffee": {"text": "Brazil weather and biennial crop effects can create strong seasonal and cyclical distortions.", "peaks": [4, 5, 6, 7, 8], "troughs": [1, 2, 9, 10], "calendar": "Frost-risk windows often dominate the prompt structure."},
    "Cotton": {"text": "Planting, export demand, and harvest timing shape old-crop/new-crop relationships.", "peaks": [3, 4, 5, 8, 9], "troughs": [10, 11, 0, 1], "calendar": "Watch export season premiums and index roll effects."},
}


@dataclass
class CurveData:
    commodity: str
    scenario: str
    months: int
    spot: float
    prices: List[float]
    structure: str
    convenience_yield: float
    risk_free: float
    storage_cost: float
    model_mode: str = "Synthetic Demo"


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def rng_for(*parts: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed("|".join(parts)))


def generate_term_structure(commodity: str, scenario: str, months: int, risk_free_pct: float) -> CurveData:
    meta = MARKET_META[commodity]
    profile = CURVE_PROFILES[commodity]
    params = SCENARIO_PARAMS[scenario]
    rng = rng_for("curve", commodity, scenario, str(months), f"{risk_free_pct:.4f}")

    spot = meta["base_spot"] * (1 + params["spot_shock"])
    raw_cy = profile["convenience_yield"] + params["convenience_boost"]
    convenience_yield = max(0.0, raw_cy)

    storage_monthly = meta["storage_cost"] / 100.0 / 12.0
    convenience_monthly = convenience_yield / 100.0 / 12.0
    rf_monthly = risk_free_pct / 100.0 / 12.0
    base_slope = profile["base_slope"] + params["curve_tilt"]

    prices: List[float] = []
    seasonal_phase = rng.uniform(-0.8, 0.8)

    for t in range(months):
        month_num = t + 1
        coc_price = spot * np.exp((rf_monthly + storage_monthly - convenience_monthly) * month_num)
        mr_factor = 1 - np.exp(-profile["mean_reversion"] * month_num / 12.0)
        tilt = base_slope * month_num
        cyclic = profile["noise"] * np.sin(month_num * 2.35 + seasonal_phase) * params["vol_mult"]
        random_micro = rng.normal(0, profile["noise"] * 0.12)
        price = spot * (1 + tilt + cyclic + random_micro) * (1 - mr_factor) + coc_price * mr_factor
        prices.append(max(price, spot * 0.5))

    if len(prices) >= 3:
        front_slope = (prices[2] - prices[0]) / prices[0]
        structure = "BACKWARDATION" if front_slope < -0.005 else "CONTANGO" if front_slope > 0.005 else "FLAT"
    else:
        structure = "FLAT"

    return CurveData(
        commodity=commodity,
        scenario=scenario,
        months=months,
        spot=spot,
        prices=prices,
        structure=structure,
        convenience_yield=convenience_yield,
        risk_free=risk_free_pct,
        storage_cost=meta["storage_cost"],
    )


def compute_roll_yield(data: CurveData) -> float:
    if len(data.prices) < 2:
        return 0.0
    return ((data.prices[0] - data.prices[1]) / data.prices[1]) * 12 * 100


def compute_basis(data: CurveData) -> float:
    return ((data.spot - data.prices[0]) / data.spot) * 100


def curve_slope_pct(data: CurveData) -> float:
    return ((data.prices[-1] - data.prices[0]) / data.prices[0]) * 100


def get_seasonal_pattern(commodity: str) -> Dict[str, Any]:
    rng = rng_for("seasonal", commodity)
    info = SEASONAL_DRIVERS[commodity]
    avg_roll, std_roll, backwardation_pct = [], [], []

    for month in range(12):
        if month in info["peaks"]:
            base, std, freq = rng.uniform(1.5, 4.5), rng.uniform(0.8, 2.0), rng.uniform(60, 90)
        elif month in info["troughs"]:
            base, std, freq = rng.uniform(-3.5, -0.5), rng.uniform(0.8, 2.2), rng.uniform(20, 45)
        else:
            base, std, freq = rng.uniform(-1.0, 1.5), rng.uniform(0.6, 1.5), rng.uniform(40, 62)
        avg_roll.append(round(float(base), 2))
        std_roll.append(round(float(std), 2))
        backwardation_pct.append(round(float(freq), 1))

    return {
        "avg_roll": avg_roll,
        "std_roll": std_roll,
        "backwardation_pct": backwardation_pct,
        "driver_text": info["text"],
        "calendar_trade": info["calendar"],
        "mode": "Synthetic seasonal composite",
    }


def simulate_basis_history(data: CurveData, days: int = 90) -> pd.DataFrame:
    rng = rng_for("basis", data.commodity, data.scenario, str(days), f"{data.risk_free:.2f}")
    dates = pd.date_range(end=datetime.now(ZoneInfo(APP_TZ)), periods=days, freq="D")
    current_basis = compute_basis(data)

    shocks = rng.normal(0, 0.08, days)
    basis = current_basis + np.cumsum(shocks)
    basis = np.clip(basis, -8, 8)

    spot_returns = rng.normal(0, MARKET_META[data.commodity]["base_vol"] / 2.5, days)
    spot_series = data.spot * np.cumprod(1 + spot_returns)
    front_series = spot_series * (1 + basis / 100)

    return pd.DataFrame({
        "date": dates,
        "spot": spot_series,
        "front": front_series,
        "basis": basis,
    })


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(cleaned[start:end + 1])


def call_claude(api_key: str, system: str, user: str, max_tokens: int = 1200) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None
    try:
        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=45,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"]
        return extract_json(raw)
    except Exception as exc:
        st.error(f"AI engine error: {exc}")
        return None


def ai_curve_structure(api_key: str, data: CurveData) -> Optional[Dict[str, Any]]:
    prices = data.prices
    m1 = prices[0]
    m3 = prices[2] if len(prices) > 2 else prices[-1]
    m6 = prices[5] if len(prices) > 5 else prices[-1]
    system = "You are a senior commodities strategist. Output valid JSON only. No markdown fence, no extra prose."
    user = f"""
Commodity: {data.commodity}
Scenario: {data.scenario}
Mode: {data.model_mode}
Spot: {data.spot:.2f}
M1: {m1:.2f}
M3: {m3:.2f}
M6: {m6:.2f}
Structure: {data.structure}
Convenience yield: {data.convenience_yield:.1f}%
Risk-free: {data.risk_free:.1f}%
Storage: {data.storage_cost:.1f}%
M1-M3 spread: {(m3-m1)/m1*100:+.2f}%
M1-M6 spread: {(m6-m1)/m1*100:+.2f}%

Return JSON with keys: steps (array of objects with label/content/conclusion), trade_idea, risk_to_view, conviction.
"""
    return call_claude(api_key, system, user)


def ai_carry(api_key: str, data: CurveData, roll_yields: List[float], avg_roll: float) -> Optional[Dict[str, Any]]:
    system = "You are a carry specialist in commodity futures. Output valid JSON only."
    roll_str = ", ".join([f"M{i+1}->{i+2}: {val:+.2f}%" for i, val in enumerate(roll_yields[:12])])
    user = f"""
Commodity: {data.commodity}
Scenario: {data.scenario}
Mode: {data.model_mode}
Roll yields: {roll_str}
Average roll yield: {avg_roll:+.2f}%
Risk-free: {data.risk_free:.1f}%
Storage: {data.storage_cost:.1f}%
Convenience yield: {data.convenience_yield:.1f}%
Net carry: {data.risk_free + data.storage_cost - data.convenience_yield:+.2f}%

Return JSON with keys: steps (array of label/content/conclusion), trade, risk.
"""
    return call_claude(api_key, system, user)


def ai_seasonal(api_key: str, commodity: str, seasonal: Dict[str, Any], current_month: int, current_roll: float) -> Optional[Dict[str, Any]]:
    months_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    system = "You are a seasonal commodity analyst. Output valid JSON only."
    user = f"""
Commodity: {commodity}
Mode: {seasonal.get('mode', 'Synthetic')}
Month: {months_names[current_month]}
Historical average roll this month: {seasonal['avg_roll'][current_month]:+.2f}%
Current roll: {current_roll:+.2f}%
Backwardation frequency: {seasonal['backwardation_pct'][current_month]:.1f}%
Driver text: {seasonal['driver_text']}
Calendar trade: {seasonal['calendar_trade']}

Return JSON with keys: steps (array of label/content/conclusion), trade, exit_signal, confirmation_signals.
"""
    return call_claude(api_key, system, user)


def ai_cross(api_key: str, snapshot: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    system = "You are a cross-commodity strategist. Output valid JSON only."
    lines = []
    for commodity, row in snapshot.items():
        lines.append(f"{commodity}: structure={row['structure']}, roll={row['roll_yield']:+.2f}%, sector={row['sector']}")
    user = "Cross-commodity snapshot:\n" + "\n".join(lines) + "\nReturn JSON with keys: steps, spread_trade, macro_read, best_long, best_short."
    return call_claude(api_key, system, user)


def metric_html(label: str, value: str, sub: str = "", color: str = "#e2e8f0") -> str:
    return f'''<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:{color}">{value}</div><div class="metric-sub">{sub}</div></div>'''


def info_box(label: str, content: str, color: str) -> None:
    st.markdown(
        f'''<div style="padding:12px;background:{color}0d;border:1px solid {color}33;border-radius:4px;margin-top:8px;font-size:12px;color:#cbd5e1;line-height:1.7;">
        <span style="font-family:IBM Plex Mono,monospace;font-size:9px;color:{color};">{label}</span><br>{content}</div>''',
        unsafe_allow_html=True,
    )


def render_steps(result: Optional[Dict[str, Any]]) -> None:
    if not result:
        return
    for idx, step in enumerate(result.get("steps", []), start=1):
        st.markdown(
            f'''<div class="ai-step"><div class="ai-step-label">STEP {idx} · {step.get("label", "")}</div>
            {step.get("content", "")}
            <div class="ai-conclusion">→ {step.get("conclusion", "")}</div></div>''',
            unsafe_allow_html=True,
        )


def apply_plot_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#0b0f1a",
        plot_bgcolor="#06080f",
        font=dict(family="IBM Plex Mono, monospace", color="#94a3b8", size=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1, font=dict(size=10)),
        margin=dict(l=10, r=10, t=36, b=10),
        hoverlabel=dict(bgcolor="#0f1520", bordercolor="rgba(255,255,255,0.2)", font=dict(family="IBM Plex Mono, monospace", size=11)),
        title=dict(text=title, font=dict(size=11, color="#4a5568", family="IBM Plex Mono, monospace"), x=0.01, y=0.98),
        height=height,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", showgrid=True, zeroline=False, linecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", showgrid=True, zeroline=False, linecolor="rgba(255,255,255,0.1)")
    return fig


@st.cache_data(show_spinner=False, ttl=60)
def get_curve_data(commodity: str, scenario: str, months: int, risk_free_pct: float) -> CurveData:
    return generate_term_structure(commodity, scenario, months, risk_free_pct)


def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #06080f; color: #e2e8f0; }
.stApp { background-color: #06080f; }
[data-testid="stSidebar"] { background-color: #0b0f1a; border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p { color: #94a3b8 !important; font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.1em; }
.metric-card { background:#0b0f1a; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:14px 16px; text-align:center; }
.metric-label { font-family:'IBM Plex Mono', monospace; font-size:9px; letter-spacing:0.18em; color:#4a5568; margin-bottom:6px; }
.metric-value { font-family:'IBM Plex Mono', monospace; font-size:21px; font-weight:700; }
.metric-sub { font-family:'IBM Plex Mono', monospace; font-size:10px; color:#64748b; margin-top:4px; }
.section-header { font-family:'IBM Plex Mono', monospace; font-size:10px; letter-spacing:0.2em; color:#4a5568; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px; margin-bottom:16px; margin-top:8px; }
.ai-box { background:#0f1520; border:1px solid rgba(167,139,250,0.2); border-left:3px solid #b794f4; border-radius:6px; padding:16px; font-size:13px; line-height:1.8; color:#cbd5e1; }
.ai-step { background:#0b0f1a; border-left:3px solid rgba(240,180,41,0.4); border-radius:4px; padding:10px 14px; margin-bottom:8px; font-size:12px; line-height:1.7; color:#94a3b8; }
.ai-step-label { font-family:'IBM Plex Mono', monospace; font-size:9px; color:#f0b429; letter-spacing:0.15em; margin-bottom:5px; }
.ai-conclusion { font-family:'IBM Plex Mono', monospace; font-size:11px; color:#e2e8f0; border-top:1px solid rgba(255,255,255,0.06); padding-top:7px; margin-top:7px; }
.pill { display:inline-block; padding:3px 10px; border-radius:3px; font-family:'IBM Plex Mono', monospace; font-size:10px; font-weight:600; letter-spacing:0.08em; }
.pill-green { background:rgba(72,187,120,0.15); color:#48bb78; border:1px solid rgba(72,187,120,0.3); }
.pill-red { background:rgba(252,129,129,0.15); color:#fc8181; border:1px solid rgba(252,129,129,0.3); }
.pill-gold { background:rgba(240,180,41,0.15); color:#f0b429; border:1px solid rgba(240,180,41,0.3); }
.stButton > button { background:rgba(66,153,225,0.12) !important; border:1px solid rgba(66,153,225,0.35) !important; color:#4299e1 !important; font-family:'IBM Plex Mono', monospace !important; font-size:10px !important; letter-spacing:0.15em !important; border-radius:4px !important; padding:8px 18px !important; }
.stTabs [data-baseweb="tab-list"] { background:transparent; border-bottom:1px solid rgba(255,255,255,0.06); gap:4px; }
.stTabs [data-baseweb="tab"] { font-family:'IBM Plex Mono', monospace; font-size:10px; letter-spacing:0.12em; color:#4a5568; background:transparent; border:none; padding:8px 16px; }
.stTabs [aria-selected="true"] { color:#4299e1 !important; background:rgba(66,153,225,0.08) !important; border-bottom:2px solid #4299e1 !important; }
.stSelectbox > div > div, .stMultiSelect > div > div { background:#0b0f1a !important; border:1px solid rgba(255,255,255,0.1) !important; color:#e2e8f0 !important; font-family:'IBM Plex Mono', monospace !important; font-size:12px !important; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""",
        unsafe_allow_html=True,
    )


def build_snapshot(data_map: Dict[str, CurveData]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for commodity, data in data_map.items():
        snapshot[commodity] = {
            "structure": data.structure,
            "roll_yield": compute_roll_yield(data),
            "sector": COMMODITY_UNIVERSE[commodity]["sector"],
        }
    return snapshot


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="Futures Curve Terminal", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    now = datetime.now(ZoneInfo(APP_TZ))

    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:16px 0 20px 0;"><div style="font-family:IBM Plex Mono,monospace;font-size:22px;color:#f0b429;">◈</div><div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#e2e8f0;letter-spacing:0.15em;">FUTURES CURVE</div><div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#4a5568;letter-spacing:0.2em;">TERMINAL</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-header">MARKET SELECTION</div>', unsafe_allow_html=True)
        sector = st.selectbox("SECTOR", ["ALL SECTORS", "ENERGY", "METALS", "AGRICULTURE"])
        universe = list(COMMODITY_UNIVERSE.keys())
        filtered = [c for c in universe if sector == "ALL SECTORS" or COMMODITY_UNIVERSE[c]["sector"] == sector.lower()]
        primary = st.selectbox("PRIMARY CONTRACT", filtered, index=0)
        compare = st.multiselect("COMPARE WITH", [c for c in filtered if c != primary], default=[], max_selections=4)

        st.markdown('<div class="section-header">MODEL SETTINGS</div>', unsafe_allow_html=True)
        scenario = st.selectbox("SCENARIO", list(SCENARIO_PARAMS.keys()))
        months_out = st.slider("MONTHS FORWARD", min_value=6, max_value=36, value=DEFAULT_MONTHS, step=3)
        risk_free_pct = st.slider("RISK-FREE RATE (%)", min_value=0.0, max_value=10.0, value=float(DEFAULT_RF), step=0.25)

        st.markdown('<div class="section-header">AI SETTINGS</div>', unsafe_allow_html=True)
        default_api_key = ""
        if "ANTHROPIC_API_KEY" in st.secrets:
            default_api_key = st.secrets["ANTHROPIC_API_KEY"]
        api_key = st.text_input("ANTHROPIC API KEY", value=default_api_key, type="password", placeholder="sk-ant-...")

        st.markdown('<div class="section-header">MODEL STATUS</div>', unsafe_allow_html=True)
        st.info("This build uses a synthetic term-structure engine for education and prototyping. Labelled simulated sections are not live exchange history.")

    st.markdown(
        f'''<div style="border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:14px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:flex-end;gap:16px;">
        <div><span style="font-family:IBM Plex Mono,monospace;font-size:16px;color:#f0b429;">◈</span>
        <span style="font-family:IBM Plex Mono,monospace;font-size:15px;color:#e2e8f0;margin-left:10px;letter-spacing:0.12em;">MACRO FUTURES CURVE TERMINAL</span>
        <span style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#4a5568;margin-left:16px;letter-spacing:0.2em;">BACKWARDATION · CONTANGO · ROLL YIELD · BASIS · SEASONALS</span></div>
        <div style="text-align:right;font-family:IBM Plex Mono,monospace;font-size:11px;color:#64748b;">{now.strftime('%H:%M %Z')}<br><span style="font-size:9px;">{now.strftime('%a %d %b %Y')}</span></div></div>''',
        unsafe_allow_html=True,
    )

    selected = list(dict.fromkeys([primary] + compare))
    data_map = {commodity: get_curve_data(commodity, scenario, months_out, risk_free_pct) for commodity in selected}
    primary_data = data_map[primary]
    meta = MARKET_META[primary]

    roll_yld = compute_roll_yield(primary_data)
    basis_val = compute_basis(primary_data)
    slope = curve_slope_pct(primary_data)

    structure_color = "#48bb78" if primary_data.structure == "BACKWARDATION" else "#fc8181" if primary_data.structure == "CONTANGO" else "#f0b429"
    roll_color = "#48bb78" if roll_yld > 0 else "#fc8181"
    basis_color = "#f0b429"

    metrics = st.columns(6)
    metric_rows = [
        ("SPOT PRICE", f"{meta['symbol']}{primary_data.spot:.2f}", meta["unit"], "#e2e8f0"),
        ("STRUCTURE", primary_data.structure, "curve regime", structure_color),
        ("ROLL YIELD", f"{roll_yld:+.2f}%", "annualised", roll_color),
        ("BASIS", f"{basis_val:+.2f}%", "spot vs front", basis_color),
        ("CURVE SLOPE", f"{slope:+.1f}%", f"M1–M{months_out}", "#94a3b8"),
        ("MODEL MODE", primary_data.model_mode, "not live exchange data", "#f0b429"),
    ]
    for col, (label, value, sub, color) in zip(metrics, metric_rows):
        col.markdown(metric_html(label, value, sub, color), unsafe_allow_html=True)

    st.caption("Bloomberg-style educational terminal. Simulated sections are clearly labelled. You can later swap the engine for live CME/ICE settlement data.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "TERM STRUCTURE",
        "ROLL YIELD & CARRY",
        "BASIS ANALYSIS",
        "SEASONAL PATTERNS",
        "CROSS-COMMODITY",
    ])

    # --------------------------------------------------------
    # TAB 1
    # --------------------------------------------------------
    with tab1:
        left, right = st.columns([3, 2])
        with left:
            st.markdown('<div class="section-header">FORWARD CURVE</div>', unsafe_allow_html=True)
            labels = [f"M{i+1}" for i in range(months_out)]
            fig = go.Figure()
            fill_color = "rgba(72,187,120,0.05)" if primary_data.structure == "BACKWARDATION" else "rgba(252,129,129,0.05)"
            fig.add_trace(go.Scatter(x=labels, y=primary_data.prices, fill="tozeroy", fillcolor=fill_color, line=dict(color="transparent"), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=labels,
                y=primary_data.prices,
                mode="lines+markers",
                name=primary,
                line=dict(color=meta["color"], width=2.6),
                marker=dict(size=5, color=meta["color"], line=dict(color="#06080f", width=1.4)),
                hovertemplate=f"<b>{primary}</b><br>%{{x}}<br>{meta['symbol']}%{{y:.2f}} {meta['unit']}<extra></extra>",
            ))
            for comp in compare[:4]:
                comp_data = data_map[comp]
                comp_meta = MARKET_META[comp]
                fig.add_trace(go.Scatter(
                    x=labels,
                    y=comp_data.prices,
                    mode="lines",
                    name=comp,
                    line=dict(color=comp_meta["color"], width=1.5, dash="dot"),
                    hovertemplate=f"<b>{comp}</b><br>%{{x}}<br>{comp_meta['symbol']}%{{y:.2f}} {comp_meta['unit']}<extra></extra>",
                ))
            fig.add_hline(y=primary_data.spot, line_dash="dot", line_color="rgba(255,255,255,0.2)", annotation_text=f"SPOT {meta['symbol']}{primary_data.spot:.2f}", annotation_font=dict(color="#64748b", size=9))
            apply_plot_theme(fig, f"{primary} FORWARD CURVE · {scenario.upper()}", 410)
            fig.update_yaxes(title_text=f"Price ({meta['unit']})", title_font=dict(size=9))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-header">ROLL TABLE</div>', unsafe_allow_html=True)
            roll_table_rows = []
            for i in range(min(months_out - 1, 12)):
                near_px = primary_data.prices[i]
                far_px = primary_data.prices[i + 1]
                roll_cost = far_px - near_px
                roll_pct = (roll_cost / near_px) * 100
                roll_table_rows.append({
                    "Roll": f"M{i+1}→M{i+2}",
                    "Near": f"{meta['symbol']}{near_px:.2f}",
                    "Far": f"{meta['symbol']}{far_px:.2f}",
                    "Roll Cost": f"{meta['symbol']}{roll_cost:+.3f}",
                    "Roll %": f"{roll_pct:+.3f}%",
                    "Ann. Carry": f"{roll_pct * 12:+.2f}%",
                    "Signal": "COLLECT" if roll_pct < 0 else "PAY",
                })
            roll_df = pd.DataFrame(roll_table_rows)
            st.dataframe(roll_df, use_container_width=True, hide_index=True)
            st.download_button("DOWNLOAD ROLL TABLE CSV", dataframe_to_csv_bytes(roll_df), file_name=f"{primary.lower().replace(' ', '_')}_roll_table.csv", mime="text/csv")

        with right:
            st.markdown('<div class="section-header">CURVE READ</div>', unsafe_allow_html=True)
            spread_3 = ((primary_data.prices[2] - primary_data.prices[0]) / primary_data.prices[0] * 100) if months_out >= 3 else 0
            spread_12 = ((primary_data.prices[11] - primary_data.prices[0]) / primary_data.prices[0] * 100) if months_out >= 12 else ((primary_data.prices[-1] - primary_data.prices[0]) / primary_data.prices[0] * 100)
            convexity = np.polyfit(range(len(primary_data.prices)), primary_data.prices, 2)[0]
            convexity_text = "convex" if convexity > 0 else "concave"
            st.markdown(
                f'''<div class="ai-box"><div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#b794f4;letter-spacing:0.15em;margin-bottom:10px;">STRUCTURAL READ</div>
                <strong>{primary}</strong> is in <strong style="color:{structure_color}">{primary_data.structure}</strong>.<br><br>
                Front spread M1–M3: <strong>{spread_3:+.2f}%</strong><br>
                Intermediate spread M1–M12: <strong>{spread_12:+.2f}%</strong><br>
                Curve shape: <strong>{convexity_text}</strong><br><br>
                <strong>Interpretation:</strong> {'Prompt barrels are richer than deferred contracts, so the market is paying for immediacy.' if primary_data.structure == 'BACKWARDATION' else 'Deferred contracts sit above prompt, which is consistent with financing and storage-led carry.' if primary_data.structure == 'CONTANGO' else 'Prompt and deferred pricing are relatively balanced, suggesting a neutral physical signal.'}
                <br><br><strong>Important:</strong> this is a synthetic model output, not live exchange settlement data.</div>''',
                unsafe_allow_html=True,
            )
            if st.button("RUN AI CURVE ANALYSIS", key="curve_ai"):
                if not api_key:
                    st.warning("Add your Anthropic API key in the sidebar or in Streamlit secrets.")
                else:
                    with st.spinner("Running AI curve analysis..."):
                        result = ai_curve_structure(api_key, primary_data)
                    render_steps(result)
                    if result and result.get("trade_idea"):
                        info_box("TRADE IDEA", result["trade_idea"], "#f0b429")
                    if result and result.get("risk_to_view"):
                        info_box("RISK TO VIEW", result["risk_to_view"], "#fc8181")

    # --------------------------------------------------------
    # TAB 2
    # --------------------------------------------------------
    with tab2:
        left, right = st.columns([3, 2])
        roll_yields = [((primary_data.prices[i] - primary_data.prices[i + 1]) / primary_data.prices[i + 1]) * 100 * 12 for i in range(len(primary_data.prices) - 1)]
        roll_labels = [f"M{i+1}→M{i+2}" for i in range(len(roll_yields))]
        avg_roll = float(np.mean(roll_yields)) if roll_yields else 0.0
        with left:
            st.markdown('<div class="section-header">ROLL YIELD TERM STRUCTURE</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=roll_labels,
                y=roll_yields,
                marker_color=["#48bb78" if x > 0 else "#fc8181" for x in roll_yields],
                name="Roll Yield",
                hovertemplate="<b>%{x}</b><br>%{y:.2f}% annualised<extra></extra>",
            ))
            fig2.add_hline(y=0, line_color="rgba(255,255,255,0.15)")
            cumulative = np.cumsum(roll_yields) / 12
            fig2.add_trace(go.Scatter(x=roll_labels, y=cumulative, mode="lines", name="Cumulative Carry", line=dict(color="#b794f4", width=2, dash="dot"), yaxis="y2"))
            apply_plot_theme(fig2, "ANNUALISED ROLL YIELD BY CONTRACT PERIOD", 390)
            fig2.update_layout(yaxis=dict(title="Annualised roll (%)", title_font=dict(size=9)), yaxis2=dict(title="Cumulative carry (%)", overlaying="y", side="right", title_font=dict(size=9)))
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="section-header">CARRY DECOMPOSITION</div>', unsafe_allow_html=True)
            implied_front_yield = ((primary_data.prices[0] / primary_data.spot) - 1) * 100 * 12
            carry_df = pd.DataFrame([
                {"Component": "Risk-Free Rate", "Value": f"+{primary_data.risk_free:.1f}%", "Interpretation": "Funding cost"},
                {"Component": "Storage Cost", "Value": f"+{primary_data.storage_cost:.1f}%", "Interpretation": "Storage and carry"},
                {"Component": "Convenience Yield", "Value": f"-{primary_data.convenience_yield:.1f}%", "Interpretation": "Benefit of physical ownership"},
                {"Component": "Net Cost-of-Carry", "Value": f"{primary_data.risk_free + primary_data.storage_cost - primary_data.convenience_yield:+.1f}%", "Interpretation": "Theoretical carry"},
                {"Component": "Implied Front Yield", "Value": f"{implied_front_yield:+.2f}%", "Interpretation": "Front futures implication"},
                {"Component": "Residual", "Value": f"{implied_front_yield - (primary_data.risk_free + primary_data.storage_cost - primary_data.convenience_yield):+.2f}%", "Interpretation": "Risk premium or scarcity effect"},
            ])
            st.dataframe(carry_df, use_container_width=True, hide_index=True)

        with right:
            st.markdown('<div class="section-header">CARRY SIGNALS</div>', unsafe_allow_html=True)
            carry_regime = "STRONG BACKWARDATION" if avg_roll > 5 else "MILD BACKWARDATION" if avg_roll > 0 else "MILD CONTANGO" if avg_roll > -5 else "DEEP CONTANGO"
            pill_class = "pill-green" if avg_roll > 0 else "pill-red"
            max_idx = int(np.argmax(roll_yields)) if roll_yields else 0
            min_idx = int(np.argmin(roll_yields)) if roll_yields else 0
            st.markdown(
                f'''<div class="ai-box"><div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#b794f4;margin-bottom:10px;letter-spacing:0.15em;">CARRY REGIME</div>
                <span class="pill {pill_class}">{carry_regime}</span><br><br>
                Best roll window: <strong>M{max_idx+1}→M{max_idx+2}</strong> at <strong style="color:#48bb78">{roll_yields[max_idx]:+.2f}%</strong><br>
                Most expensive roll: <strong>M{min_idx+1}→M{min_idx+2}</strong> at <strong style="color:#fc8181">{roll_yields[min_idx]:+.2f}%</strong><br><br>
                Net theoretical carry: <strong>{primary_data.risk_free + primary_data.storage_cost - primary_data.convenience_yield:+.2f}%</strong><br>
                Average model roll yield: <strong>{avg_roll:+.2f}%</strong><br><br>
                This section remains model-derived unless you later connect live futures settlements.</div>''',
                unsafe_allow_html=True,
            )
            if st.button("RUN AI CARRY ANALYSIS", key="carry_ai"):
                if not api_key:
                    st.warning("Add your Anthropic API key in the sidebar or in Streamlit secrets.")
                else:
                    with st.spinner("Running AI carry analysis..."):
                        result = ai_carry(api_key, primary_data, roll_yields, avg_roll)
                    render_steps(result)
                    if result and result.get("trade"):
                        info_box("CARRY TRADE", result["trade"], "#48bb78")
                    if result and result.get("risk"):
                        info_box("CARRY RISK", result["risk"], "#fc8181")

    # --------------------------------------------------------
    # TAB 3
    # --------------------------------------------------------
    with tab3:
        left, right = st.columns([3, 2])
        basis_hist = simulate_basis_history(primary_data, 90)
        z_score = (basis_val - float(basis_hist["basis"].mean())) / max(float(basis_hist["basis"].std()), 1e-6)
        with left:
            st.markdown('<div class="section-header">SIMULATED 90-DAY BASIS HISTORY</div>', unsafe_allow_html=True)
            fig3 = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], shared_xaxes=True, vertical_spacing=0.06)
            fig3.add_trace(go.Scatter(x=basis_hist["date"], y=basis_hist["spot"], name="Spot", line=dict(color="#f0b429", width=2)), row=1, col=1)
            fig3.add_trace(go.Scatter(x=basis_hist["date"], y=basis_hist["front"], name="Front", line=dict(color=meta["color"], width=2, dash="dot")), row=1, col=1)
            fig3.add_trace(go.Scatter(x=basis_hist["date"], y=basis_hist["basis"], fill="tozeroy", fillcolor="rgba(183,148,244,0.08)", line=dict(color="#b794f4", width=1.5), name="Basis %"), row=2, col=1)
            fig3.add_hline(y=0, line_color="rgba(255,255,255,0.1)", row=2, col=1)
            fig3.update_layout(
                paper_bgcolor="#0b0f1a",
                plot_bgcolor="#06080f",
                font=dict(family="IBM Plex Mono, monospace", color="#94a3b8", size=10),
                height=430,
                title=dict(text="SIMULATED 90-DAY BASIS HISTORY", font=dict(size=11, color="#4a5568"), x=0.01, y=0.99),
                margin=dict(l=10, r=10, t=36, b=10),
            )
            fig3.update_yaxes(row=2, col=1, title_text="Basis (%)", title_font=dict(size=9), gridcolor="rgba(255,255,255,0.04)")
            fig3.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
            fig3.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
            st.plotly_chart(fig3, use_container_width=True)

            stats_cols = st.columns(5)
            stats = [
                ("CURRENT", f"{basis_val:+.2f}%", "#94a3b8"),
                ("90D MEAN", f"{basis_hist['basis'].mean():+.2f}%", "#94a3b8"),
                ("Z-SCORE", f"{z_score:+.2f}σ", "#48bb78" if abs(z_score) > 1.5 else "#94a3b8"),
                ("MIN", f"{basis_hist['basis'].min():+.2f}%", "#fc8181"),
                ("MAX", f"{basis_hist['basis'].max():+.2f}%", "#48bb78"),
            ]
            for col, (label, value, color) in zip(stats_cols, stats):
                col.markdown(metric_html(label, value, "simulated", color), unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-header">BASIS INTERPRETATION</div>', unsafe_allow_html=True)
            theoretical_basis = primary_data.risk_free + primary_data.storage_cost
            extreme = abs(z_score) > 1.5
            st.markdown(
                f'''<div class="ai-box"><div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#b794f4;margin-bottom:10px;letter-spacing:0.15em;">BASIS SIGNAL</div>
                Current basis is <strong>{basis_val:+.2f}%</strong>. Positive basis means spot is above front futures; negative basis means front futures exceed spot.<br><br>
                Theoretical carry basis reference: <strong>+{theoretical_basis:.2f}%</strong><br>
                Residual to theory: <strong>{basis_val - theoretical_basis:+.2f}%</strong><br>
                Z-score vs simulated 90-day path: <strong>{z_score:+.2f}σ</strong><br><br>
                {'This looks statistically stretched versus the model path.' if extreme else 'This sits within a more normal model range.'}<br><br>
                Important: this history is simulated for demonstration, not downloaded market history.</div>''',
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # TAB 4
    # --------------------------------------------------------
    with tab4:
        left, right = st.columns([3, 2])
        seasonal = get_seasonal_pattern(primary)
        months_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        current_month = now.month - 1
        with left:
            st.markdown('<div class="section-header">SEASONAL COMPOSITE</div>', unsafe_allow_html=True)
            fig4 = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.08)
            seasonal_colors = ["#f0b429" if i == current_month else ("#48bb78" if v > 0 else "#fc8181") for i, v in enumerate(seasonal["avg_roll"])]
            fig4.add_trace(go.Bar(x=months_names, y=seasonal["avg_roll"], marker_color=seasonal_colors, name="Avg Roll"), row=1, col=1)
            upper = [x + y for x, y in zip(seasonal["avg_roll"], seasonal["std_roll"])]
            lower = [x - y for x, y in zip(seasonal["avg_roll"], seasonal["std_roll"])]
            fig4.add_trace(go.Scatter(x=months_names, y=upper, line=dict(color="rgba(183,148,244,0.3)", width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
            fig4.add_trace(go.Scatter(x=months_names, y=lower, fill="tonexty", fillcolor="rgba(183,148,244,0.06)", line=dict(color="rgba(183,148,244,0.3)", width=0.5), name="±1σ", hoverinfo="skip"), row=1, col=1)
            fig4.add_trace(go.Scatter(x=months_names, y=seasonal["backwardation_pct"], mode="lines+markers", line=dict(color="#4fd1c5", width=2), marker=dict(size=5), name="Backwardation Frequency"), row=2, col=1)
            fig4.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)", row=2, col=1)
            fig4.add_vrect(x0=current_month - 0.4, x1=current_month + 0.4, fillcolor="rgba(240,180,41,0.05)", line_color="rgba(240,180,41,0.3)", line_width=1)
            fig4.update_layout(
                paper_bgcolor="#0b0f1a",
                plot_bgcolor="#06080f",
                font=dict(family="IBM Plex Mono, monospace", color="#94a3b8", size=10),
                height=435,
                title=dict(text=f"SEASONAL COMPOSITE — {primary.upper()}", font=dict(size=11, color="#4a5568"), x=0.01, y=0.99),
                margin=dict(l=10, r=10, t=36, b=10),
            )
            fig4.update_yaxes(row=2, col=1, title_text="Backwardation Freq. (%)", title_font=dict(size=9), gridcolor="rgba(255,255,255,0.04)")
            fig4.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
            fig4.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
            st.plotly_chart(fig4, use_container_width=True)

        with right:
            st.markdown('<div class="section-header">SEASONAL EDGE</div>', unsafe_allow_html=True)
            cur_roll = seasonal["avg_roll"][current_month]
            cur_back = seasonal["backwardation_pct"][current_month]
            cur_std = seasonal["std_roll"][current_month]
            cur_label = months_names[current_month]
            pill_class = "pill-green" if cur_roll > 0 else "pill-red"
            st.markdown(
                f'''<div class="ai-box"><div style="font-family:IBM Plex Mono,monospace;font-size:9px;color:#b794f4;margin-bottom:10px;letter-spacing:0.15em;">CURRENT MONTH — {cur_label.upper()}</div>
                <span class="pill {pill_class}">{'SEASONAL TAILWIND' if cur_roll > 1 else 'SEASONAL HEADWIND' if cur_roll < -1 else 'SEASONALLY NEUTRAL'}</span><br><br>
                Historical average roll: <strong>{cur_roll:+.2f}%</strong><br>
                Backwardation frequency: <strong>{cur_back:.1f}%</strong><br>
                1σ range: <strong>[{cur_roll-cur_std:+.1f}%, {cur_roll+cur_std:+.1f}%]</strong><br><br>
                Current model roll: <strong>{roll_yld:+.2f}%</strong><br>
                Deviation vs seasonal norm: <strong>{roll_yld - cur_roll:+.2f}%</strong><br><br>
                Driver: {seasonal['driver_text']}<br><br>
                Calendar idea: {seasonal['calendar_trade']}<br><br>
                This seasonal panel is also modelled, not a downloaded historical backtest.</div>''',
                unsafe_allow_html=True,
            )
            if st.button("RUN AI SEASONAL ANALYSIS", key="seasonal_ai"):
                if not api_key:
                    st.warning("Add your Anthropic API key in the sidebar or in Streamlit secrets.")
                else:
                    with st.spinner("Running AI seasonal analysis..."):
                        result = ai_seasonal(api_key, primary, seasonal, current_month, roll_yld)
                    render_steps(result)
                    if result and result.get("trade"):
                        info_box("SEASONAL TRADE", result["trade"], "#4fd1c5")
                    if result and result.get("exit_signal"):
                        info_box("EXIT SIGNAL", result["exit_signal"], "#f0b429")
                    if result and result.get("confirmation_signals"):
                        st.markdown('<div class="section-header">CONFIRMATION SIGNALS</div>', unsafe_allow_html=True)
                        for signal in result["confirmation_signals"]:
                            st.markdown(f'<div style="padding:6px 10px;background:rgba(255,255,255,0.02);border-left:2px solid rgba(79,209,197,0.4);border-radius:3px;margin-bottom:5px;font-size:12px;color:#94a3b8;">• {signal}</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # TAB 5
    # --------------------------------------------------------
    with tab5:
        left, right = st.columns([3, 2])
        compare_pool = list(dict.fromkeys([primary] + compare))
        if len(compare_pool) == 1:
            compare_pool = list(dict.fromkeys([primary] + filtered[:4]))
        compare_pool = compare_pool[:5]
        pool_data = {commodity: get_curve_data(commodity, scenario, months_out, risk_free_pct) for commodity in compare_pool}

        with left:
            st.markdown('<div class="section-header">NORMALISED CURVE COMPARISON</div>', unsafe_allow_html=True)
            fig5 = go.Figure()
            for commodity, data in pool_data.items():
                commodity_meta = MARKET_META[commodity]
                norm = [(px / data.spot - 1) * 100 for px in data.prices]
                fig5.add_trace(go.Scatter(
                    x=[f"M{i+1}" for i in range(months_out)],
                    y=norm,
                    mode="lines+markers",
                    name=f"{commodity} ({data.structure})",
                    line=dict(color=commodity_meta["color"], width=2.5 if commodity == primary else 1.5),
                    marker=dict(size=5 if commodity == primary else 3, color=commodity_meta["color"]),
                    hovertemplate=f"<b>{commodity}</b><br>%{{x}}<br>%{{y:+.2f}}% vs spot<extra></extra>",
                ))
            fig5.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")
            apply_plot_theme(fig5, "NORMALISED FORWARD CURVES (% VS SPOT)", 390)
            fig5.update_yaxes(title_text="% vs spot", title_font=dict(size=9))
            st.plotly_chart(fig5, use_container_width=True)

            league_rows: List[Dict[str, Any]] = []
            for commodity, data in pool_data.items():
                ry = compute_roll_yield(data)
                bv = compute_basis(data)
                league_rows.append({
                    "Commodity": commodity,
                    "Sector": COMMODITY_UNIVERSE[commodity]["sector"].title(),
                    "Structure": data.structure,
                    "Spot": f"{MARKET_META[commodity]['symbol']}{data.spot:.2f}",
                    "Ann. Roll": f"{ry:+.2f}%",
                    "Ann. Roll Num": ry,
                    "Basis": f"{bv:+.2f}%",
                    "Signal": "COLLECT" if ry > 0 else "PAY",
                })
            league_df = pd.DataFrame(league_rows).sort_values("Ann. Roll Num", ascending=False).drop(columns=["Ann. Roll Num"])
            st.markdown('<div class="section-header">ROLL YIELD LEAGUE TABLE</div>', unsafe_allow_html=True)
            st.dataframe(league_df, use_container_width=True, hide_index=True)

        with right:
            st.markdown('<div class="section-header">ROLL YIELD COMPARISON</div>', unsafe_allow_html=True)
            ry_values = [compute_roll_yield(pool_data[c]) for c in compare_pool]
            fig6 = go.Figure()
            fig6.add_trace(go.Bar(y=compare_pool, x=ry_values, orientation="h", marker_color=["#48bb78" if x > 0 else "#fc8181" for x in ry_values], hovertemplate="<b>%{y}</b><br>%{x:+.2f}%<extra></extra>"))
            fig6.add_vline(x=0, line_color="rgba(255,255,255,0.2)")
            apply_plot_theme(fig6, "ROLL YIELD COMPARISON", 300)
            st.plotly_chart(fig6, use_container_width=True)

            st.markdown('<div class="section-header">SECTOR SNAPSHOT</div>', unsafe_allow_html=True)
            sector_buckets: Dict[str, List[float]] = {}
            for commodity, data in pool_data.items():
                sector_name = COMMODITY_UNIVERSE[commodity]["sector"]
                sector_buckets.setdefault(sector_name, []).append(compute_roll_yield(data))
            for sector_name, values in sector_buckets.items():
                avg_val = float(np.mean(values))
                color = "#48bb78" if avg_val > 0 else "#fc8181"
                st.markdown(f'<div style="padding:9px 12px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:4px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;"><span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#94a3b8;">{sector_name.upper()}</span><span style="font-family:IBM Plex Mono,monospace;font-size:13px;font-weight:700;color:{color};">{avg_val:+.2f}%</span></div>', unsafe_allow_html=True)

            if st.button("RUN AI CROSS-COMMODITY ANALYSIS", key="cross_ai"):
                if not api_key:
                    st.warning("Add your Anthropic API key in the sidebar or in Streamlit secrets.")
                else:
                    with st.spinner("Running AI cross-commodity analysis..."):
                        result = ai_cross(api_key, build_snapshot(pool_data))
                    render_steps(result)
                    if result and result.get("spread_trade"):
                        info_box("SPREAD TRADE", result["spread_trade"], "#4299e1")
                    if result and result.get("macro_read"):
                        info_box("MACRO READ", result["macro_read"], "#f0b429")
                    if result and result.get("best_long"):
                        info_box("BEST LONG", result["best_long"], "#48bb78")
                    if result and result.get("best_short"):
                        info_box("BEST SHORT", result["best_short"], "#fc8181")

    st.markdown("---")
    st.caption("Deployment note: for Streamlit Community Cloud, keep this file in your repo with a requirements.txt alongside it. Put your API key in secrets, not in the code.")


if __name__ == "__main__":
    main()
