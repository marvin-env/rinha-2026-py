from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

RESOURCES = Path(__file__).resolve().parent.parent / "resources"

with (RESOURCES / "normalization.json").open() as f:
    NORM = json.load(f)

with (RESOURCES / "mcc_risk.json").open() as f:
    MCC_RISK: dict[str, float] = json.load(f)

MAX_AMOUNT = float(NORM["max_amount"])
MAX_INSTALLMENTS = float(NORM["max_installments"])
AMOUNT_VS_AVG_RATIO = float(NORM["amount_vs_avg_ratio"])
MAX_MINUTES = float(NORM["max_minutes"])
MAX_KM = float(NORM["max_km"])
MAX_TX_COUNT_24H = float(NORM["max_tx_count_24h"])
MAX_MERCHANT_AVG_AMOUNT = float(NORM["max_merchant_avg_amount"])


def _clamp(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def vectorize(payload: dict) -> np.ndarray:
    tx = payload["transaction"]
    customer = payload["customer"]
    merchant = payload["merchant"]
    terminal = payload["terminal"]
    last = payload.get("last_transaction")

    amount = float(tx["amount"])
    installments = float(tx["installments"])
    requested_at = _parse_iso(tx["requested_at"])

    avg_amount = float(customer["avg_amount"]) or 1e-9
    tx_count_24h = float(customer["tx_count_24h"])
    known = set(customer.get("known_merchants") or [])

    merchant_id = merchant["id"]
    mcc = str(merchant["mcc"])
    merchant_avg = float(merchant["avg_amount"])

    is_online = 1.0 if terminal["is_online"] else 0.0
    card_present = 1.0 if terminal["card_present"] else 0.0
    km_from_home = float(terminal["km_from_home"])

    if last is None:
        minutes_since_last = -1.0
        km_from_last = -1.0
    else:
        prev_ts = _parse_iso(last["timestamp"])
        delta_min = (requested_at - prev_ts).total_seconds() / 60.0
        minutes_since_last = _clamp(delta_min / MAX_MINUTES)
        km_from_last = _clamp(float(last["km_from_current"]) / MAX_KM)

    unknown_merchant = 0.0 if merchant_id in known else 1.0
    mcc_risk = float(MCC_RISK.get(mcc, 0.5))

    weekday = requested_at.weekday()  # mon=0..sun=6 — matches spec

    vec = np.array(
        [
            _clamp(amount / MAX_AMOUNT),
            _clamp(installments / MAX_INSTALLMENTS),
            _clamp((amount / avg_amount) / AMOUNT_VS_AVG_RATIO),
            requested_at.hour / 23.0,
            weekday / 6.0,
            minutes_since_last,
            km_from_last,
            _clamp(km_from_home / MAX_KM),
            _clamp(tx_count_24h / MAX_TX_COUNT_24H),
            is_online,
            card_present,
            unknown_merchant,
            mcc_risk,
            _clamp(merchant_avg / MAX_MERCHANT_AVG_AMOUNT),
        ],
        dtype=np.float32,
    )
    return vec
