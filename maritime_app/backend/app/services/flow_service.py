from __future__ import annotations

import pandas as pd


def choose_adaptive_freq(df: pd.DataFrame, target_bins: int = 240) -> str:
    if df is None or len(df) == 0:
        return "1h"

    ts = pd.to_datetime(df["timestamp"], utc=True)
    tmin = ts.min()
    tmax = ts.max()
    span_s = max(1.0, (tmax - tmin).total_seconds())

    # target: avere abbastanza bin per costruire 168 step di history + margine
    # Con dataset molto corti (pochi minuti) serve scendere fino a 1-2 secondi.
    step_s = max(1.0, span_s / float(target_bins))

    # clamp a range ragionevole
    step_s = min(step_s, 300.0)  # max 5 min

    # quantizza a valori "standard"
    candidates = [1, 2, 5, 10, 15, 30, 60, 120, 300]
    best = min(candidates, key=lambda c: abs(c - step_s))
    return f"{int(best)}s"


def compute_flow_series(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["timestamp", "value"])

    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.set_index("timestamp")

    # Flow proxy: number of AIS points per time bin (can be replaced by gate-crossing)
    s = d["MMSI"].resample(freq).count().rename("value")
    out = s.reset_index().rename(columns={"timestamp": "timestamp"})
    return out
