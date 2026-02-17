from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class AppState:
    ais_df: Optional[pd.DataFrame] = None
    flow_series: Optional[pd.DataFrame] = None  # columns: timestamp,value
    flow_freq: Optional[str] = None
    density_real: Optional[dict] = None
    density_pred: dict = field(default_factory=dict)  # key: (t_ref,h)


state = AppState()
