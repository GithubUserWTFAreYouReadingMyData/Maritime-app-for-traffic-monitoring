from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from app.core.config import settings
from app.mstffn.model import MSTFFN

logger = logging.getLogger(__name__)


def _time_features_from_timestamp(ts: pd.Timestamp) -> np.ndarray:
    ts = pd.Timestamp(ts)
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    hour = ts.hour / 23.0
    dow = ts.dayofweek / 6.0
    month = (ts.month - 1) / 11.0
    return np.array([hour, dow, month], dtype=np.float32)


@dataclass
class MSTFFNService:
    device: str = "cpu"

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA richiesto ma non disponibile, uso CPU")
            self.device = "cpu"
        self.model = MSTFFN(d_model=128, n_heads=8, n_layers=4).to(self.device)
        self._try_load_weights()
        self.model.eval()

    def _try_load_weights(self) -> None:
        path = Path(settings.model_weights_path)
        if not path.exists():
            logger.info("Pesi MSTFFN non trovati (%s). Uso pesi iniziali.", path)
            return

        try:
            obj = torch.load(path, map_location=self.device)
            state_dict = obj.get("state_dict") if isinstance(obj, dict) else obj
            self.model.load_state_dict(state_dict)
            logger.info("Caricati pesi MSTFFN da %s", path)
        except Exception as e:
            logger.warning("Impossibile caricare pesi MSTFFN da %s: %s", path, e)

    def load_weights(self) -> str:
        self._try_load_weights()
        return str(Path(settings.model_weights_path))

    def save_weights(self) -> str:
        path = Path(settings.model_weights_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict()}, path)
        return str(path)

    def _build_multiscale(self, flow_values: np.ndarray) -> Dict[str, torch.Tensor]:
        # flow_values: 168 values (last 168 steps)
        low = flow_values[-24:].astype(np.float32)
        med = flow_values[-72:].astype(np.float32)
        high = flow_values[-168:].astype(np.float32)
        return {
            "low": torch.from_numpy(low).unsqueeze(0).to(self.device),
            "medium": torch.from_numpy(med).unsqueeze(0).to(self.device),
            "high": torch.from_numpy(high).unsqueeze(0).to(self.device),
        }

    def predict_rollout(
        self,
        flow_series: pd.DataFrame,
        t_ref: pd.Timestamp,
        horizons: List[int],
        freq: str = "1h",
    ) -> Dict[str, List[float]]:
        # Backward compatible: interpreta horizons come "numero di step" sulla freq passata.
        return self.predict_rollout_steps(flow_series=flow_series, t_ref=t_ref, horizon_steps=horizons, freq=freq)

    def predict_rollout_steps(
        self,
        flow_series: pd.DataFrame,
        t_ref: pd.Timestamp,
        horizon_steps: List[int],
        freq: str,
    ) -> Dict[str, List[float]]:
        if flow_series is None or len(flow_series) == 0:
            raise ValueError("flow_series non disponibile")

        fs = flow_series.copy()
        fs["timestamp"] = pd.to_datetime(fs["timestamp"], utc=True)
        fs = fs.sort_values("timestamp")
        fs = fs.set_index("timestamp").asfreq(freq)
        fs["value"] = fs["value"].interpolate(limit_direction="both")

        t_ref = pd.Timestamp(t_ref)
        t_ref = t_ref.tz_convert("UTC") if t_ref.tzinfo else t_ref.tz_localize("UTC")
        if t_ref not in fs.index:
            nearest = fs.index[np.argmin(np.abs((fs.index - t_ref).total_seconds()))]
            t_ref = nearest

        # history: 168 step (non ore)
        step_delta = pd.to_timedelta(freq)
        start = t_ref - step_delta * 168
        if start < fs.index.min():
            raise ValueError("Non ci sono abbastanza step storici per 168 step prima di t_ref")

        window = fs.loc[start:t_ref]["value"].to_numpy()
        if len(window) < 169:
            raise ValueError("Finestra storica insufficiente")

        hist = window[-168:]

        mu_list: List[float] = []
        sigma_list: List[float] = []

        max_step = max(horizon_steps)
        current_hist = hist.copy()

        for step in range(1, max_step + 1):
            multiscale = self._build_multiscale(current_hist)
            tf = _time_features_from_timestamp(t_ref + step_delta * step)
            tf_t = torch.from_numpy(tf).unsqueeze(0).to(self.device)

            with torch.no_grad():
                mu, sigma = self.model(multiscale, tf_t)

            mu_v = float(mu.cpu().numpy()[0])
            sigma_v = float(sigma.cpu().numpy()[0])

            mu_v = max(0.0, mu_v)

            if step in horizon_steps:
                mu_list.append(mu_v)
                sigma_list.append(sigma_v)

            current_hist = np.roll(current_hist, -1)
            current_hist[-1] = mu_v

        return {"mu": mu_list, "sigma": sigma_list, "t_ref": [t_ref.isoformat()] * len(horizon_steps), "horizons": horizon_steps}
