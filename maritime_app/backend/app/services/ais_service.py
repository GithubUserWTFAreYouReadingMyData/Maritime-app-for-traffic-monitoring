from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class AISService:
    cache_dir: Path = Path(settings.raw_data_dir)

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_uniparthenope_url(self, filename: str) -> str:
        filename = filename.strip().lstrip("/")
        return f"https://data.meteo.uniparthenope.it/instruments/aisnet0/csv/{filename}"

    def load_csv_from_url(self, url: str, filename: Optional[str] = None) -> pd.DataFrame:
        if filename:
            cache_path = self.cache_dir / filename
            if cache_path.exists():
                logger.info("Usando cache: %s", cache_path)
                return pd.read_csv(cache_path)

        headers = {"User-Agent": "Mozilla/5.0"}
        last_error: Optional[Exception] = None

        # 1) Tentativi diretti
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, timeout=60)
                r.raise_for_status()
                text = r.content.decode("utf-8", errors="ignore")
                last_error = None
                break
            except Exception as e:
                last_error = e
                logger.warning("Download diretto fallito (tentativo %s/3): %s", attempt + 1, e)
        else:
            text = ""

        # 2) Fallback proxy (solo trasporto; i dati restano reali)
        if last_error is not None:
            proxy_url = settings.cors_proxy + url
            logger.warning("Provo proxy per ottenere dati REALI: %s", proxy_url)
            try:
                r = requests.get(proxy_url, headers=headers, timeout=60)
                r.raise_for_status()
                text = r.content.decode("utf-8", errors="ignore")
                last_error = None
            except Exception as e:
                last_error = e

        if last_error is not None:
            raise RuntimeError(f"Impossibile scaricare dati REALI da uniparthenope: {last_error}")

        df = pd.read_csv(io.StringIO(text))

        if df is None or len(df) == 0:
            raise RuntimeError("Dataset reale scaricato ma vuoto")

        if filename:
            cache_path = self.cache_dir / filename
            df.to_csv(cache_path, index=False)
        return df

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def find_col(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        ts_col = find_col(["timestamp", "BaseDateTime", "TIMESTAMP", "Timestamp", "datetime", "time", "DATE_TIME"])
        mmsi_col = find_col(["MMSI", "mmsi", "VesselID", "SHIP_ID", "id", "vessel_id", "ID"])
        lon_col = find_col(["Lon", "LON", "Longitude", "lon", "longitude", "LONGITUDE", "Long"])
        lat_col = find_col(["Lat", "LAT", "Latitude", "lat", "latitude", "LATITUDE"])
        head_col = find_col([
            "Heading",
            "HEADING",
            "TrueHeading",
            "TRUEHEADING",
            "HDG",
            "hdg",
            "head",
            "HEAD",
            "COG",
            "Course",
            "course",
        ])
        spd_col = find_col([
            "Speed",
            "SPEED",
            "SOG",
            "sog",
            "SpeedOverGround",
            "velocity",
            "Velocity",
        ])

        course_col = find_col([
            "Course",
            "COURSE",
            "course",
            "COG",
            "cog",
            "CourseOverGround",
        ])

        ren = {}
        if ts_col and ts_col != "timestamp":
            ren[ts_col] = "timestamp"
        if mmsi_col and mmsi_col != "MMSI":
            ren[mmsi_col] = "MMSI"
        if lon_col and lon_col != "Lon":
            ren[lon_col] = "Lon"
        if lat_col and lat_col != "Lat":
            ren[lat_col] = "Lat"
        if head_col and head_col != "Heading":
            ren[head_col] = "Heading"
        if spd_col and spd_col != "Speed":
            ren[spd_col] = "Speed"
        if course_col and course_col != "Course":
            ren[course_col] = "Course"
        if ren:
            df = df.rename(columns=ren)

        if "Heading" not in df.columns:
            df["Heading"] = 0.0
        if "Speed" not in df.columns:
            df["Speed"] = 0.0
        if "Course" not in df.columns:
            df["Course"] = np.nan

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        else:
            df["timestamp"] = pd.Timestamp.utcnow()

        if "MMSI" in df.columns:
            df["MMSI"] = df["MMSI"].astype(str)
        else:
            df["MMSI"] = [f"UNK_{i}" for i in range(len(df))]

        df["Lon"] = pd.to_numeric(df.get("Lon"), errors="coerce")
        df["Lat"] = pd.to_numeric(df.get("Lat"), errors="coerce")
        df["Speed"] = pd.to_numeric(df.get("Speed"), errors="coerce").fillna(0.0)
        df["Heading"] = pd.to_numeric(df.get("Heading"), errors="coerce").fillna(0.0)
        df["Course"] = pd.to_numeric(df.get("Course"), errors="coerce")

        df = df.dropna(subset=["Lon", "Lat", "timestamp"])

        b = settings.test_area_bounds
        df = df[(df["Lon"].between(b["lon_min"], b["lon_max"])) & (df["Lat"].between(b["lat_min"], b["lat_max"]))]

        df = df.sort_values("timestamp")
        return df

    def to_geojson_points(self, df: pd.DataFrame, limit: int = 20000) -> dict:
        if limit and len(df) > limit:
            df = df.sample(limit, random_state=42)

        features = []
        for _, r in df.iterrows():
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(r["Lon"]), float(r["Lat"])]},
                    "properties": {
                        "mmsi": r["MMSI"],
                        "timestamp": pd.Timestamp(r["timestamp"]).isoformat(),
                        "speed": float(r.get("Speed", 0.0)),
                        "heading": float(r.get("Heading", 0.0)),
                        "course": None if pd.isna(r.get("Course")) else float(r.get("Course")),
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def to_geojson_trajectories(self, df: pd.DataFrame, vessel_limit: int = 200, point_limit_per_vessel: int = 200) -> dict:
        vessels = df.groupby("MMSI")
        mmsis = list(vessels.groups.keys())
        if len(mmsis) > vessel_limit:
            mmsis = list(np.random.RandomState(42).choice(mmsis, vessel_limit, replace=False))

        features = []
        for m in mmsis:
            vdf = vessels.get_group(m).sort_values("timestamp")
            if len(vdf) > point_limit_per_vessel:
                vdf = vdf.iloc[-point_limit_per_vessel:]
            coords = [[float(x), float(y)] for x, y in zip(vdf["Lon"], vdf["Lat"])]
            if len(coords) < 2:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"mmsi": m},
                }
            )
        return {"type": "FeatureCollection", "features": features}
