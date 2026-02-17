from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.services.ais_service import AISService
from app.services.flow_service import compute_flow_series, choose_adaptive_freq
from app.services.mstffn_service import MSTFFNService
from app.services.heatmap_service import HeatmapService
from app.services.temporal_service import TemporalService
from app.services.traffic_density_service import TrafficDensityService
from app.services.state import state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_ais = AISService()
_mstffn = MSTFFNService(device="cpu")
_heatmap = HeatmapService()
_temporal = TemporalService()
_traffic_density = TrafficDensityService()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/model/status")
def model_status():
    return {
        "weights_path": settings.model_weights_path,
    }


@router.post("/model/save")
def model_save():
    try:
        path = _mstffn.save_weights()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"saved_to": path}


@router.post("/model/load")
def model_load():
    try:
        path = _mstffn.load_weights()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"loaded_from": path}


@router.post("/ais/load")
def load_ais(
    filename: Optional[str] = None,
    url: Optional[str] = None,
):
    """Carica dati AIS REALI da meteo.uniparthenope.

    - Se passi `filename`, costruisce l'URL ufficiale uniparthenope.
    - Se passi `url`, lo usa direttamente (deve puntare comunque ai CSV reali).
    - Se non passi nulla, usa `settings.validation_source`.
    """

    if filename:
        resolved_url = _ais.build_uniparthenope_url(filename)
        cache_name = filename
    else:
        resolved_url = url or settings.validation_source
        cache_name = "validation_ais.csv"

    df_raw = _ais.load_csv_from_url(resolved_url, filename=cache_name)
    df = _ais.normalize_columns(df_raw)

    if df is None or len(df) == 0:
        raise HTTPException(
            status_code=502,
            detail=(
                "Dataset reale scaricato ma nessun record valido nel Golfo di Napoli. "
                "Verifica che il file sia quello corretto o che contenga coordinate Lon/Lat compatibili."
            ),
        )

    state.ais_df = df
    flow_freq = choose_adaptive_freq(df, target_bins=300)
    state.flow_freq = flow_freq
    state.flow_series = compute_flow_series(df, freq=flow_freq)

    return {
        "source_url": resolved_url,
        "flow_freq": flow_freq,
        "rows": int(len(df)),
        "mmsi_unique": int(df["MMSI"].nunique()),
        "time_min": df["timestamp"].min().isoformat() if len(df) else None,
        "time_max": df["timestamp"].max().isoformat() if len(df) else None,
    }


@router.get("/ais/points")
def ais_points(
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = Query(20000, ge=1000, le=80000),
    latest: bool = Query(True),
    minSpeed: float = Query(0.0, ge=0.0),
    maxSpeed: float = Query(60.0, ge=0.0),
):
    if state.ais_df is None:
        raise HTTPException(status_code=400, detail="Carica prima i dati con /api/ais/load")

    df = state.ais_df
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end, utc=True)]

    df = df[(df["Speed"] >= minSpeed) & (df["Speed"] <= maxSpeed)]

    if latest:
        # Restituisci l'ultima osservazione per nave per coerenza con /predict (che usa tail(1) per MMSI)
        df = df.sort_values(["MMSI", "timestamp"]).groupby("MMSI").tail(1)

    return _ais.to_geojson_points(df, limit=limit)


@router.get("/ais/speed_stats")
def ais_speed_stats():
    if state.ais_df is None or len(state.ais_df) == 0:
        raise HTTPException(status_code=400, detail="AIS non disponibile. Carica AIS.")

    s = pd.to_numeric(state.ais_df.get("Speed"), errors="coerce").fillna(0.0)
    return {
        "n": int(len(s)),
        "min": float(s.min()),
        "max": float(s.max()),
        "gt5": int((s > 5.0).sum()),
        "gt15": int((s > 15.0).sum()),
    }


@router.get("/ais/trajectories")
def ais_trajectories(
    start: Optional[str] = None,
    end: Optional[str] = None,
    vessel_limit: int = Query(200, ge=10, le=2000),
):
    if state.ais_df is None:
        raise HTTPException(status_code=400, detail="Carica prima i dati con /api/ais/load")

    df = state.ais_df
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end, utc=True)]

    return _ais.to_geojson_trajectories(df, vessel_limit=vessel_limit)


@router.get("/flow/series")
def flow_series():
    if state.flow_series is None:
        raise HTTPException(status_code=400, detail="Flow non disponibile. Carica AIS.")
    return {
        "series": [
            {"timestamp": pd.Timestamp(r["timestamp"]).isoformat(), "value": float(r["value"])}
            for _, r in state.flow_series.iterrows()
        ]
    }


@router.post("/predict")
def predict(
    t_ref: Optional[str] = None,
    horizons: Optional[List[int]] = Query(None),
    horizons_min: Optional[List[int]] = Query([5, 15, 30, 60]),
    mmsi: Optional[List[str]] = Query(None),
):
    if state.flow_series is None or len(state.flow_series) == 0:
        raise HTTPException(status_code=400, detail="Flow non disponibile. Carica AIS.")

    if state.ais_df is None or len(state.ais_df) == 0:
        raise HTTPException(status_code=400, detail="AIS non disponibile. Carica AIS.")

    fs = state.flow_series
    freq = state.flow_freq or "1h"
    if t_ref is None:
        # pick last timestamp available
        t_ref_ts = pd.to_datetime(fs["timestamp"].max(), utc=True)
    else:
        t_ref_ts = pd.to_datetime(t_ref, utc=True)

    # Converti orizzonti in ore -> step sulla freq adattiva
    step_delta = pd.to_timedelta(freq)
    step_seconds = max(1.0, step_delta.total_seconds())
    if horizons_min is not None:
        try:
            horizons_minutes = [int(x) for x in horizons_min]
        except TypeError:
            horizons_minutes = [5, 15, 30, 60]
        horizons_hours = [float(m) / 60.0 for m in horizons_minutes]
    else:
        horizons_hours = [float(x) for x in (horizons or [1, 2, 3])]
        horizons_minutes = [int(round(h * 60.0)) for h in horizons_hours]

    horizon_steps = [max(1, int(round((h * 3600.0) / step_seconds))) for h in horizons_hours]

    try:
        pred = _mstffn.predict_rollout_steps(fs, t_ref_ts, horizon_steps=horizon_steps, freq=freq)
        
        # Salva predizione nello stato per uso futuro
        state.last_prediction = {
            't_ref': t_ref_ts.isoformat(),
            'horizons': horizons_hours,
            'horizons_min': horizons_minutes,
            'flow_freq': freq,
            'horizon_steps': horizon_steps,
            'mu': pred["mu"],
            'sigma': pred["sigma"],
            'layers': {}  # Sarà popolato dopo
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ---- Predizione nave-per-nave (kinematica) usando speed/heading reali a t_ref ----
    df = state.ais_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # usa l'ultima osservazione <= t_ref per ogni MMSI
    df_ref = df[df["timestamp"] <= t_ref_ts].sort_values(["MMSI", "timestamp"]).groupby("MMSI").tail(1)
    if len(df_ref) == 0:
        # fallback: usa l'ultimo timestamp disponibile
        df_ref = df.sort_values(["MMSI", "timestamp"]).groupby("MMSI").tail(1)

    import numpy as np

    # bearing stimato dal track (ultimi 2 punti) come fallback robusto quando HEADING/COG mancano
    df_last2 = df[df["timestamp"] <= t_ref_ts].sort_values(["MMSI", "timestamp"]).groupby("MMSI").tail(2)
    if len(df_last2) > 0:
        def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
            lat1 = np.deg2rad(float(lat1))
            lat2 = np.deg2rad(float(lat2))
            dlon = np.deg2rad(float(lon2) - float(lon1))
            y = np.sin(dlon) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            brg = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0
            return float(brg)

        rows = []
        for m, g in df_last2.groupby("MMSI"):
            g = g.sort_values("timestamp")
            if len(g) < 2:
                continue
            a = g.iloc[-2]
            b = g.iloc[-1]
            rows.append({"MMSI": m, "track_bearing": _bearing_deg(a["Lat"], a["Lon"], b["Lat"], b["Lon"])})
        if rows:
            tb = pd.DataFrame(rows)
            df_ref = df_ref.merge(tb, on="MMSI", how="left")
    if "track_bearing" not in df_ref.columns:
        df_ref["track_bearing"] = np.nan

    if mmsi:
        try:
            mmsi_list = [str(x) for x in mmsi]
            df_ref = df_ref[df_ref["MMSI"].isin(mmsi_list)]
            if len(df_ref) == 0:
                raise HTTPException(status_code=400, detail="Nessuna nave trovata per i MMSI richiesti")
        except TypeError:
            pass  # mmsi non è iterabile, ignora filtro

    def _pick_bearing(heading_deg, course_deg, track_bearing) -> float:
        try:
            h = float(heading_deg)
        except Exception:
            h = float("nan")
        try:
            c = float(course_deg)
        except Exception:
            c = float("nan")

        try:
            t = float(track_bearing)
        except Exception:
            t = float("nan")

        h_valid = (not np.isnan(h)) and (0.0 < h < 360.0) and (h != 511.0)
        c_valid = (not np.isnan(c)) and (0.0 <= c < 360.0) and (c != 360.0)
        t_valid = (not np.isnan(t)) and (0.0 <= t < 360.0)
        if c_valid and not h_valid:
            return c
        if h_valid:
            return h
        if c_valid:
            return c
        if t_valid:
            return t
        return 0.0

    def propagate(lat, lon, speed_kn, heading_deg, course_deg, hours):
        # approssimazione locale: 1 kn = 0.514444 m/s
        v = float(speed_kn) * 0.514444
        dist_m = v * 3600.0 * float(hours)
        # track_bearing viene gestito a livello chiamante
        bearing = _pick_bearing(heading_deg, course_deg, course_deg)
        brg = np.deg2rad(float(bearing) % 360.0)
        d_north = dist_m * np.cos(brg)
        d_east = dist_m * np.sin(brg)
        dlat = d_north / 111320.0
        clat = np.cos(np.deg2rad(float(lat)))
        dlon = d_east / (111320.0 * max(0.2, clat))
        return float(lat + dlat), float(lon + dlon)

    out = {
        "t_ref": t_ref_ts.isoformat(),
        "horizons": [int(round(h)) for h in horizons_hours],
        "horizons_min": horizons_minutes,
        "flow_freq": freq,
        "horizon_steps": horizon_steps,
        "mu": pred["mu"],
        "sigma": pred["sigma"],
        "layers": {},
    }

    def _meters_to_deg(lat: float, d_north_m: float, d_east_m: float):
        dlat = d_north_m / 111320.0
        clat = np.cos(np.deg2rad(float(lat)))
        dlon = d_east_m / (111320.0 * max(0.2, clat))
        return float(dlat), float(dlon)

    for h_min, h_hr, mu, sigma in zip(horizons_minutes, horizons_hours, pred["mu"], pred["sigma"]):
        features = []
        t_pred = (t_ref_ts + pd.Timedelta(minutes=int(h_min))).isoformat()

        # Campionamento nuvola come 'corridoio' di traiettoria.
        # NOTE: sigma è su flow (non spazio). Qui lo mappiamo su metri per visualizzare una dispersione.
        base_std_m = 120.0
        scale_std_m = 120.0
        std_m = float(base_std_m + scale_std_m * max(0.0, float(sigma)))
        samples_per_vessel = 30
        steps_per_traj = 10

        for _, r in df_ref.iterrows():
            lat0 = float(r["Lat"])
            lon0 = float(r["Lon"])
            bearing = _pick_bearing(r.get("Heading", 0.0), r.get("Course", np.nan), r.get("track_bearing", np.nan))
            # usa il bearing scelto direttamente (evita doppio picking dentro propagate)
            lat1, lon1 = propagate(lat0, lon0, r.get("Speed", 0.0), bearing, bearing, h_hr)
            mmsi = int(r["MMSI"])

            # punto centrale futuro (massima probabilità a t_pred)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lon1), float(lat1)]},
                    "properties": {
                        "mmsi": mmsi,
                        "t": t_pred,
                        "speed": float(r.get("Speed", 0.0)),
                        "heading": float(r.get("Heading", 0.0)),
                        "course": None if pd.isna(r.get("Course")) else float(r.get("Course")),
                        "track_bearing": None if pd.isna(r.get("track_bearing")) else float(r.get("track_bearing")),
                        "used_bearing": float(bearing),
                        "mu": float(mu),
                        "sigma": float(sigma),
                        "p": 1.0,
                        "kind": "center",
                    },
                }
            )

            rng = np.random.default_rng(seed=(mmsi + int(h_min) * 1000003) % (2**32 - 1))
            # campiona punti lungo la traiettoria (interpolazione) con rumore
            for k in range(1, steps_per_traj + 1):
                frac = float(k) / float(steps_per_traj)
                lat_c = float(lat0 + (lat1 - lat0) * frac)
                lon_c = float(lon0 + (lon1 - lon0) * frac)
                # rumore cresce leggermente con il tempo (più incertezza in avanti)
                std_k = std_m * (0.6 + 0.8 * frac)

                for _ in range(max(1, samples_per_vessel // steps_per_traj)):
                    dn, de = rng.normal(0.0, std_k, size=2)
                    dlat, dlon = _meters_to_deg(lat_c, float(dn), float(de))
                    lat_s = float(lat_c + dlat)
                    lon_s = float(lon_c + dlon)
                    r2 = (dn * dn + de * de) / max(1.0, std_k * std_k)
                    p = float(np.exp(-0.5 * r2))
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [lon_s, lat_s]},
                            "properties": {
                                "mmsi": mmsi,
                                "t": t_pred,
                                "speed": float(r.get("Speed", 0.0)),
                                "heading": float(r.get("Heading", 0.0)),
                                "course": None if pd.isna(r.get("Course")) else float(r.get("Course")),
                                "track_bearing": None if pd.isna(r.get("track_bearing")) else float(r.get("track_bearing")),
                                "used_bearing": float(bearing),
                                "mu": float(mu),
                                "sigma": float(sigma),
                                "p": p,
                                "kind": "sample",
                            },
                        }
                    )

        out["layers"][str(h_min)] = {"points": {"type": "FeatureCollection", "features": features}}

    # Aggiorna stato con layers completi
    state.last_prediction['layers'] = out["layers"]

    return out


@router.get("/heatmap/density")
def density_heatmap(
    include_predictions: bool = Query(False),
    include_speed_categories: bool = Query(True),
):
    """Genera heatmap di densità del traffico marittimo"""
    if state.ais_df is None:
        raise HTTPException(status_code=400, detail="Carica prima i dati AIS con /api/ais/load")
    
    # Recupera dati predetti se disponibili
    prediction_data = None
    if include_predictions and hasattr(state, 'last_prediction'):
        prediction_data = state.last_prediction
    
    # Genera heatmap avanzata
    heatmap = _heatmap.create_advanced_density_heatmap(
        vessels_df=state.ais_df,
        prediction_data=prediction_data,
        include_speed_categories=include_speed_categories
    )
    
    return heatmap


@router.get("/heatmap/speed")
def speed_heatmap():
    """Genera heatmap stratificata per categoria di velocità"""
    if state.ais_df is None:
        raise HTTPException(status_code=400, detail="Carica prima i dati AIS con /api/ais/load")
    
    heatmap = _heatmap.create_speed_density_heatmap(state.ais_df)
    return heatmap


@router.get("/heatmap/temporal")
def temporal_heatmap(
    time_windows: List[str] = Query(["1h", "6h", "12h", "24h"])
):
    """Genera heatmap temporali multiple"""
    if state.ais_df is None:
        raise HTTPException(status_code=400, detail="Carica prima i dati AIS con /api/ais/load")
    
    heatmap = _heatmap.create_temporal_heatmap(state.ais_df, time_windows)
    return heatmap


@router.get("/temporal/comparison")
def temporal_comparison(
    reference_time: Optional[str] = None,
):
    """Crea confronto temporale tra dati reali e predizioni MSTFFN"""
    if state.ais_df is None:
        raise HTTPException(status_code=400, detail="Carica prima i dati AIS con /api/ais/load")
    
    if not hasattr(state, 'last_prediction') or state.last_prediction is None:
        raise HTTPException(status_code=400, detail="Esegui prima una predizione con /api/predict")
    
    # Converti tempo di riferimento
    ref_time = None
    if reference_time:
        ref_time = pd.to_datetime(reference_time, utc=True)
    
    # Crea confronto temporale
    comparison = _temporal.create_temporal_comparison(
        real_data=state.ais_df,
        prediction_data=state.last_prediction,
        reference_time=ref_time
    )
    
    return comparison


@router.get("/temporal/series")
def temporal_series(
    window_hours: int = Query(24, ge=1, le=168),
):
    """Crea analisi delle serie temporali del traffico"""
    if state.flow_series is None:
        raise HTTPException(status_code=400, detail="Carica prima i dati AIS con /api/ais/load")
    
    if not hasattr(state, 'last_prediction') or state.last_prediction is None:
        raise HTTPException(status_code=400, detail="Esegui prima una predizione con /api/predict")
    
    # Crea analisi serie temporali
    series_analysis = _temporal.create_time_series_analysis(
        flow_series=state.flow_series,
        prediction_data=state.last_prediction,
        window_hours=window_hours
    )
    
    return series_analysis


@router.post("/predict/single/{mmsi}")
def predict_single_vessel(
    mmsi: int,
    horizons: Optional[List[int]] = Query(None),
):
    """Esegue predizione MSTFFN per una singola nave (MMSI)"""
    if state.ais_df is None:
        raise HTTPException(status_code=400, detail="Carica prima i dati AIS con /api/ais/load")
    
    if not hasattr(state, 'last_prediction') or state.last_prediction is None:
        raise HTTPException(status_code=400, detail="Esegui prima una predizione globale con /api/predict")
    
    # Filtra predizioni globali per MMSI specifico
    prediction_data = state.last_prediction
    filtered_prediction = {
        't_ref': prediction_data['t_ref'],
        'horizons': prediction_data['horizons'],
        'horizons_min': prediction_data['horizons_min'],
        'flow_freq': prediction_data['flow_freq'],
        'horizon_steps': prediction_data['horizon_steps'],
        'mu': prediction_data['mu'],
        'sigma': prediction_data['sigma'],
        'layers': {}
    }
    
    # Filtra layer per MMSI
    for horizon_key, layer_data in prediction_data['layers'].items():
        if 'points' in layer_data and layer_data['points'].get('features'):
            filtered_features = [
                feature for feature in layer_data['points']['features']
                if feature.get('properties', {}).get('mmsi') == mmsi
            ]
            
            filtered_prediction['layers'][horizon_key] = {
                'points': {
                    'type': 'FeatureCollection',
                    'features': filtered_features
                }
            }
    
    return filtered_prediction


@router.get("/vessel/{mmsi}/info")
def get_vessel_info(mmsi: int):
    """Ottiene informazioni dettagliate su una singola nave"""
    if state.ais_df is None:
        raise HTTPException(status_code=400, detail="Carica prima i dati AIS con /api/ais/load")
    
    # Filtra dati per MMSI
    vessel_data = state.ais_df[state.ais_df['MMSI'] == mmsi]
    
    if len(vessel_data) == 0:
        raise HTTPException(status_code=404, detail=f"Nave con MMSI {mmsi} non trovata")
    
    # Calcola statistiche
    latest = vessel_data.iloc[-1]
    
    return {
        'mmsi': mmsi,
        'latest_position': {
            'lon': float(latest['Lon']),
            'lat': float(latest['Lat']),
            'timestamp': latest['timestamp'].isoformat(),
            'speed': float(latest['Speed']) if 'Speed' in latest else 0.0,
            'heading': float(latest['Heading']) if 'Heading' in latest else 0.0
        },
        'statistics': {
            'total_records': len(vessel_data),
            'time_span': {
                'start': vessel_data['timestamp'].min().isoformat(),
                'end': vessel_data['timestamp'].max().isoformat()
            },
            'avg_speed': float(vessel_data['Speed'].mean()) if 'Speed' in vessel_data.columns else 0.0,
            'max_speed': float(vessel_data['Speed'].max()) if 'Speed' in vessel_data.columns else 0.0
        }
    }


@router.post("/traffic/density_grid")
def density_grid(
    time_horizon_minutes: int = Query(120, ge=10, le=180, description="Orizzonte temporale in minuti (massimo 180)"),
    grid_resolution_km: float = Query(1.0, ge=0.5, le=5.0, description="Risoluzione griglia in km"),
):
    """
    Calcola la griglia di densità del traffico marittimo per la sicurezza dei piccoli veicoli non-AIS
    
    - Suddivide il Golfo di Napoli in una griglia di celle 1km
    - Calcola il numero di navi previste per cella ogni 10 minuti
    - Classifica il rischio: verde (0-2 navi), giallo (3-5), rosso (6+)
    """
    if state.ais_df is None:
        raise HTTPException(status_code=400, detail="Carica prima i dati AIS con /api/ais/load")
    
    if not hasattr(state, 'last_prediction') or state.last_prediction is None:
        raise HTTPException(status_code=400, detail="Esegui prima una predizione con /api/predict")
    
    try:
        # Estrai posizioni predette dallo stato
        prediction_data = state.last_prediction
        vessel_positions = []
        
        # Processa tutti i layer temporali delle previsioni
        for time_key, layer_data in prediction_data.get('layers', {}).items():
            if 'points' in layer_data and layer_data['points'].get('features'):
                time_min = int(time_key)  # tempo in minuti
                
                for feature in layer_data['points']['features']:
                    props = feature.get('properties', {})
                    geom = feature.get('geometry', {})
                    
                    if geom.get('type') == 'Point' and geom.get('coordinates'):
                        lon, lat = geom['coordinates']
                        
                        # Considera solo punti centrali (kind: "center") per densità
                        if props.get('kind') == 'center':
                            vessel_positions.append({
                                "lat": lat,
                                "lon": lon,
                                "time_minutes": time_min,
                                "mmsi": props.get('mmsi'),
                                "speed": props.get('speed', 0)
                            })
        
        # Calcola griglia di densità
        density_result = _traffic_density.calculate_density_grid(
            vessel_positions=vessel_positions,
            time_horizon_minutes=time_horizon_minutes,
            grid_resolution_km=grid_resolution_km
        )
        
        # Aggiungi statistiche aggiuntive
        total_vessels = sum(grid["vessel_count"] for grid in density_result["density_grids"].values())
        max_density = max(
            max(grid["density_grid"][i][j] 
                for i in range(len(grid["density_grid"])) 
                for j in range(len(grid["density_grid"][0])))
            for grid in density_result["density_grids"].values()
        )
        
        density_result["statistics"] = {
            "total_predicted_vessels": total_vessels,
            "max_vessels_per_cell": max_density,
            "timesteps_calculated": len(density_result["density_grids"]),
            "grid_cells": density_result["grid_metadata"]["grid_shape"][0] * density_result["grid_metadata"]["grid_shape"][1]
        }
        
        # Salva nello stato per uso futuro
        state.last_density_grid = density_result
        
        return density_result
        
    except Exception as e:
        logger.error(f"Errore nel calcolo della griglia di densità: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore nel calcolo della griglia: {str(e)}")


@router.get("/traffic/cell_info")
def get_cell_info(
    lat: float = Query(..., ge=-90, le=90, description="Latitudine della cella"),
    lon: float = Query(..., ge=-180, le=180, description="Longitudine della cella"),
):
    """
    Ottiene informazioni dettagliate sulla densità per una specifica cella della griglia
    
    Ritorna l'andamento temporale del numero di navi e del livello di rischio per quella cella
    """
    if not hasattr(state, 'last_density_grid') or state.last_density_grid is None:
        raise HTTPException(status_code=400, detail="Esegui prima il calcolo della griglia con /api/traffic/density_grid")
    
    try:
        cell_info = _traffic_density.get_cell_info(
            lat=lat,
            lon=lon, 
            density_data=state.last_density_grid
        )
        
        return cell_info
        
    except Exception as e:
        logger.error(f"Errore nel ottenere informazioni cella: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")
