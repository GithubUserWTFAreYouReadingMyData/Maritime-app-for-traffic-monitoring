from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HeatmapService:
    grid_size: Tuple[int, int] = (80, 80)
    sigma: float = 1.2

    def density_grid_from_points(self, df: pd.DataFrame) -> Dict:
        b = settings.test_area_bounds
        nx, ny = self.grid_size
        lon_bins = np.linspace(b["lon_min"], b["lon_max"], nx + 1)
        lat_bins = np.linspace(b["lat_min"], b["lat_max"], ny + 1)

        H, _, _ = np.histogram2d(df["Lon"], df["Lat"], bins=[lon_bins, lat_bins])
        H = gaussian_filter(H, sigma=self.sigma)
        if H.max() > 0:
            H = H / H.max()

        return {
            "bounds": b,
            "grid_size": [nx, ny],
            "values": H.tolist(),
        }

    def density_grid_from_trajectories(
        self,
        df: pd.DataFrame,
        vessel_limit: int = 400,
        max_points: int = 200_000,
    ) -> Dict:
        """Costruisce una densità lungo i tragitti.

        Densifica i segmenti tra punti consecutivi per MMSI per evitare heatmap "solo sui punti".
        """
        if df is None or len(df) == 0:
            return self.density_grid_from_points(df)

        b = settings.test_area_bounds
        nx, ny = self.grid_size
        lon_bins = np.linspace(b["lon_min"], b["lon_max"], nx + 1)
        lat_bins = np.linspace(b["lat_min"], b["lat_max"], ny + 1)

        d = df[["MMSI", "Lon", "Lat", "timestamp"]].copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.sort_values(["MMSI", "timestamp"])

        mmsis = d["MMSI"].unique().tolist()
        if len(mmsis) > vessel_limit:
            mmsis = list(np.random.RandomState(42).choice(mmsis, vessel_limit, replace=False))
            d = d[d["MMSI"].isin(mmsis)]

        xs: list[float] = []
        ys: list[float] = []

        # parametro: ~0.0015 deg ~ 150m in lat
        step_deg = 0.0015

        for _, g in d.groupby("MMSI"):
            lon = g["Lon"].to_numpy(dtype=float)
            lat = g["Lat"].to_numpy(dtype=float)
            if len(lon) < 2:
                continue

            for i in range(len(lon) - 1):
                x0, y0 = lon[i], lat[i]
                x1, y1 = lon[i + 1], lat[i + 1]

                dx = x1 - x0
                dy = y1 - y0
                dist = float(np.hypot(dx, dy))
                n = int(max(2, min(40, dist / step_deg)))

                for t in np.linspace(0.0, 1.0, n, dtype=float):
                    xs.append(x0 + dx * t)
                    ys.append(y0 + dy * t)
                    if len(xs) >= max_points:
                        break
                if len(xs) >= max_points:
                    break
            if len(xs) >= max_points:
                break

        if len(xs) == 0:
            return self.density_grid_from_points(df)

        H, _, _ = np.histogram2d(xs, ys, bins=[lon_bins, lat_bins])
        H = gaussian_filter(H, sigma=self.sigma)
        if H.max() > 0:
            H = H / H.max()

        return {
            "bounds": b,
            "grid_size": [nx, ny],
            "values": H.tolist(),
        }

    def prior_density(self) -> Dict:
        # simple gaussian prior around center
        b = settings.test_area_bounds
        nx, ny = self.grid_size
        lon = np.linspace(b["lon_min"], b["lon_max"], nx)
        lat = np.linspace(b["lat_min"], b["lat_max"], ny)
        Lon, Lat = np.meshgrid(lon, lat, indexing="ij")
        c_lon = (b["lon_min"] + b["lon_max"]) / 2
        c_lat = (b["lat_min"] + b["lat_max"]) / 2
        Z = np.exp(-(((Lon - c_lon) ** 2) / 0.01 + ((Lat - c_lat) ** 2) / 0.006))
        Z = Z / Z.max()
        return {"bounds": b, "grid_size": [nx, ny], "values": Z.tolist()}

    def scale_density(self, prior_grid: Dict, scale: float) -> Dict:
        Z = np.array(prior_grid["values"], dtype=np.float32)
        Z = Z * float(scale)
        if Z.max() > 0:
            Z = Z / Z.max()
        out = dict(prior_grid)
        out["values"] = Z.tolist()
        return out

    def sample_points_from_grid(self, grid: Dict, n_points: int = 800) -> Dict:
        b = grid["bounds"]
        Z = np.array(grid["values"], dtype=np.float64)
        if Z.sum() <= 0:
            return {"type": "FeatureCollection", "features": []}

        Z = Z / Z.sum()
        nx, ny = Z.shape
        flat = Z.reshape(-1)
        idx = np.random.choice(np.arange(flat.size), size=min(n_points, flat.size), replace=True, p=flat)

        xs = idx // ny
        ys = idx % ny

        lon_min, lon_max = b["lon_min"], b["lon_max"]
        lat_min, lat_max = b["lat_min"], b["lat_max"]

        dlon = (lon_max - lon_min) / nx
        dlat = (lat_max - lat_min) / ny

        feats = []
        for x, y in zip(xs, ys):
            lon = lon_min + (x + np.random.rand()) * dlon
            lat = lat_min + (y + np.random.rand()) * dlat
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                    "properties": {"weight": 1.0},
                }
            )

        return {"type": "FeatureCollection", "features": feats}

    def create_advanced_density_heatmap(self, 
                                     vessels_df: pd.DataFrame,
                                     prediction_data: Optional[Dict] = None,
                                     include_speed_categories: bool = True) -> Dict:
        """
        Crea heatmap avanzata con KDE e supporto per predizioni MSTFFN
        Coerente con l'articolo IEEE 2025
        """
        if vessels_df is None or len(vessels_df) == 0:
            return self._empty_advanced_heatmap()
        
        b = settings.test_area_bounds
        nx, ny = self.grid_size
        
        # Crea griglia ad alta risoluzione
        lon_grid = np.linspace(b["lon_min"], b["lon_max"], nx)
        lat_grid = np.linspace(b["lat_min"], b["lat_max"], ny)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Calcola densità base con KDE
        density_real = self._compute_kde_density_advanced(vessels_df, lon_mesh, lat_mesh)
        
        result = {
            'type': 'advanced_heatmap',
            'bounds': b,
            'grid': {
                'lon': lon_grid.tolist(),
                'lat': lat_grid.tolist()
            },
            'density_real': density_real.tolist(),
            'metadata': {
                'total_vessels': len(vessels_df),
                'grid_resolution': [nx, ny],
                'max_density': float(np.max(density_real)),
                'mean_density': float(np.mean(density_real))
            }
        }
        
        # Aggiungi densità predetta se disponibile
        if prediction_data and 'layers' in prediction_data:
            density_pred = self._compute_prediction_density_advanced(prediction_data, lon_mesh, lat_mesh)
            result['density_pred'] = density_pred.tolist()
            result['metadata']['has_predictions'] = True
        else:
            result['density_pred'] = None
            result['metadata']['has_predictions'] = False
        
        # Aggiungi categorie di velocità se richiesto
        if include_speed_categories and 'Speed' in vessels_df.columns:
            speed_heatmaps = self._create_speed_category_heatmaps(vessels_df, lon_mesh, lat_mesh)
            result['speed_categories'] = speed_heatmaps
        
        # Calcola gradienti per aree di cambiamento rapido
        gradient_magnitude = self._compute_gradient_magnitude_advanced(density_real)
        result['gradient'] = gradient_magnitude.tolist()
        
        return result

    def _compute_kde_density_advanced(self, df: pd.DataFrame, lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> np.ndarray:
        """Calcola densità usando Kernel Density Estimation avanzato"""
        if len(df) < 2:
            return np.zeros_like(lon_mesh)
        
        points = df[['Lon', 'Lat']].values
        points = points[~np.isnan(points).any(axis=1)]
        
        if len(points) < 2:
            return np.zeros_like(lon_mesh)
        
        try:
            # KDE con bandwidth adattivo
            kde = gaussian_kde(points.T, bw_method='scott')
            positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
            density = kde(positions).reshape(lon_mesh.shape)
            
            # Normalizzazione
            if np.max(density) > 0:
                density = density / np.max(density)
            
            # Applica filtro Gaussian per smoothness
            density = gaussian_filter(density, sigma=self.sigma)
            
            return density
            
        except Exception as e:
            logger.warning(f"Errore nel calcolo KDE avanzato: {e}")
            return np.zeros_like(lon_mesh)

    def _compute_prediction_density_advanced(self, prediction_data: Dict, lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> np.ndarray:
        """Calcola densità da dati di predizione MSTFFN"""
        density = np.zeros_like(lon_mesh)
        
        if 'layers' not in prediction_data:
            return density
        
        for horizon, layer_data in prediction_data['layers'].items():
            if 'points' in layer_data and 'features' in layer_data['points']:
                points = []
                weights = []
                
                for feature in layer_data['points']['features']:
                    coords = feature['geometry']['coordinates']
                    if len(coords) >= 2:
                        points.append([coords[0], coords[1]])
                        weight = feature['properties'].get('p', 1.0)
                        weights.append(weight)
                
                if len(points) > 1:
                    points = np.array(points)
                    weights = np.array(weights)
                    
                    try:
                        kde = gaussian_kde(points.T, weights=weights, bw_method='scott')
                        positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()])
                        layer_density = kde(positions).reshape(lon_mesh.shape)
                        
                        density = (density + layer_density) / 2
                        
                    except Exception as e:
                        logger.warning(f"Errore nel calcolo densità predizione: {e}")
        
        return gaussian_filter(density, sigma=self.sigma)

    def _create_speed_category_heatmaps(self, df: pd.DataFrame, lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> Dict:
        """Crea heatmap per categorie di velocità"""
        speed_categories = {
            'slow': (0, 5),
            'medium': (5, 15), 
            'fast': (15, 50)
        }
        
        heatmaps = {}
        
        for category, (min_speed, max_speed) in speed_categories.items():
            category_df = df[
                (df['Speed'] >= min_speed) & 
                (df['Speed'] < max_speed)
            ]
            
            if len(category_df) > 0:
                density = self._compute_kde_density_advanced(category_df, lon_mesh, lat_mesh)
                heatmaps[category] = density.tolist()
            else:
                heatmaps[category] = np.zeros_like(lon_mesh).tolist()
        
        return heatmaps

    def _compute_gradient_magnitude_advanced(self, density: np.ndarray) -> np.ndarray:
        """Calcola magnitudine del gradiente"""
        gy, gx = np.gradient(density)
        magnitude = np.sqrt(gx**2 + gy**2)
        return magnitude

    def _empty_advanced_heatmap(self) -> Dict:
        """Ritorna heatmap avanzata vuota"""
        b = settings.test_area_bounds
        nx, ny = self.grid_size
        
        lon_grid = np.linspace(b["lon_min"], b["lon_max"], nx)
        lat_grid = np.linspace(b["lat_min"], b["lat_max"], ny)
        empty_density = np.zeros((ny, nx))
        
        return {
            'type': 'advanced_heatmap',
            'bounds': b,
            'grid': {
                'lon': lon_grid.tolist(),
                'lat': lat_grid.tolist()
            },
            'density_real': empty_density.tolist(),
            'density_pred': None,
            'gradient': empty_density.tolist(),
            'speed_categories': {
                'slow': empty_density.tolist(),
                'medium': empty_density.tolist(),
                'fast': empty_density.tolist()
            },
            'metadata': {
                'total_vessels': 0,
                'grid_resolution': [nx, ny],
                'max_density': 0.0,
                'mean_density': 0.0,
                'has_predictions': False
            }
        }
