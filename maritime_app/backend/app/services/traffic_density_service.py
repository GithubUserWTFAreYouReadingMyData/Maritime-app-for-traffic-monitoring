from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrafficDensityService:
    """Servizio per il calcolo della griglia di densità del traffico marittimo"""
    
    def __init__(self):
        # Limiti geografici del Golfo di Napoli
        self.gulf_bounds = {
            "north": 40.95,
            "south": 40.50, 
            "east": 14.45,
            "west": 13.90
        }
        
        # Soglie di rischio per densità navi per cella 1km
        self.risk_thresholds = {
            "low": 2,      # 0-2 navi = verde
            "medium": 5,   # 3-5 navi = giallo  
            "high": 999    # 6+ navi = rosso
        }
    
    def create_grid(self, grid_resolution_km: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea la griglia geografica per il Golfo di Napoli
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (lon_grid, lat_grid) coordinate delle celle
        """
        # Conversione: 1 grado lat ≈ 111 km, 1 grado lon ≈ 111km * cos(lat)
        lat_center = (self.gulf_bounds["north"] + self.gulf_bounds["south"]) / 2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(lat_center))
        
        # Dimensioni griglia
        lat_range = self.gulf_bounds["north"] - self.gulf_bounds["south"]
        lon_range = self.gulf_bounds["east"] - self.gulf_bounds["west"]
        
        lat_cells = int(np.ceil(lat_range / (grid_resolution_km / km_per_deg_lat)))
        lon_cells = int(np.ceil(lon_range / (grid_resolution_km / km_per_deg_lon)))
        
        # Coordinate delle celle
        lat_step = lat_range / lat_cells
        lon_step = lon_range / lon_cells
        
        lat_coords = np.linspace(
            self.gulf_bounds["south"] + lat_step/2,
            self.gulf_bounds["north"] - lat_step/2,
            lat_cells
        )
        lon_coords = np.linspace(
            self.gulf_bounds["west"] + lon_step/2,
            self.gulf_bounds["east"] - lon_step/2,
            lon_cells
        )
        
        return lon_coords, lat_coords
    
    def point_to_grid_cell(self, lat: float, lon: float, lon_coords: np.ndarray, lat_coords: np.ndarray) -> Tuple[int, int]:
        """
        Converte coordinate lat/lon in indici della griglia
        
        Returns:
            Tuple[int, int]: (lon_idx, lat_idx) indici della cella
        """
        lon_idx = np.argmin(np.abs(lon_coords - lon))
        lat_idx = np.argmin(np.abs(lat_coords - lat))
        return lon_idx, lat_idx
    
    def calculate_density_grid(
        self, 
        vessel_positions: List[Dict], 
        time_horizon_minutes: int = 180,
        grid_resolution_km: float = 1.0
    ) -> Dict:
        """
        Calcola la griglia di densità per tutti i timestep
        
        Args:
            vessel_positions: Lista di posizioni nave con timestamp
            time_horizon_minutes: Orizzonte temporale massimo (minuti)
            grid_resolution_km: Risoluzione griglia in km
            
        Returns:
            Dict: Griglia di densità con metadati
        """
        # Crea griglia
        lon_coords, lat_coords = self.create_grid(grid_resolution_km)
        n_lon = len(lon_coords)
        n_lat = len(lat_coords)
        
        # Timestep da 10 minuti fino all'orizzonte massimo
        timesteps = list(range(10, min(time_horizon_minutes + 1, 181), 10))
        
        # Inizializza griglie per ogni timestep
        density_grids = {}
        
        for t_min in timesteps:
            # Filtra posizioni per questo timestep
            positions_at_t = [
                pos for pos in vessel_positions 
                if pos.get("time_minutes") == t_min
            ]
            
            # Crea griglia di conteggio
            density_grid = np.zeros((n_lat, n_lon), dtype=int)
            
            # Conta navi per cella
            for pos in positions_at_t:
                try:
                    lat = float(pos.get("lat", 0))
                    lon = float(pos.get("lon", 0))
                    
                    # Verifica che sia nel Golfo di Napoli
                    if (self.gulf_bounds["south"] <= lat <= self.gulf_bounds["north"] and
                        self.gulf_bounds["west"] <= lon <= self.gulf_bounds["east"]):
                        
                        lon_idx, lat_idx = self.point_to_grid_cell(lat, lon, lon_coords, lat_coords)
                        density_grid[lat_idx, lon_idx] += 1
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Posizione non valida: {pos}, errore: {e}")
                    continue
            
            # Converti in livelli di rischio
            risk_grid = self._density_to_risk(density_grid)
            
            density_grids[str(t_min)] = {
                "time_minutes": t_min,
                "density_grid": density_grid.tolist(),
                "risk_grid": risk_grid.tolist(),
                "vessel_count": int(np.sum(density_grid))
            }
        
        return {
            "grid_metadata": {
                "bounds": self.gulf_bounds,
                "grid_resolution_km": grid_resolution_km,
                "lon_coords": lon_coords.tolist(),
                "lat_coords": lat_coords.tolist(),
                "grid_shape": [n_lat, n_lon],
                "time_horizon_minutes": time_horizon_minutes,
                "timesteps": timesteps,
                "risk_thresholds": self.risk_thresholds
            },
            "density_grids": density_grids
        }
    
    def _density_to_risk(self, density_grid: np.ndarray) -> np.ndarray:
        """
        Converte griglia di densità in livelli di rischio
        
        Args:
            density_grid: Griglia con conteggio navi per cella
            
        Returns:
            np.ndarray: Griglia con livelli rischio (0=verde, 1=giallo, 2=rosso)
        """
        risk_grid = np.zeros_like(density_grid)
        
        # Medio rischio (giallo)
        risk_grid[(density_grid > self.risk_thresholds["low"]) & 
                 (density_grid <= self.risk_thresholds["medium"])] = 1
        
        # Alto rischio (rosso)
        risk_grid[density_grid > self.risk_thresholds["medium"]] = 2
        
        return risk_grid
    
    def get_cell_info(
        self, 
        lat: float, 
        lon: float, 
        density_data: Dict
    ) -> Dict:
        """
        Ottiene informazioni dettagliate per una specifica cella
        
        Args:
            lat, lon: Coordinate della cella
            density_data: Dati di densità calcolati
            
        Returns:
            Dict: Informazioni cella per tutti i timestep
        """
        lon_coords = np.array(density_data["grid_metadata"]["lon_coords"])
        lat_coords = np.array(density_data["grid_metadata"]["lat_coords"])
        
        try:
            lon_idx, lat_idx = self.point_to_grid_cell(lat, lon, lon_coords, lat_coords)
        except ValueError:
            return {"error": "Coordinate fuori dalla griglia"}
        
        cell_info = {
            "coordinates": {"lat": lat, "lon": lon},
            "grid_indices": {"lat_idx": int(lat_idx), "lon_idx": int(lon_idx)},
            "timesteps": []
        }
        
        # Estrai dati per ogni timestep
        for t_key, t_data in density_data["density_grids"].items():
            density = t_data["density_grid"][lat_idx][lon_idx]
            risk = t_data["risk_grid"][lat_idx][lon_idx]
            
            cell_info["timesteps"].append({
                "time_minutes": t_data["time_minutes"],
                "vessel_count": density,
                "risk_level": risk,
                "risk_label": self._risk_level_to_label(risk)
            })
        
        return cell_info
    
    def _risk_level_to_label(self, risk_level: int) -> str:
        """Converte livello rischio numerico in etichetta"""
        labels = {0: "basso", 1: "medio", 2: "alto"}
        return labels.get(risk_level, "sconosciuto")
