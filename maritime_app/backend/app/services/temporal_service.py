from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)


class TemporalService:
    """
    Servizio per l'analisi temporale e la sincronizzazione tra dati reali e predizioni
    Implementa la logica temporale coerente con l'articolo MSTFFN IEEE 2025
    """

    def __init__(self):
        self.reference_time = None
        self.time_windows = {
            'short': {'minutes': 30, 'description': 'Termine breve (30 min)'},
            'medium': {'minutes': 120, 'description': 'Termine medio (2 ore)'},
            'long': {'minutes': 360, 'description': 'Termine lungo (6 ore)'}
        }

    def create_temporal_comparison(self, 
                                 real_data: pd.DataFrame,
                                 prediction_data: Dict,
                                 reference_time: Optional[datetime] = None) -> Dict:
        """
        Crea confronto temporale tra dati reali e predizioni MSTFFN
        
        Args:
            real_data: DataFrame AIS con timestamp
            prediction_data: Risultati predizione MSTFFN
            reference_time: Tempo di riferimento per il confronto
            
        Returns:
            Dict con analisi temporale completa
        """
        if real_data is None or len(real_data) == 0:
            return self._empty_temporal_comparison()
        
        # Converti timestamp
        real_data = real_data.copy()
        real_data['timestamp'] = pd.to_datetime(real_data['timestamp'], utc=True)
        
        # Determina tempo di riferimento
        if reference_time is None:
            reference_time = real_data['timestamp'].max()
        else:
            reference_time = pd.to_datetime(reference_time, utc=True)
        
        self.reference_time = reference_time
        
        # Analizza predizioni per orizzonti temporali
        temporal_analysis = {}
        
        if 'horizons_min' in prediction_data and 'layers' in prediction_data:
            for i, horizon_min in enumerate(prediction_data['horizons_min']):
                horizon_key = str(horizon_min)
                if horizon_key in prediction_data['layers']:
                    pred_time = reference_time + timedelta(minutes=horizon_min)
                    
                    # Estrai dati reali vicini al tempo predetto
                    real_nearby = self._extract_real_nearby_time(real_data, pred_time, window_minutes=15)
                    
                    # Analizza confronto
                    comparison = self._analyze_prediction_vs_real(
                        real_nearby, 
                        prediction_data['layers'][horizon_key],
                        pred_time
                    )
                    
                    temporal_analysis[horizon_key] = {
                        'prediction_time': pred_time.isoformat(),
                        'horizon_minutes': horizon_min,
                        'comparison': comparison,
                        'real_vessels_count': len(real_nearby),
                        'predicted_points_count': len(prediction_data['layers'][horizon_key].get('points', {}).get('features', []))
                    }
        
        return {
            'type': 'temporal_comparison',
            'reference_time': reference_time.isoformat(),
            'temporal_analysis': temporal_analysis,
            'summary': self._create_temporal_summary(temporal_analysis),
            'metadata': {
                'total_real_vessels': len(real_data),
                'time_range': {
                    'start': real_data['timestamp'].min().isoformat(),
                    'end': real_data['timestamp'].max().isoformat()
                }
            }
        }

    def create_time_series_analysis(self, 
                                 flow_series: pd.DataFrame,
                                 prediction_data: Dict,
                                 window_hours: int = 24) -> Dict:
        """
        Crea analisi temporale delle serie storiche del traffico
        
        Args:
            flow_series: Serie del traffico storico
            prediction_data: Predizioni MSTFFN
            window_hours: Finestra temporale per analisi
            
        Returns:
            Dict con analisi delle serie temporali
        """
        if flow_series is None or len(flow_series) == 0:
            return self._empty_time_series_analysis()
        
        # Converti timestamp
        flow_series = flow_series.copy()
        flow_series['timestamp'] = pd.to_datetime(flow_series['timestamp'], utc=True)
        
        # Filtra per finestra temporale
        if self.reference_time:
            start_time = self.reference_time - timedelta(hours=window_hours)
            end_time = self.reference_time + timedelta(hours=window_hours)
            flow_series = flow_series[
                (flow_series['timestamp'] >= start_time) & 
                (flow_series['timestamp'] <= end_time)
            ]
        
        # Crea punti temporali per predizioni
        prediction_points = []
        if 'horizons_min' in prediction_data and 'mu' in prediction_data:
            for i, horizon_min in enumerate(prediction_data['horizons_min']):
                pred_time = self.reference_time + timedelta(minutes=horizon_min)
                prediction_points.append({
                    'timestamp': pred_time.isoformat(),
                    'predicted_flow': prediction_data['mu'][i],
                    'uncertainty': prediction_data['sigma'][i],
                    'horizon_minutes': horizon_min
                })
        
        # Calcola statistiche
        real_values = flow_series['value'].values if 'value' in flow_series.columns else []
        
        return {
            'type': 'time_series_analysis',
            'reference_time': self.reference_time.isoformat() if self.reference_time else None,
            'historical_data': {
                'timestamps': flow_series['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist(),
                'values': flow_series['value'].tolist() if 'value' in flow_series.columns else []
            },
            'predictions': prediction_points,
            'statistics': {
                'historical_mean': float(np.mean(real_values)) if len(real_values) > 0 else 0.0,
                'historical_std': float(np.std(real_values)) if len(real_values) > 0 else 0.0,
                'historical_min': float(np.min(real_values)) if len(real_values) > 0 else 0.0,
                'historical_max': float(np.max(real_values)) if len(real_values) > 0 else 0.0
            }
        }

    def _extract_real_nearby_time(self, df: pd.DataFrame, target_time: datetime, window_minutes: int = 15) -> pd.DataFrame:
        """Estrai dati reali vicini a un tempo target"""
        start_time = target_time - timedelta(minutes=window_minutes)
        end_time = target_time + timedelta(minutes=window_minutes)
        
        return df[
            (df['timestamp'] >= start_time) & 
            (df['timestamp'] <= end_time)
        ].copy()

    def _analyze_prediction_vs_real(self, 
                                   real_data: pd.DataFrame, 
                                   prediction_layer: Dict,
                                   prediction_time: datetime) -> Dict:
        """Analizza confronto tra predizione e dati reali"""
        
        # Statistiche dati reali
        real_stats = {
            'vessel_count': len(real_data),
            'mean_speed': float(real_data['Speed'].mean()) if 'Speed' in real_data.columns else 0.0,
            'speed_std': float(real_data['Speed'].std()) if 'Speed' in real_data.columns else 0.0,
            'area_coverage': self._calculate_area_coverage(real_data)
        }
        
        # Statistiche predizioni
        pred_features = prediction_layer.get('points', {}).get('features', [])
        pred_points = []
        
        for feature in pred_features:
            coords = feature['geometry']['coordinates']
            if len(coords) >= 2:
                pred_points.append([coords[0], coords[1]])
        
        pred_stats = {
            'point_count': len(pred_points),
            'mean_probability': float(np.mean([f['properties'].get('p', 1.0) for f in pred_features])),
            'area_coverage': self._calculate_area_coverage_points(pred_points)
        }
        
        # Calcola metriche di confronto
        comparison_metrics = self._calculate_comparison_metrics(real_data, pred_points)
        
        return {
            'real_statistics': real_stats,
            'prediction_statistics': pred_stats,
            'comparison_metrics': comparison_metrics,
            'prediction_time': prediction_time.isoformat()
        }

    def _calculate_area_coverage(self, df: pd.DataFrame) -> Dict:
        """Calcola area di copertura dei dati reali"""
        if len(df) == 0 or 'Lon' not in df.columns or 'Lat' not in df.columns:
            return {'area_km2': 0.0, 'bounds': None}
        
        lon_min, lon_max = df['Lon'].min(), df['Lon'].max()
        lat_min, lat_max = df['Lat'].min(), df['Lat'].max()
        
        # Approssimazione area in km² (formula semplificata)
        lat_center = (lat_min + lat_max) / 2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(lat_center))
        
        area_km2 = (lon_max - lon_min) * km_per_deg_lon * (lat_max - lat_min) * km_per_deg_lat
        
        return {
            'area_km2': float(area_km2),
            'bounds': {
                'lon_min': float(lon_min),
                'lon_max': float(lon_max),
                'lat_min': float(lat_min),
                'lat_max': float(lat_max)
            }
        }

    def _calculate_area_coverage_points(self, points: List) -> Dict:
        """Calcola area di copertura dei punti predetti"""
        if len(points) == 0:
            return {'area_km2': 0.0, 'bounds': None}
        
        points_array = np.array(points)
        lon_min, lon_max = points_array[:, 0].min(), points_array[:, 0].max()
        lat_min, lat_max = points_array[:, 1].min(), points_array[:, 1].max()
        
        # Approssimazione area
        lat_center = (lat_min + lat_max) / 2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(lat_center))
        
        area_km2 = (lon_max - lon_min) * km_per_deg_lon * (lat_max - lat_min) * km_per_deg_lat
        
        return {
            'area_km2': float(area_km2),
            'bounds': {
                'lon_min': float(lon_min),
                'lon_max': float(lon_max),
                'lat_min': float(lat_min),
                'lat_max': float(lat_max)
            }
        }

    def _calculate_comparison_metrics(self, real_data: pd.DataFrame, pred_points: List) -> Dict:
        """Calcola metriche di confronto tra reale e predetto"""
        if len(real_data) == 0 or len(pred_points) == 0:
            return {
                'spatial_overlap': 0.0,
                'density_ratio': 0.0,
                'center_distance_km': 0.0
            }
        
        # Calcola centroidi
        real_centroid = [
            real_data['Lon'].mean(),
            real_data['Lat'].mean()
        ]
        
        pred_centroid = np.mean(pred_points, axis=0) if len(pred_points) > 0 else [0, 0]
        
        # Distanza tra centroidi (in km)
        lat_center = (real_centroid[1] + pred_centroid[1]) / 2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(lat_center))
        
        distance_km = np.sqrt(
            ((pred_centroid[0] - real_centroid[0]) * km_per_deg_lon) ** 2 +
            ((pred_centroid[1] - real_centroid[1]) * km_per_deg_lat) ** 2
        )
        
        # Rapporto densità
        real_area = self._calculate_area_coverage(real_data)['area_km2']
        pred_area = self._calculate_area_coverage_points(pred_points)['area_km2']
        
        density_ratio = (len(pred_points) / max(pred_area, 0.1)) / (len(real_data) / max(real_area, 0.1))
        
        return {
            'spatial_overlap': max(0.0, 1.0 - distance_km / 10.0),  # Normalizzato
            'density_ratio': float(density_ratio),
            'center_distance_km': float(distance_km)
        }

    def _create_temporal_summary(self, temporal_analysis: Dict) -> Dict:
        """Crea riassunto dell'analisi temporale"""
        if not temporal_analysis:
            return {
                'total_comparisons': 0,
                'average_accuracy': 0.0,
                'best_horizon': None,
                'worst_horizon': None
            }
        
        accuracies = []
        center_distances = []
        
        for horizon_key, analysis in temporal_analysis.items():
            if 'comparison' in analysis and 'comparison_metrics' in analysis['comparison']:
                metrics = analysis['comparison']['comparison_metrics']
                accuracies.append(metrics.get('spatial_overlap', 0.0))
                center_distances.append(metrics.get('center_distance_km', 0.0))
        
        if accuracies:
            best_idx = np.argmax(accuracies)
            worst_idx = np.argmin(accuracies)
            horizon_keys = list(temporal_analysis.keys())
            
            return {
                'total_comparisons': len(temporal_analysis),
                'average_accuracy': float(np.mean(accuracies)),
                'average_center_distance_km': float(np.mean(center_distances)),
                'best_horizon': {
                    'horizon_minutes': temporal_analysis[horizon_keys[best_idx]]['horizon_minutes'],
                    'accuracy': float(accuracies[best_idx])
                },
                'worst_horizon': {
                    'horizon_minutes': temporal_analysis[horizon_keys[worst_idx]]['horizon_minutes'],
                    'accuracy': float(accuracies[worst_idx])
                }
            }
        
        return {
            'total_comparisons': len(temporal_analysis),
            'average_accuracy': 0.0,
            'average_center_distance_km': 0.0,
            'best_horizon': None,
            'worst_horizon': None
        }

    def _empty_temporal_comparison(self) -> Dict:
        """Ritorna confronto temporale vuoto"""
        return {
            'type': 'temporal_comparison',
            'reference_time': None,
            'temporal_analysis': {},
            'summary': {
                'total_comparisons': 0,
                'average_accuracy': 0.0,
                'best_horizon': None,
                'worst_horizon': None
            },
            'metadata': {
                'total_real_vessels': 0,
                'time_range': {'start': None, 'end': None}
            }
        }

    def _empty_time_series_analysis(self) -> Dict:
        """Ritorna analisi serie temporali vuota"""
        return {
            'type': 'time_series_analysis',
            'reference_time': None,
            'historical_data': {'timestamps': [], 'values': []},
            'predictions': [],
            'statistics': {
                'historical_mean': 0.0,
                'historical_std': 0.0,
                'historical_min': 0.0,
                'historical_max': 0.0
            }
        }
