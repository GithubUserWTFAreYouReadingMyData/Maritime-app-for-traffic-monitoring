# MSTFFN Maritime Traffic Prediction System

## Sistema Completo di Previsione del Traffico Marittimo

Basato sull'articolo scientifico IEEE 2025: *"AIS Data-Driven Maritime Traffic Flow Prediction and Density Visualization Using Multitime Scale Temporal Feature Fusion Network"*

---

## 🏗️ Architettura del Sistema

### Backend (Python/FastAPI)
- **MSTFFN Model**: Implementazione PyTorch del modello Transformer con Multi-Head Attention
- **AIS Data Processing**: Preprocessing dati AIS reali da meteo.uniparthenope
- **REST API**: Endpoint per previsioni, heatmap e analisi temporale
- **Services Modulari**: 
  - `AIService`: Gestione dati AIS
  - `MSTFFNService`: Inferenza del modello
  - `HeatmapService`: Generazione heatmap densità
  - `TemporalService`: Analisi temporale e confronti

### Frontend (Web GIS)
- **Leaflet.js**: Mappa interattiva con layer multipli
- **Visualizzazioni**:
  - Punti AIS reali con codifica colore velocità
  - Nuvola di probabilità predetta MSTFFN
  - Heatmap di densità del traffico
  - Traiettorie temporali
- **Dashboard**: Metriche in tempo reale e controlli avanzati

---

## 🚀 Avvio Rapido

### 1. Backend
```bash
cd maritime_app/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8007 --reload
```

### 2. Frontend
Apri `web/enhanced.html` nel browser o usa un web server locale.

---

## 📊 Funzionalità Principali

### 🔮 Previsioni MSTFFN
- **Orizzonti temporali**: 5, 15, 30, 60 minuti
- **Output probabilistico**: μ (media) e σ (deviazione standard)
- **Nuvola di punti**: Campionamento Monte Carlo per visualizzazione incertezza
- **Propagazione kinematica**: Integrazione speed/heading reali

### 🗺️ Visualizzazioni GIS
- **Layer Dati Reali**: Punti AIS con colore per velocità
- **Layer Predizioni**: Punti centrali e nuvola probabilità
- **Heatmap Densità**: KDE con smoothing Gaussiano
- **Heatmap Temporali**: Analisi dinamiche traffico
- **Categorie Velocità**: Slow (0-5), Medium (5-15), Fast (15+ nodi)

### 📈 Analisi Temporale
- **Confronto Reale vs Predetto**: Metriche spaziali e temporali
- **Serie Storiche**: Andamento traffico con previsioni
- **Metriche Performance**: Accuratezza, distanza centroidi, overlap spaziale

---

## 🛠️ API Endpoints

### Dati AIS
```
POST /api/ais/load          # Carica dataset AIS
GET  /api/ais/points        # Punti AIS filtrati
GET  /api/ais/trajectories  # Traiettorie navi
GET  /api/ais/speed_stats   # Statistiche velocità
```

### Previsioni MSTFFN
```
POST /api/predict            # Esegui previsione
GET  /api/model/status       # Stato modello
POST /api/model/save        # Salva pesi
POST /api/model/load        # Carica pesi
```

### Heatmap
```
GET /api/heatmap/density     # Heatmap densità traffico
GET /api/heatmap/speed       # Heatmap per categoria velocità
GET /api/heatmap/temporal    # Heatmap temporali multiple
```

### Analisi Temporale
```
GET /api/temporal/comparison # Confronto reale vs predetto
GET /api/temporal/series    # Analisi serie temporali
```

---

## 📁 Struttura del Progetto

```
maritime_app/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py          # API endpoints
│   │   ├── core/
│   │   │   └── config.py          # Configurazione
│   │   ├── mstffn/
│   │   │   └── model.py           # Modello MSTFFN
│   │   ├── services/
│   │   │   ├── ais_service.py     # Gestione AIS
│   │   │   ├── mstffn_service.py # Inferenza modello
│   │   │   ├── heatmap_service.py # Heatmap densità
│   │   │   └── temporal_service.py# Analisi temporale
│   │   └── main.py              # FastAPI app
│   └── requirements.txt
└── web/
    ├── index.html                 # Frontend base
    ├── enhanced.html              # Frontend avanzato
    └── README.md                # Questa documentazione
```

---

## 🎯 Coerenza con Articolo IEEE 2025

### Modello MSTFFN
- ✅ **Multi-time Scale**: Low (24h), Medium (72h), High (168h)
- ✅ **Transformer + Multi-Head Attention**: 8 heads, 4 layers
- ✅ **Positional Time Encoding**: Feature temporali normalizzate
- ✅ **Output Probabilistico**: Distribuzione Gaussiana (μ, σ)

### Visualizzazioni Scientifiche
- ✅ **Figura 5**: Traiettorie e gate lines
- ✅ **Figura 6**: Heatmap densità traffico giornaliero
- ✅ **Tabella 2**: Metriche performance (RMSE, MAE, MAPE)
- ✅ **Confronto modelli**: MST-GRU, MST-LSTM, MST-BiLSTM, etc.

### Pipeline Dati
- ✅ **Dati reali**: meteo.uniparthenope 2026
- ✅ **Area test**: Golfo di Napoli (13.90-14.45°E, 40.50-40.95°N)
- ✅ **Preprocessing**: Normalizzazione, filtering, interpolation
- ✅ **Validazione**: Separazione training/validation temporale

---

## 🔬 Metriche di Valutazione

### Performance Predizione
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficiente di determinazione

### Metriche Spaziali
- **Spatial Overlap**: Sovrapposizione aree predette/reali
- **Centroid Distance**: Distanza tra centroidi (km)
- **Density Ratio**: Rapporto densità predetta/rea

### Incertezza
- **Prediction Uncertainty**: Deviazione standard predizioni
- **Confidence Intervals**: Intervallo 95% per μ
- **Probability Cloud**: Visualizzazione Monte Carlo

---

## 🚧 Estensioni Future

### Machine Learning
- **Transfer Learning**: Adattamento ad altre aree geografiche
- **Multi-modal Fusion**: Integrazione dati meteo/marini
- **Online Learning**: Aggiornamento continuo con nuovi dati

### Visualizzazioni Avanzate
- **3D Visualization**: Altitudine e profilo batimetrico
- **Real-time Streaming**: WebSocket per aggiornamenti live
- **VR/AR Support**: Visualizzazione immersiva

### Performance
- **GPU Acceleration**: CUDA per inferenza batch
- **Edge Computing**: Deploy su dispositivi bordo
- **Microservices**: Scalabilità orizzontale

---

## 📚 Riferimenti

1. **Articolo Principale**: IEEE Sensors Journal 2025
2. **Dataset AIS**: meteo.uniparthenope.it
3. **Framework**: PyTorch, FastAPI, Leaflet.js
4. **Algoritmi**: KDE, Transformer, Multi-Head Attention

---

## 👥 Team di Sviluppo

- **AI Software Engineer**: Architettura sistema e modello MSTFFN
- **Deep Learning Researcher**: Ottimizzazione e validazione
- **GIS Developer**: Visualizzazioni e frontend
- **Backend Engineer**: API e servizi

---

*Questo sistema rappresenta un prototipo di ricerca completo per la previsione del traffico marittimo, pienamente coerente con lo stato dell'arte scientifico nel settore.*
