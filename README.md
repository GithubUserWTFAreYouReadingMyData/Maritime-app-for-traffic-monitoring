[README_traffic_safety.md](https://github.com/user-attachments/files/25371269/README_traffic_safety.md)
# Maritime Traffic Safety - Documentazione Tecnica

## Sistema di Previsione Densità Traffico Marittimo

### Architettura del Sistema

**Componenti Principali:**
- **Backend**: FastAPI con modello MSTFFN (Multi-time Scale Temporal Feature Fusion Network)
- **Frontend**: Leaflet.js con griglia interattiva
- **Dati**: AIS reali da meteo.uniparthenope.it

### Funzionalità Implementate

#### 1. **Caricamento Dati AIS**
- Fonte: `https://data.meteo.uniparthenope.it/instruments/aisnet0/csv/aisnet_20260120Z082324.csv`
- Area: Golfo di Napoli (lat: 40.50-40.95, lon: 13.90-14.45)
- Endpoint: `POST /api/ais/load`

#### 2. **Modello MSTFFN (Non Modificato)**
- **Architettura**: Multi-time Scale Temporal Feature Fusion Network
- **Parametri**: d_model=128, n_heads=8, n_layers=4
- **Multi-scale**: Low (24h), Medium (72h), High (168h) windows
- **Output**: Previsioni flusso traffico (μ, σ) per timestep
- **Endpoint**: `POST /api/predict`

#### 3. **Calcolo Griglia Densità**
- **Risoluzione**: 1km × 1km (configurabile: 0.5km, 1km, 2km)
- **Area**: Golfo di Napoli completo
- **Timestep**: 10 minuti (fino a 3 ore = 18 timestep)
- **Soglie Rischio**:
  - 🟢 **Basso**: 0-2 navi per cella
  - 🟡 **Medio**: 3-5 navi per cella  
  - 🔴 **Alto**: 6+ navi per cella
- **Endpoint**: `POST /api/traffic/density_grid`

#### 4. **Visualizzazione Interattiva**
- **Mappa**: Leaflet.js con OpenStreetMap
- **Griglia**: Tutte le celle cliccabili (vuote e occupate)
- **Navi AIS**: Cerchi colorati per velocità
- **Traiettorie**: Linee blu che mostrano percorso storico
- **Timeline**: Click su cella per dettagli temporali

### Interfaccia Utente

#### **Pannello Controllo**
- **Orizzonte Temporale**: 60/120/180 minuti
- **Risoluzione Griglia**: 0.5/1.0/2.0 km
- **Toggle Traiettorie**: 
  - ✅ Attivato: Mostra tutte le posizioni storiche + traiettorie
  - ❌ Disattivato: Mostra solo posizioni attuali delle navi
- **Statistiche Real-time**: Navi totali, densità massima, celle griglia, timestep

#### **Legenda Completa**
- **Livelli Rischio**: Verde (basso), Giallo (medio), Rosso (alto)
- **Velocità Navi**: Blu (lento ≤5 nodi), Arancione (medio 5-15), Rosso (veloce ≥15)
- **Elementi**: Traiettorie blu, Celle vuote grigie

### Flusso Operativo

#### **1. Caricamento Dati**
```
1. Load AIS Data → Scarica e processa dati AIS da Uniparthenope
2. Show AIS Vessels → Visualizza navi con/senza traiettorie storiche
```

#### **2. Previsione Traffico**
```
1. Predict Traffic → Esegue modello MSTFFN su dati AIS
2. Calcola griglia densità per tutti i timestep
3. Mostra griglia interattiva con colori rischio
```

#### **3. Analisi Sicurezza**
```
1. Click su cella griglia → Timeline dettagliata rischio
2. Toggle traiettorie → Analisi flusso traffico completo
3. Selettore timestep → Visualizzazione evoluzione temporale
```

### File Principali

#### **Backend**
- `traffic_density_service.py`: Servizio calcolo griglia densità
- `mstffn_service.py`: Modello MSTFFN originale (non modificato)
- `routes.py`: API endpoints per densità e AIS
- `config.py`: Configurazione percorsi e limiti geografici

#### **Frontend**
- `traffic_safety_fixed.html`: Interfaccia utente completa
- **Funzionalità**: Mappa interattiva, griglia rischio, navi AIS

### Specifiche Tecniche

#### **Griglia Densità**
- **Dimensioni**: Variabili based su risoluzione selezionata
- **Coordinate**: Calcolate da limiti Golfo di Napoli
- **Aggiornamento**: Manuale tramite pulsante "Predict Traffic"
- **Persistenza**: Dati salvati in `state.last_density_grid`

#### **Visualizzazione Navi**
- **Modalità Attuale**: Solo ultime posizioni (latest=true)
- **Modalità Completa**: Tutte le posizioni storiche (trajectories)
- **Colori**: Basati su velocità reali non aggregate

### Sicurezza per Piccoli Veicoli Non-AIS

#### **Zone Pericolose Identificate**
- **Alta Densità**: >6 navi per km² → Evitare assolutamente
- **Media Densità**: 3-5 navi per km² → Navigazione con cautela
- **Bassa Densità**: 0-2 navi per km² → Zone relativamente sicure

#### **Pattern Traffico**
- **Corridoi Principali**: Identificabili dalle traiettorie
- **Punti Critici**: Intersezioni ad alta densità
- **Evoluzione Temporale**: Previsione fino a 3 ore

### Utilizzo Pratico

#### **Per Navigatori Non-AIS**
1. **Carica dati** e **esegui previsione**
2. **Identifica zone sicure** (celle verdi) nel proprio percorso
3. **Monitora evoluzione** rischio tramite selettore timestep
4. **Pianifica rotta** evitando celle rosse/arancioni

#### **Per Autorità Marittime**
1. **Analisi pattern** traffico in tempo reale
2. **Identificazione aree** ad alto rischio collisione
3. **Pianificazione** misure sicurezza specifiche
4. **Monitoraggio** evoluzione situazione temporale


---
**Versione**: 1.0  
**Data**: 17 Febbraio 2026  
**Stato**: Completato e testato
