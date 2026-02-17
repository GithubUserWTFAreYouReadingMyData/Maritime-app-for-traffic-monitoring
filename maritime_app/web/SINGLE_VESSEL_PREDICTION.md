# 🚢 Predizione Singola Nave - Funzionalità Avanzata

## 🎯 **Panoramica**

La funzionalità di **predizione singola nave** permette di visualizzare le previsioni MSTFFN per una specifica nave selezionata, con evidenziazione visiva e controlli dedicati.

---

## 🖱️ **Come Utilizzare**

### **1. Selezione Nave**
1. **Carica i dati AIS** dal pannello "Dati"
2. **Clicca su qualsiasi punto AIS** sulla mappa
3. Si aprirà un popup con dettagli della nave

### **2. Predizione Singola**
1. Nel popup della nave, clicca il pulsante **"🚢 Predici Nave"**
2. Il sistema eseguirà la predizione MSTFFN solo per quella nave
3. Appariranno solo le previsioni della nave selezionata

### **3. Reset Visualizzazione**
1. Usa il pulsante **"🔄 Mostra Tutte le Navi"** nel pannello
2. Torna alla visualizzazione globale di tutte le navi

---

## 🎨 **Caratteristiche Visuali**

### **Evidenziazione Nave Selezionata**
- **Punto AIS**: Bordo più spesso e opacità aumentata
- **Predizioni**: Marker rossi con bordo bianco prominente
- **Nuvola Probabilità**: Cerchi rossi con opacità evidenziata

### **Popup Migliorato**
```
📋 Dettagli Nave
━━━━━━━━━━━━━━━━━━━━━━━━
MMSI: 241019000
Velocità: 12.5 nodi
Rotta: 45°
Timestamp: 20/01/2026 08:33:20
━━━━━━━━━━━━━━━━━━━━━━━━
[ 🚢 Predici Nave ]
```

### **Controlli Dedicati**
- **Pannello "Previsioni"**: Mostra controlli per nave singola
- **Bottone Reset**: "🔄 Mostra Tutte le Navi"
- **Stato**: Indica visualizzazione singola attiva

---

## 🔧 **Implementazione Tecnica**

### **Frontend JavaScript**
```javascript
// Funzione principale per predizione singola
async function predictSingleVessel(mmsi) {
  selectedVessel = mmsi;
  
  // Chiamata API per predizione singola
  const response = await fetch(`${API}/predict/single/${mmsi}`, {
    method: 'POST',
    body: JSON.stringify({ horizons_min: [parseInt(horizon)] })
  });
  
  // Visualizzazione solo per nave selezionata
  displaySingleVesselPrediction(predictionData, mmsi);
}

// Reset a tutte le navi
function resetToAllVessels() {
  selectedVessel = null;
  document.getElementById('singleVesselControls').style.display = 'none';
  updatePredictionLayers(); // Mostra tutte le predizioni
}
```

### **Backend API**
```python
# Endpoint per predizione singola nave
@router.post("/predict/single/{mmsi}")
def predict_single_vessel(mmsi: int):
    # Filtra predizioni globali per MMSI specifico
    filtered_prediction = filter_prediction_by_mmsi(
        prediction_data=state.last_prediction,
        target_mmsi=mmsi
    )
    return filtered_prediction

# Endpoint info nave
@router.get("/vessel/{mmsi}/info")
def get_vessel_info(mmsi: int):
    # Statistiche dettagliate nave singola
    vessel_data = state.ais_df[state.ais_df['MMSI'] == mmsi]
    return vessel_statistics
```

---

## 📊 **Vantaggi Funzionali**

### **1. Analisi Focalizzata**
- **Focus su singola nave**: Elimina rumore visivo
- **Studio traiettoria individuale**: Analisi percorso specifico
- **Incertezza individuale**: Valutazione rischio nave-specifica

### **2. Interazione Utente**
- **Selezione intuitiva**: Click-to-predict
- **Feedback visivo immediato**: Evidenziazione nave
- **Controllo semplice**: Reset con un click

### **3. Applicazioni Pratiche**
- **Monitoraggio nave specifica**: Tracking individuale
- **Analisi rischio collisione**: Valutazione interazioni
- **Studio casi**: Ricerca su traiettorie anomale

---

## 🔄 **Flusso Operativo**

### **Workflow Standard**
```
1. Carica Dati AIS
   ↓
2. Clicca Nave Interesse
   ↓
3. "🚢 Predici Nave"
   ↓
4. Analisi Predizioni Singole
   ↓
5. "🔄 Mostra Tutte le Navi" (Reset)
```

### **Fallback System**
- Se API singola fallisce → Usa predizione globale filtrata
- Garantisce sempre funzionalità anche con backend limitato
- Messaggio errore user-friendly

---

## 🎯 **Casi d'Uso**

### **1. Ricerca Scientifica**
- **Studio traiettorie individuali**: Analisi pattern navigazione
- **Validazione modello**: Confronto predizione vs reale nave-specifica
- **Analisi incertezza**: Studio distribuzione probabilità singola

### **2. Operazioni Marittime**
- **Monitoraggio critico**: Navi ad alto valore/rischio
- **Pianificazione rotte**: Previsione percorso specifico
- **Sicurezza**: Valutazione rischio collisione

### **3. Training & Education**
- **Dimostrazione interattiva**: Insegnamento MSTFFN
- **Casi studio**: Esempi concreti predizione
- **Visualizzazione comprensibile**: Semplificazione concetti

---

## 🔮 **Estensioni Future**

### **1. Multi-Ship Selection**
- Selezione multiple navi con CTRL+click
- Confronto traiettorie parallele
- Analisi interazioni tra navi selezionate

### **2. Advanced Analytics**
- Metriche performance nave-specifiche
- Storico predizioni per nave
- Trend accuracy temporali

### **3. Alert System**
- Notifiche per anomalie traiettoria
- Warning rischio collisione
- Automatic vessel tracking

---

## 📈 **Impatto Scientifico**

Questa funzionalità aggiunge valore significativo:

1. **Validazione Granulare**: Test modello a livello singola nave
2. **Studio Casi Specifici**: Analisi situazioni particolari
3. **User Experience**: Interazione più intuitiva e focalizzata
4. **Research Applications**: Base per studi avanzati

---

## 🎉 **Conclusione**

La predizione singola nave trasforma il sistema da **analisi globale** a **studio focalizzato**, mantenendo coerenza scientifica e aggiungendo valore pratico per ricerca e operazioni marittime.

**Pronto per utilizzo immediato** nel sistema MSTFFN Maritime Traffic! 🚢⚓
