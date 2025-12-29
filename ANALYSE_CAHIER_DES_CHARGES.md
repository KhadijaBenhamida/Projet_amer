# ANALYSE PROFONDE: Cahier des charges vs État actuel du projet

## 📋 OBJECTIF REEL DU PROJET (selon cahier des charges)

### Titre officiel:
**"Prédiction des événements climatiques extrêmes à partir de données météorologiques historiques"**

### Objectifs principaux:
1. **Prédire des anomalies ou événements extrêmes** (canicules, fortes précipitations, vagues de froid, sécheresse)
2. Concevoir un système d'aide à la décision
3. Anticiper les risques et mesures préventives

### Points CRITIQUES du cahier des charges:

> **"Il doit surtout focaliser l'entraînement sur la détection et prédiction des événements extrêmes, qui sont rares et nécessitent un traitement spécifique"**

---

## ⚠️ GAP MAJEUR: Cahier des charges vs Implémentation actuelle

### Ce que le projet DEVRAIT faire:

1. **Détection événements extrêmes**
   - Canicule: Température > 42°C pendant 3 jours
   - Vague de froid: Température < -5°C pendant 3 jours
   - Fortes précipitations
   - Sécheresse

2. **Ontologie climatique**
   - Représentation des phénomènes extrêmes
   - Relations avec variables météo
   - Règles d'inférence: `IF Température > 42°C pendant 3 jours THEN Canicule`
   - Moteur de règles pour alertes

3. **Deep Learning focalisé sur les extrêmes**
   - Traitement spécifique des événements rares
   - Techniques de rééquilibrage:
     * Oversampling ciblé
     * Weighted Loss
     * Focal Loss
   - Métriques adaptées: ROC, Recall, F1-score

4. **Visualisation alertes**
   - Interface web pour prédictions événements extrêmes
   - Système d'alertes

### Ce que le projet fait ACTUELLEMENT:

1. ❌ **Prédiction température uniquement (régression)**
   - RMSE 0.16°C avec Linear Regression
   - Aucune classification événements extrêmes
   - Aucune colonne "canicule", "vague_froid", etc.

2. ❌ **Pas d'ontologie climatique**
   - Aucun graphe de connaissances
   - Aucun moteur de règles
   - Aucune formalisation des seuils

3. ❌ **Pas de traitement spécifique des extrêmes**
   - Outliers conservés mais traités comme normaux
   - Aucun Oversampling
   - Aucun Weighted Loss ou Focal Loss
   - Pas de F1-score ou ROC

4. ❌ **Pas d'interface web**
   - Aucune technologie JavaScript
   - Pas de React/Vue/Angular
   - Pas d'API Node.js

---

## 🔴 PROBLEMES CRITIQUES

### 1. Mauvaise compréhension de l'objectif

**Ce qui a été fait**: Prédiction continue de température (regression)
**Ce qui était demandé**: Classification/détection d'événements extrêmes

**Exemple**:
```python
# Actuel (FAUX)
model.predict(X) → 28.5°C  # Prédiction température

# Attendu (CORRECT)
model.predict(X) → {
    "temperature": 28.5,
    "is_heatwave": False,
    "is_cold_wave": False,
    "extreme_event": None,
    "alert_level": 0
}
```

### 2. Architecture inadaptée

**Actuel**:
- Linear Regression: 0.16°C RMSE ✅ (excellent pour regression)
- LSTM/CNN-LSTM: 6-11°C RMSE ❌ (mauvais pour regression)

**Problème**: Les modèles DL échouent car le problème est MAL POSÉ

**Solution attendue**:
- LSTM/GRU pour **classification** événements extrêmes
- Weighted Loss pour gérer classe minoritaire (événements rares)
- Focal Loss pour se concentrer sur cas difficiles
- Métriques: Recall, F1-score, ROC-AUC (pas RMSE!)

### 3. Pas de traitement des événements rares

**Données actuelles** (train.parquet):
- Canicules (T ≥ 33°C): 3.47% des données (~25,000 heures)
- Grand froid (T ≤ -10°C): 0.90% des données (~6,500 heures)

**Problème**: Ces classes sont MINORITAIRES → DL prédit toujours "normal"

**Solutions manquantes**:
```python
# 1. Oversampling ciblé
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y_extreme_events)

# 2. Weighted Loss
class_weights = compute_class_weight('balanced', classes=[0,1,2], y=y)
model.compile(loss=weighted_categorical_crossentropy(class_weights))

# 3. Focal Loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        return -alpha * (1 - y_pred)**gamma * y_true * K.log(y_pred)
    return loss
```

### 4. Ontologie climatique absente

**Attendu** (exemples):
```python
# Règles météo
rules = {
    "canicule": {
        "condition": "temperature > 33 AND duration >= 3_days",
        "severity": "high" if T > 40 else "medium",
        "alert": True
    },
    "vague_froid": {
        "condition": "temperature < -5 AND duration >= 3_days",
        "severity": "high" if T < -15 else "medium",
        "alert": True
    }
}

# Ontologie OWL/RDF
Canicule subClassOf EventExtreme
Canicule has_threshold "33°C"
Canicule has_duration "3 days"
Canicule influences_by Humidity, Wind
```

**Actuel**: RIEN

---

## 📊 COMPARAISON: Actuel vs Attendu

| Aspect | Actuel | Attendu | Gap |
|--------|--------|---------|-----|
| **Objectif** | Prédiction température (régression) | Détection événements extrêmes (classification) | ❌ CRITIQUE |
| **Target** | Temperature continue | Classe: Normal/Canicule/Froid/Sécheresse | ❌ CRITIQUE |
| **Modèle DL** | LSTM régression (RMSE 6-11°C) | LSTM classification (F1-score, Recall) | ❌ CRITIQUE |
| **Traitement extrêmes** | Aucun | Oversampling, Weighted Loss, Focal Loss | ❌ MANQUANT |
| **Ontologie** | Aucune | Graphe connaissances + moteur règles | ❌ MANQUANT |
| **Métriques** | RMSE, MAE, R² | F1-score, Recall, Precision, ROC-AUC | ❌ INADAPTÉ |
| **Features** | 62 features engineered | Features + règles ontologie | ⚠️ PARTIEL |
| **Interface Web** | Aucune | React/Vue + Node.js API | ❌ MANQUANT |
| **Alertes** | Aucune | Système d'alertes automatique | ❌ MANQUANT |
| **Big Data** | Parquet files (local) | Hadoop/Spark distribué | ⚠️ PARTIEL |

---

## 🎯 CE QU'IL FAUT FAIRE MAINTENANT

### Phase 1: Reformulation du problème (URGENT)

**1. Créer la variable target "extreme_event"**
```python
def classify_extreme_events(df):
    """
    Classifie chaque observation en:
    0 = Normal
    1 = Canicule (T >= 33°C)
    2 = Vague froid (T <= -5°C)
    3 = Sécheresse (si pluie disponible)
    """
    conditions = [
        (df['temperature'] >= 33),  # Canicule
        (df['temperature'] <= -5),  # Vague froid
    ]
    choices = [1, 2]  # Labels
    df['extreme_event'] = np.select(conditions, choices, default=0)
    return df
```

**2. Ajouter détection de durée**
```python
def detect_heatwave(df):
    """Canicule = T >= 33°C pendant >= 3 jours consécutifs"""
    hot_days = df['temperature'] >= 33
    # Compter jours consécutifs
    consecutive = hot_days.groupby((hot_days != hot_days.shift()).cumsum()).cumcount() + 1
    df['is_heatwave'] = (hot_days & (consecutive >= 72))  # 72 heures = 3 jours
    return df
```

### Phase 2: Architecture Deep Learning adaptée

**1. Modèle de classification**
```python
# LSTM pour séquences temporelles
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes: Normal, Canicule, Froid, Sécheresse
])

# Focal Loss pour gérer classes déséquilibrées
model.compile(
    optimizer='adam',
    loss=focal_loss(alpha=0.25, gamma=2.0),
    metrics=['accuracy', 'Precision', 'Recall']
)
```

**2. Traitement déséquilibre**
```python
# Weighted Loss
class_counts = df['extreme_event'].value_counts()
class_weights = {
    0: 1.0,  # Normal (70%)
    1: 10.0,  # Canicule (3%)
    2: 20.0,  # Froid (0.9%)
    3: 15.0   # Sécheresse (rare)
}

# Ou SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### Phase 3: Ontologie climatique

**1. Définir règles**
```python
ontology_rules = {
    "Canicule": {
        "seuil_temperature": 33,
        "duree_min": 3,  # jours
        "variables_influentes": ["humidity", "wind_speed"],
        "severite": {
            "moderate": (33, 37),
            "severe": (37, 42),
            "extreme": (42, float('inf'))
        }
    },
    "VagueFroid": {
        "seuil_temperature": -5,
        "duree_min": 3,
        "variables_influentes": ["wind_chill", "pressure"],
        "severite": {
            "moderate": (-5, -10),
            "severe": (-10, -20),
            "extreme": (float('-inf'), -20)
        }
    }
}
```

**2. Moteur d'inférence**
```python
def infer_alert(predictions, ontology):
    """
    Génère alertes basées sur prédictions + règles ontologie
    """
    alerts = []
    for pred in predictions:
        if pred['temperature'] > ontology['Canicule']['seuil_temperature']:
            severity = get_severity(pred['temperature'], ontology['Canicule'])
            alerts.append({
                "type": "Canicule",
                "severity": severity,
                "temperature": pred['temperature'],
                "alert_level": 3 if severity == "extreme" else 2
            })
    return alerts
```

### Phase 4: Interface Web + API

**1. API Node.js**
```javascript
// server.js
const express = require('express');
const app = express();

app.post('/predict', async (req, res) => {
    const data = req.body;
    // Appel modèle Python
    const prediction = await predictExtremeEvents(data);
    // Inférence ontologie
    const alerts = await inferAlerts(prediction);
    res.json({ prediction, alerts });
});
```

**2. Frontend React**
```jsx
// Dashboard.jsx
function ExtremeEventsMonitor() {
    const [predictions, setPredictions] = useState([]);
    const [alerts, setAlerts] = useState([]);
    
    return (
        <div>
            <AlertPanel alerts={alerts} />
            <PredictionChart predictions={predictions} />
            <HeatmapVis events={predictions} />
        </div>
    );
}
```

---

## 📈 METRIQUES ATTENDUES

### Actuelles (FAUX pour ce projet):
- RMSE: 0.16°C ❌ (métrique régression)
- MAE: 0.12°C ❌ 
- R²: 0.9998 ❌

### Attendues (CORRECT):
- **F1-score** (balance Precision/Recall): > 0.85 pour événements extrêmes
- **Recall** (détection): > 0.90 (ne pas manquer événements critiques!)
- **Precision**: > 0.80 (éviter fausses alertes)
- **ROC-AUC**: > 0.95 (discrimination classes)
- **Confusion Matrix**: voir vrais/faux positifs par classe

**Exemple résultats attendus**:
```
Classification Report:

                Precision  Recall  F1-score  Support
Normal              0.96    0.98      0.97   500000
Canicule            0.85    0.91      0.88    25000
Vague Froid         0.88    0.87      0.87     6500
Sécheresse          0.82    0.78      0.80    15000

Accuracy: 0.95
Macro avg: 0.88  0.89  0.88
Weighted avg: 0.95  0.95  0.95
```

---

## 🚀 PLAN D'ACTION URGENT

### Semaine 1-2: REFORMULER LE PROBLEME
1. Créer colonnes classification (`extreme_event`, `is_heatwave`, etc.)
2. Implémenter règles détection durée (3 jours consécutifs)
3. Analyser distribution classes (déséquilibre)
4. Créer dataset étiqueté pour classification

### Semaine 3-4: DEEP LEARNING CLASSIFICATION
1. Architecture LSTM/GRU pour classification multi-classe
2. Implémenter Focal Loss ou Weighted Loss
3. Oversampling SMOTE si nécessaire
4. Entraîner avec métriques F1-score, Recall, ROC
5. Comparer LSTM vs GRU vs Transformer

### Semaine 5: ONTOLOGIE + INTERFACE
1. Définir règles ontologie climatique (JSON/OWL)
2. Moteur d'inférence pour alertes
3. API Node.js pour prédictions
4. Interface React avec visualisations

### Semaine 6: FINALISATION
1. Tests intégration
2. Documentation
3. Présentation finale

---

## 💡 CONCLUSION

### État actuel du projet:
**Hors-sujet par rapport au cahier des charges**

Le projet actuel prédit la température (régression), alors qu'il devrait **détecter et prédire des événements extrêmes** (classification).

### Actions immédiates:
1. ✅ **Reformuler le problème**: Regression → Classification événements extrêmes
2. ✅ **Créer target**: Ajouter colonne `extreme_event` (Normal/Canicule/Froid/etc.)
3. ✅ **Architecture DL**: LSTM classification avec Weighted Loss
4. ✅ **Ontologie**: Définir règles et moteur d'inférence
5. ✅ **Interface Web**: React + Node.js + API

### Temps estimé pour correction:
- **Reformulation + nouveau dataset**: 3-4 jours
- **Modèle DL classification**: 5-7 jours
- **Ontologie + moteur règles**: 2-3 jours
- **Interface Web**: 4-5 jours
- **Total**: ~3 semaines

### Note importante:
Le travail actuel (features engineering, preprocessing, Linear Regression) n'est PAS perdu:
- Features peuvent être réutilisées pour classification
- Preprocessing pipeline OK
- Infrastructure code réutilisable

Mais l'objectif principal doit CHANGER: de "prédire température" à "détecter événements extrêmes".
