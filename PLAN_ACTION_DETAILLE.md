# 🎯 PROJET REFORMULÉ: Prédiction Événements Climatiques Extrêmes

**Date**: 28 Décembre 2024  
**Status**: ✅ Phase 1 TERMINÉE (Classification + Ontologie)  
**Prochaine**: Phase 2 - Deep Learning LSTM Classification

---

## 📋 RÉSUMÉ EXÉCUTIF

### Problème Identifié
Le projet initial était **INCORRECT** :
- ❌ Objectif: Prédiction température (régression)
- ❌ Target: `temperature` (valeur continue)
- ❌ Métriques: RMSE, MAE
- ❌ Résultats: Linear Regression 0.16°C, LSTM 6-11°C

### Solution Implémentée
Reformulation complète conforme au **cahier des charges** :
- ✅ Objectif: **Classification événements extrêmes**
- ✅ Target: `extreme_event` (0=Normal, 1=Canicule, 2=Froid)
- ✅ Métriques: F1-score, Recall, Precision, ROC-AUC
- ✅ Ontologie climatique + moteur d'inférence
- ✅ Base pour interface Web + API

---

## 🎉 RÉALISATIONS (Phase 1 - Complétée)

### 1. Classification Événements Extrêmes ✅

**Script**: `scripts/01_create_extreme_events_classification_v2.py`

**Méthode**:
- Rolling mean 48h pour lisser variations
- Seuils basés standards météorologiques:
  * Canicule: moyenne 48h >= 28°C
  * Froid prolongé: moyenne 48h <= 2°C

**Résultats**:
```
Distribution Train (725,176 samples):
- Normal:         594,618 (82.00%)
- Canicule:        71,868 (9.91%)
- Froid prolongé:  58,690 (8.09%)

Class Weights (balanced):
- Normal:         0.4065
- Canicule:       3.3635
- Froid prolongé: 4.1187

Ratio déséquilibre: 10.1:1 (modéré)
```

**Stratégie déséquilibre**: 
- ✅ Weighted Loss (suffit pour ratio <20:1)
- ⚠️ Focal Loss recommandé pour améliorer Recall événements rares

**Fichiers créés**:
- `data/processed/splits_classified/train_classified.parquet` (725k samples)
- `data/processed/splits_classified/val_classified.parquet` (208k samples)
- `data/processed/splits_classified/test_classified.parquet` (107k samples)
- `models/analysis/class_weights.json`
- `models/analysis/class_distribution.png`
- `models/analysis/temperature_by_class.png`
- `models/analysis/events_timeline.png`

---

### 2. Ontologie Climatique + Moteur d'Inférence ✅

**Script**: `knowledge_base/climate_ontology.py`

**Composants**:

#### A. Ontologie (Knowledge Graph)
- **4 concepts principaux**: Canicule, VagueFroid, Sécheresse, PrécipitationIntense
- **Propriétés**: Seuils température, durée minimale, facteurs aggravants
- **Impacts**: Santé publique, infrastructures, agriculture
- **Populations vulnérables**: Personnes âgées, enfants, sans-abri

#### B. Base de règles (8 règles)
Format: `IF conditions THEN conclusion WITH confidence`

**Règles Canicule**:
1. Extrême: `temp_48h >= 42°C` → ROUGE (confidence 1.0)
2. Sévère: `temp_48h 37-42°C` → ORANGE (confidence 0.95)
3. Modérée: `temp_48h 33-37°C` → JAUNE (confidence 0.90)
4. Faible: `temp_48h 28-33°C` → VERT (confidence 0.80)

**Règles Froid**:
1. Extrême: `temp_48h <= -20°C` → ROUGE (confidence 1.0)
2. Sévère: `temp_48h -10 to -20°C` → ORANGE (confidence 0.95)
3. Modéré: `temp_48h -5 to -10°C` → JAUNE (confidence 0.90)
4. Faible: `temp_48h 0-2°C` → VERT (confidence 0.80)

#### C. Moteur d'Inférence
- **Classe**: `InferenceEngine`
- **Méthode**: `infer(data)` → Liste alertes
- **Fonctions**:
  * Évaluation règles sur données temps réel
  * Génération alertes multi-niveaux (VERT/JAUNE/ORANGE/ROUGE)
  * Recommandations personnalisées par type événement
  * Traitement batch: `infer_batch(dataframe)`

**Tests Réels** (échantillon 1000 lignes):
```
Événements inférés:
- NORMAL:      669 (66.9%)
- VAGUE_FROID: 331 (33.1%)

Niveaux alerte:
- VERT:   897 (89.7%)
- JAUNE:   98 (9.8%)
- ORANGE:   5 (0.5%)
```

**Fichiers créés**:
- `knowledge_base/climate_ontology.json` (définitions concepts)
- `knowledge_base/climate_rules.json` (règles inférence)
- `knowledge_base/inference_sample.parquet` (échantillon testé)

---

## 🔄 COMPARAISON: Avant vs Après

| Aspect | ❌ Avant (Incorrect) | ✅ Après (Conforme) |
|--------|---------------------|---------------------|
| **Objectif** | Régression température | Classification événements |
| **Target** | `temperature` (continue) | `extreme_event` (0/1/2) |
| **Problème** | Prédire "28.5°C" | Détecter "Canicule oui/non" |
| **Métriques** | RMSE, MAE, R² | F1-score, Recall, ROC-AUC |
| **Meilleur modèle** | Linear Reg 0.16°C | À venir: LSTM classification |
| **DL "échec"** | LSTM 6-11°C (wrong problem!) | LSTM classification avec Focal Loss |
| **Déséquilibre** | Non traité | Weighted Loss + Class weights |
| **Ontologie** | ❌ Absente | ✅ 4 concepts, 8 règles |
| **Inférence** | ❌ Aucune | ✅ Moteur règles automatique |
| **Interface** | ❌ Aucune | 🔜 React + Node.js API |
| **Alertes** | ❌ Aucune | ✅ 4 niveaux (VERT/JAUNE/ORANGE/ROUGE) |

---

## 📊 STATISTIQUES CLÉS

### Distribution Données
```
Total samples: 1,041,268 (2015-2021, 8 stations)
├─ Train: 725,176 (69.6%)
├─ Val:   208,218 (20.0%)
└─ Test:  107,874 (10.4%)

Période: 2015-01-01 à 2021-12-31 (7 ans)
Granularité: 1 heure
```

### Événements Détectés
```
Canicules (Train):
- Total: 71,868 échantillons (9.91%)
- Température moyenne: 30.9°C
- Range: 16.7°C à 48.3°C
- Pics: Miami (40-48°C), Phoenix

Froids prolongés (Train):
- Total: 58,690 échantillons (8.09%)
- Température moyenne: -2.2°C
- Range: -30.6°C à 18.3°C
- Pics: Denver (-30°C), Boston

Ratio déséquilibre: 10.1:1 (modéré)
```

### Features (72 colonnes)
```
Originales (11):
- station_id, year, month, day, hour, minute
- temperature, dewpoint, wind_direction, wind_speed, sea_level_pressure

Engineered (62):
- Lags: 21 features (1h à 168h)
- Rolling stats: 16 features (6h, 12h, 24h windows)
- Temporal: 8 cyclical (sin/cos month, day, hour)
- Interactions: 5 features
- Différences: 1 feature

Nouvelles (4):
- extreme_event: Classification (0/1/2)
- temp_rolling_48h: Moyenne mobile 48h
- is_hot: Booléen température >= 30°C
- is_cold: Booléen température <= 0°C
```

---

## 🚀 PLAN D'ACTION DÉTAILLÉ

### ✅ **PHASE 1: FONDATIONS (TERMINÉE)**

**Durée**: 2 jours  
**Status**: ✅ 100% COMPLÉTÉ

#### 1.1 Classification Événements ✅
- [x] Créer target `extreme_event` (0/1/2)
- [x] Détection avec rolling mean 48h
- [x] Calculer class weights (0.41 / 3.36 / 4.12)
- [x] Sauvegarder datasets classifiés (725k + 208k + 107k)
- [x] Visualisations (distribution, température par classe, timeline)

#### 1.2 Ontologie + Inférence ✅
- [x] Définir ontologie 4 concepts (Canicule, Froid, Sécheresse, Pluie)
- [x] Créer 8 règles inférence (4 canicule + 4 froid)
- [x] Implémenter moteur inférence (`InferenceEngine`)
- [x] Tester sur données réelles (1000 samples)
- [x] Générer recommandations automatiques

**Livrables Phase 1**:
- ✅ Datasets classifiés (3 fichiers .parquet)
- ✅ Ontologie JSON (concepts + propriétés)
- ✅ Règles JSON (8 règles formalisées)
- ✅ Moteur inférence Python (classe `InferenceEngine`)
- ✅ Visualisations (3 PNG)
- ✅ Class weights JSON

---

### 🔜 **PHASE 2: DEEP LEARNING (EN COURS)**

**Durée**: 4-5 jours  
**Status**: 🟡 PRÊT À DÉMARRER  
**Priorité**: ⭐⭐⭐⭐⭐ HAUTE

#### 2.1 Architecture LSTM Classification

**Fichier**: `models/lstm_classifier.py` (déjà créé)

**Composants**:
- [x] Focal Loss implementation (alpha=0.25, gamma=2.0)
- [x] Weighted Loss wrapper (avec class_weights)
- [x] Architecture LSTM bidirectionnelle (128 → 64 units)
- [x] Architecture GRU alternative (plus rapide)
- [x] Fonction création séquences (`create_sequences`, 72h window)
- [x] Fonction évaluation complète (`evaluate_classifier`)

**À exécuter**:
```bash
# Entraîner 3 modèles en parallèle:
python models/lstm_classifier.py

# Modèles créés:
# 1. LSTM + Focal Loss     → models/lstm_focal_loss.keras
# 2. LSTM + Weighted Loss  → models/lstm_weighted_loss.keras
# 3. LSTM + CrossEntropy   → models/lstm_baseline.keras
```

**Objectifs**:
- F1-score Macro >= 0.80
- F1-score Canicule >= 0.85
- F1-score Froid >= 0.85
- Recall événements >= 0.90 (priorité: ne pas manquer événements)
- ROC-AUC >= 0.85

**Hyperparamètres**:
```python
sequence_length = 72  # 3 jours historique
batch_size = 256
epochs = 100 (avec EarlyStopping patience=15)
learning_rate = 0.001 (avec ReduceLROnPlateau)
```

#### 2.2 Entraînement et Évaluation

**Étapes**:
1. Charger datasets classifiés
2. Créer séquences temporelles (X: 72h, y: label)
3. Entraîner 3 modèles (Focal / Weighted / Baseline)
4. Comparer performances (F1, Recall, ROC-AUC)
5. Sélectionner meilleur modèle
6. Sauvegarder modèle final + poids

**Métriques à surveiller**:
- **F1-score**: Équilibre Precision/Recall
- **Recall**: Priorité #1 (ne pas manquer événements extrêmes)
- **Precision**: Éviter fausses alertes
- **ROC-AUC**: Performance globale classification
- **Confusion matrix**: Analyser erreurs

**Temps estimé**: 2-3 jours (incluant expérimentations)

---

### 🔜 **PHASE 3: API BACKEND (Node.js)**

**Durée**: 2-3 jours  
**Status**: 🔵 EN ATTENTE Phase 2  
**Priorité**: ⭐⭐⭐⭐ HAUTE

#### 3.1 Architecture API

**Stack**:
- Node.js + Express.js
- TensorFlow.js (chargement modèle Keras)
- REST API + WebSocket (alertes temps réel)

**Endpoints**:

```javascript
// Prédictions
POST /api/predict
Body: { 
  "station_id": "KMIA",
  "features": [temperature, dewpoint, ...],
  "sequence_length": 72
}
Response: {
  "prediction": {
    "event_type": "CANICULE",
    "probability": 0.87,
    "severity": "SEVERE",
    "alert_level": "ORANGE"
  },
  "inference": {
    "rule_triggered": "CANICULE_SEVERE",
    "confidence": 0.95,
    "recommendations": [...]
  }
}

// Alertes actives
GET /api/alerts
Response: {
  "alerts": [
    {
      "id": "alert_001",
      "station": "KMIA",
      "type": "CANICULE",
      "level": "ORANGE",
      "timestamp": "2024-12-28T15:30:00Z"
    }
  ]
}

// Historique prédictions
GET /api/history/:station_id?from=date&to=date

// WebSocket temps réel
WS /ws/alerts
```

#### 3.2 Intégration

**Composants**:
1. **Model Server**: Charger modèle LSTM Keras
2. **Inference Engine**: Importer moteur Python (via child_process ou API)
3. **Database**: PostgreSQL/MongoDB pour historique
4. **Cache**: Redis pour prédictions récentes
5. **Queue**: Bull/RabbitMQ pour traitement batch

**Fichiers à créer**:
```
backend/
├── server.js              # Express app
├── routes/
│   ├── predict.js         # Prédictions endpoint
│   ├── alerts.js          # Alertes endpoint
│   └── history.js         # Historique endpoint
├── services/
│   ├── modelService.js    # Chargement/prédiction LSTM
│   ├── inferenceService.js # Moteur règles
│   └── alertService.js    # Gestion alertes
├── models/
│   └── Alert.js           # Modèle DB alertes
├── utils/
│   ├── preprocessor.js    # Preprocessing features
│   └── validator.js       # Validation inputs
└── package.json
```

**Temps estimé**: 2-3 jours

---

### 🔜 **PHASE 4: FRONTEND (React)**

**Durée**: 3-4 jours  
**Status**: 🔵 EN ATTENTE Phase 3  
**Priorité**: ⭐⭐⭐ MOYENNE

#### 4.1 Interface Web

**Stack**:
- React 18 + TypeScript
- Tailwind CSS / Material-UI
- Recharts / D3.js (visualisations)
- Socket.io-client (WebSocket)
- React Query (API calls)

**Pages**:

1. **Dashboard Principal**
   - Carte interactive stations (avec alertes)
   - Timeline événements dernières 7 jours
   - Statistiques temps réel (nombre alertes actives)
   - Top alertes actives (cards avec niveau couleur)

2. **Prédictions**
   - Formulaire saisie données station
   - Prédiction instantanée (event + probabilité)
   - Recommandations affichées
   - Graphique confiance par classe

3. **Alertes**
   - Liste alertes actives (filtrable)
   - Historique alertes (timeline)
   - Notifications push (WebSocket)
   - Export PDF/CSV

4. **Visualisations**
   - Heatmap température par station
   - Graphiques distribution événements
   - Courbes température + prédictions
   - Matrice confusion modèle

5. **Admin**
   - Configuration seuils alertes
   - Gestion stations
   - Logs système

#### 4.2 Composants React

```tsx
// Composants principaux
<Dashboard />
  ├─ <StationMap />          // Carte stations avec pins
  ├─ <AlertsPanel />         // Alertes actives
  ├─ <EventsTimeline />      // Timeline événements
  └─ <StatsCards />          // Statistiques KPI

<PredictionForm />
  ├─ <StationSelector />     // Sélection station
  ├─ <FeaturesInput />       // Saisie features
  └─ <PredictionResult />    // Résultat + viz

<AlertsList />
  ├─ <AlertCard />           // Card alerte individuelle
  ├─ <AlertFilters />        // Filtres type/niveau
  └─ <AlertNotifications />  // Notifications WebSocket

<Visualizations />
  ├─ <TemperatureHeatmap />  // Heatmap température
  ├─ <EventsChart />         // Distribution événements
  └─ <ConfusionMatrix />     // Matrice confusion modèle
```

**Temps estimé**: 3-4 jours

---

### 🔜 **PHASE 5: TESTS & DOCUMENTATION**

**Durée**: 2-3 jours  
**Status**: 🔵 EN ATTENTE Phase 4  
**Priorité**: ⭐⭐ BASSE

#### 5.1 Tests

**Backend**:
- Tests unitaires (Jest) endpoints API
- Tests intégration model service
- Tests E2E (Postman/Supertest)

**Frontend**:
- Tests composants (React Testing Library)
- Tests intégration (Cypress)

**Modèle**:
- Validation test set final
- Tests edge cases (valeurs extrêmes)
- Benchmarks performances

#### 5.2 Documentation

**Technique**:
1. **README.md**
   - Installation
   - Configuration
   - Lancement (backend + frontend)

2. **ARCHITECTURE.md**
   - Diagrammes système
   - Flow données
   - Technologies utilisées

3. **API_DOCUMENTATION.md**
   - Endpoints détaillés
   - Exemples requêtes/réponses
   - Codes erreur

4. **MODEL_DOCUMENTATION.md**
   - Architecture LSTM
   - Hyperparamètres
   - Performances
   - Interprétation résultats

**Utilisateur**:
1. **GUIDE_UTILISATEUR.md**
   - Comment utiliser interface
   - Interprétation alertes
   - Actions recommandées

2. **FAQ.md**
   - Questions fréquentes

**Temps estimé**: 2-3 jours

---

## 📅 PLANNING GLOBAL

```
TOTAL: ~3 semaines (14-18 jours ouvrables)

├─ ✅ Phase 1: Fondations (2j) ─────────── TERMINÉE
│  ├─ Classification événements (1j)
│  └─ Ontologie + inférence (1j)
│
├─ 🟡 Phase 2: Deep Learning (4-5j) ────── EN COURS
│  ├─ Architecture LSTM (1j)
│  ├─ Entraînement modèles (2j)
│  └─ Évaluation + sélection (1-2j)
│
├─ 🔵 Phase 3: API Backend (2-3j) ─────── EN ATTENTE
│  ├─ Endpoints REST (1j)
│  ├─ Intégration modèle (1j)
│  └─ WebSocket alertes (1j)
│
├─ 🔵 Phase 4: Frontend React (3-4j) ──── EN ATTENTE
│  ├─ Dashboard + cartes (2j)
│  ├─ Prédictions + viz (1j)
│  └─ Alertes temps réel (1j)
│
└─ 🔵 Phase 5: Tests + Docs (2-3j) ───── EN ATTENTE
   ├─ Tests (1-2j)
   └─ Documentation (1j)
```

**Date début**: 27 Décembre 2024  
**Date fin estimée**: 15-20 Janvier 2025  
**Statut actuel**: Phase 1 ✅ | Phase 2 prête 🟡

---

## 🎯 OBJECTIFS FINAUX

### Techniques
- [x] Classification multi-classe (Normal/Canicule/Froid)
- [x] Ontologie 4 concepts + 8 règles
- [x] Moteur inférence automatique
- [ ] LSTM classification F1-score >= 0.80
- [ ] Recall événements >= 0.90
- [ ] API REST + WebSocket
- [ ] Interface Web réactive
- [ ] Alertes temps réel

### Livrables
- [ ] Code source (backend + frontend + notebooks)
- [ ] Modèles entraînés (.keras files)
- [ ] Ontologie + règles (JSON)
- [ ] API documentée (OpenAPI/Swagger)
- [ ] Interface Web déployable
- [ ] Documentation technique
- [ ] Guide utilisateur
- [ ] Rapport final
- [ ] Présentation PowerPoint

---

## 📈 PROCHAINES ACTIONS IMMÉDIATES

### 1. Entraîner LSTM Classification ⭐⭐⭐⭐⭐
```bash
python models/lstm_classifier.py
```
**Durée**: 2-3 heures (entraînement)  
**Objectif**: F1-score >= 0.80, Recall >= 0.90

### 2. Analyser Résultats
- Comparer 3 modèles (Focal/Weighted/Baseline)
- Identifier meilleur modèle
- Analyser confusion matrix
- Sauvegarder modèle final

### 3. Créer API Prototype
- Setup Express.js projet
- Endpoint `/predict` avec TensorFlow.js
- Intégration moteur inférence
- Tests Postman

### 4. Interface Web Minimale
- Dashboard basique React
- Formulaire prédiction
- Affichage résultat + alertes

---

## 💡 NOTES IMPORTANTES

### Différences Clés: Régression vs Classification

**Avant (Régression)**:
```python
# Input: [temperature, dewpoint, ...]
model.predict(X) → [28.5]  # Température prédite
loss = MSE(y_true=28.5, y_pred=28.3)
metric = RMSE = 0.16°C  ← ✅ Excellent pour régression
```

**Maintenant (Classification)**:
```python
# Input: [sequence 72h features]
model.predict(X) → [0.05, 0.87, 0.08]  # Probas classes
prediction = argmax → 1 (CANICULE)
loss = FocalLoss(y_true=[0,1,0], y_pred=[0.05,0.87,0.08])
metrics = {
  'f1_macro': 0.82,
  'recall_canicule': 0.91,  ← ⭐ Priorité
  'recall_froid': 0.88
}
```

### Pourquoi Focal Loss?

**Problème**: Classes déséquilibrées (82% Normal vs 10% Canicule vs 8% Froid)

**Solution Focal Loss**:
```python
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

# Exemples:
p_t = 0.99 (prédiction facile)    → FL = -0.25 * 0.01^2 * log(0.99) ≈ 0.00003
p_t = 0.60 (prédiction difficile) → FL = -0.25 * 0.40^2 * log(0.60) ≈ 0.02044

→ Focus sur exemples difficiles (événements rares)
```

**Avantages**:
- Augmente poids exemples mal classifiés
- Réduit poids exemples faciles (Normal)
- Améliore Recall classe minoritaire (événements extrêmes)

---

## 📚 RESSOURCES

### Documentation Créée
- [ANALYSE_CAHIER_DES_CHARGES.md](ANALYSE_CAHIER_DES_CHARGES.md) - Gap analysis détaillée
- Ce document (PLAN_ACTION_DETAILLE.md) - Roadmap complète

### Scripts Python
- `scripts/01_create_extreme_events_classification_v2.py` - Classification
- `knowledge_base/climate_ontology.py` - Ontologie + inférence
- `models/lstm_classifier.py` - Architecture LSTM (prêt)

### Visualisations
- `models/analysis/class_distribution.png` - Distribution classes
- `models/analysis/temperature_by_class.png` - Température par classe
- `models/analysis/events_timeline.png` - Timeline événements

### Données
- `data/processed/splits_classified/*.parquet` - Datasets classifiés
- `knowledge_base/*.json` - Ontologie + règles
- `models/analysis/class_weights.json` - Poids classes

---

## ✅ VALIDATION CAHIER DES CHARGES

| Exigence | Status | Implémentation |
|----------|--------|----------------|
| **Deep Learning séries temporelles** | 🟡 En cours | LSTM 128→64, séquences 72h |
| **Classification événements extrêmes** | ✅ OK | 3 classes (Normal/Canicule/Froid) |
| **Traitement déséquilibre classes** | ✅ OK | Weighted Loss + Focal Loss |
| **Métriques F1-score, Recall** | 🟡 En cours | Fonctions prêtes |
| **Ontologie climatique** | ✅ OK | 4 concepts, propriétés, impacts |
| **Règles IF-THEN** | ✅ OK | 8 règles (Canicule + Froid) |
| **Moteur inférence** | ✅ OK | Classe `InferenceEngine` |
| **Alertes automatiques** | ✅ OK | 4 niveaux (VERT/JAUNE/ORANGE/ROUGE) |
| **Recommandations** | ✅ OK | Générées par ontologie |
| **Interface JavaScript** | 🔵 À faire | React + Node.js API |
| **Visualisations** | ✅ Partiel | Graphiques stats (interface à faire) |
| **API temps réel** | 🔵 À faire | WebSocket alertes |

**Conformité globale**: 60% (Phase 1-2) → 100% après Phase 3-4

---

## 🎓 APPRENTISSAGES CLÉS

### 1. Importance de comprendre le problème
- ❌ 3 mois perdus sur mauvaise formulation (régression)
- ✅ 2 jours reformulation complète (classification)
- 📖 Leçon: TOUJOURS analyser cahier des charges en profondeur AVANT de coder

### 2. Deep Learning n'est pas magique
- LSTM "échoue" à 6-11°C RMSE sur régression → Normal, Linear Reg meilleur
- LSTM excellera sur classification → Bon problème, bon outil
- 📖 Leçon: Choisir algorithme adapté au problème

### 3. Ingénierie des connaissances est cruciale
- Ontologie structure domaine métier
- Règles explicites complètent prédictions DL
- Explainability: règles "IF-THEN" compréhensibles vs boîte noire DL
- 📖 Leçon: Combiner DL (prédictions) + règles (validation/alertes)

### 4. Déséquilibre classes doit être traité
- 82% Normal vs 10% Canicule vs 8% Froid
- Focal Loss focus sur événements rares (priorité business!)
- Recall > Precision pour événements extrêmes (ne pas manquer!)
- 📖 Leçon: Métriques business-driven (ici: Recall événements)

---

## 📞 CONTACT & SUPPORT

**Auteur**: System  
**Date**: 28 Décembre 2024  
**Version**: 1.0  

**Questions/Issues**:
- Phase 2 LSTM: Ajuster hyperparamètres si F1 < 0.80
- Phase 3 API: Intégration TensorFlow.js + Python inference
- Phase 4 Frontend: WebSocket temps réel

---

*Document vivant - Mis à jour au fur et à mesure de l'avancement du projet*
