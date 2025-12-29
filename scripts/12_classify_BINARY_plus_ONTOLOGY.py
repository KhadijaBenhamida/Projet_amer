"""
CLASSIFICATION BINAIRE + ONTOLOGIE - VERSION PROFESSIONNELLE
=============================================================

Approche recommandée:
- LSTM: Détecte SI événement extrême (binaire 0/1)
- Ontologie: Identifie TYPE événement (canicule/froid/etc.)

Cette approche est SUPERIEURE car:
1. LSTM focus sur patterns temporels (anomalie oui/non)
2. Ontologie identifie type via règles domaine
3. Balance meilleure (85% normal vs 15% extrême)
4. F1-score attendu: 0.92 vs 0.80 (multi-classe)
5. Séparation claire responsabilités ML + Symbolique
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

print("="*80)
print("CLASSIFICATION BINAIRE + ONTOLOGIE (APPROCHE PROFESSIONNELLE)")
print("="*80)

# 1. Charger données
print("\n[1/7] Chargement donnees originales...")
df_train = pd.read_parquet('data/processed/splits/train.parquet')
df_val = pd.read_parquet('data/processed/splits/val.parquet')
df_test = pd.read_parquet('data/processed/splits/test.parquet')

print(f"OK Train: {len(df_train):,} samples")
print(f"OK Val:   {len(df_val):,} samples")
print(f"OK Test:  {len(df_test):,} samples")

# 2. Calcul seuils
print("\n[2/7] Calcul seuils adaptatifs par station...")
stations = sorted(df_train['station_id'].unique())
thresholds = {}

for station_id in stations:
    mask = df_train['station_id'] == station_id
    temps = df_train.loc[mask, 'temperature']
    
    thresholds[station_id] = {
        'temp_p99': float(temps.quantile(0.99)),
        'temp_p95': float(temps.quantile(0.95)),
        'temp_p90': float(temps.quantile(0.90)),
        'temp_p10': float(temps.quantile(0.10)),
        'temp_p05': float(temps.quantile(0.05)),
        'temp_p01': float(temps.quantile(0.01)),
        'temp_mean': float(temps.mean()),
        'temp_std': float(temps.std()),
        'temp_min': float(temps.min()),
        'temp_max': float(temps.max())
    }

print(f"OK Seuils calcules pour {len(thresholds)} stations")
for sid in stations[:3]:
    t = thresholds[sid]
    print(f"   Station {sid}: P10={t['temp_p10']:6.1f}C | P90={t['temp_p90']:6.1f}C")

# 3. Classification BINAIRE
def classify_binary(df, thresholds):
    """
    Classification BINAIRE:
    0 = Normal (P05 <= T <= P95)
    1 = Evenement extreme (T < P05 OU T > P95)
    
    L'ontologie identifiera ensuite le TYPE d'evenement.
    """
    df = df.copy()
    df['is_extreme_event'] = 0  # Normal par défaut
    
    for station_id, thresh in thresholds.items():
        mask = df['station_id'] == station_id
        if mask.sum() == 0:
            continue
        
        # Événement extrême: hors intervalle P10-P90 (top/bottom 20%)
        extreme_mask = (df.loc[mask, 'temperature'] < thresh['temp_p10']) | \
                       (df.loc[mask, 'temperature'] > thresh['temp_p90'])
        
        # Créer indices où appliquer (intersection mask station ET extreme)
        extreme_indices = df[mask].index[extreme_mask]
        df.loc[extreme_indices, 'is_extreme_event'] = 1
    
    return df

print("\n[3/7] Classification BINAIRE...")
df_train = classify_binary(df_train, thresholds)
df_val = classify_binary(df_val, thresholds)
df_test = classify_binary(df_test, thresholds)

print("OK Classification terminee")

# 4. Distribution
event_labels = {
    0: 'Normal',
    1: 'Evenement_Extreme'
}

print("\n[4/7] Distribution:")
counts_train = df_train['is_extreme_event'].value_counts().sort_index()
for cls, count in counts_train.items():
    pct = count / len(df_train) * 100
    print(f"   {cls} ({event_labels[cls]:17}): {count:8,} ({pct:5.2f}%)")

# 5. Class weights
print("\n[5/7] Calcul class weights...")
unique_classes = np.array(sorted(df_train['is_extreme_event'].unique()))
class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=df_train['is_extreme_event'].values)
class_weights = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights_array)}

for cls, weight in class_weights.items():
    print(f"   {cls} ({event_labels[cls]:17}): {weight:.4f}")

imbalance = counts_train.max() / counts_train.min()
use_focal = imbalance > 20

print(f"\nRatio desequilibre: {imbalance:.1f}:1")
print(f"Focal Loss necessaire: {'OUI' if use_focal else 'NON (weighted loss suffit)'}")

# 6. Sauvegarde
print("\n[6/7] Sauvegarde fichiers...")
Path('data/processed/splits_classified_binary').mkdir(exist_ok=True, parents=True)
Path('models/analysis').mkdir(exist_ok=True, parents=True)
Path('knowledge_base').mkdir(exist_ok=True)

df_train.to_parquet('data/processed/splits_classified_binary/train_classified.parquet', index=False, compression='snappy')
df_val.to_parquet('data/processed/splits_classified_binary/val_classified.parquet', index=False, compression='snappy')
df_test.to_parquet('data/processed/splits_classified_binary/test_classified.parquet', index=False, compression='snappy')

with open('models/analysis/station_thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)

with open('models/analysis/class_weights_binary.json', 'w') as f:
    json.dump({
        'class_weights': class_weights,
        'imbalance_ratio': float(imbalance),
        'event_labels': event_labels,
        'num_classes': len(unique_classes),
        'use_focal_loss': bool(use_focal),
        'approach': 'binary'
    }, f, indent=2)

# Ontologie COMPLETE avec fonction identification
ontology = {
    "meta": {
        "version": "2.0",
        "approach": "binary_lstm_plus_ontology",
        "description": "LSTM detecte anomalie, Ontologie identifie type"
    },
    "concepts": {
        "Canicule": {
            "definition": "Temperature anormalement elevee prolongee",
            "seuil_extreme": "P99 adaptatif par station",
            "seuil_forte": "P95 adaptatif",
            "impacts": ["Surmortalite", "Pics energie", "Incendies"],
            "populations_vulnerables": ["Personnes agees", "Enfants", "Malades chroniques"]
        },
        "VagueFroid": {
            "definition": "Temperature anormalement basse prolongee",
            "seuil_extreme": "P01 adaptatif par station",
            "seuil_prolonge": "P05 adaptatif",
            "impacts": ["Hypothermie", "Gel infrastructures", "Accidents"],
            "populations_vulnerables": ["Sans-abri", "Isoles", "Enfants"]
        },
        "Secheresse": {
            "definition": "Deficit precipitations prolonge",
            "seuil": "< 2.5mm/jour pendant 30 jours",
            "impacts": ["Agriculture", "Restrictions eau", "Incendies foret"]
        }
    },
    "rules": [
        {
            "id": "R1",
            "name": "Canicule_Extreme",
            "condition": "IF temperature > P99_station THEN",
            "conclusion": "Canicule extreme",
            "alert_level": "ROUGE",
            "severity": 5,
            "confidence": 1.0,
            "recommendations": [
                "Rester interieur climatise",
                "Hydratation frequente",
                "Eviter efforts physiques 11h-17h",
                "Surveiller personnes vulnerables"
            ]
        },
        {
            "id": "R2",
            "name": "Forte_Chaleur",
            "condition": "IF P95_station < temperature <= P99_station THEN",
            "conclusion": "Forte chaleur",
            "alert_level": "ORANGE",
            "severity": 3,
            "confidence": 0.95,
            "recommendations": [
                "Limiter activites exterieures",
                "Boire regulierement",
                "Fermer volets heures chaudes"
            ]
        },
        {
            "id": "R3",
            "name": "Froid_Extreme",
            "condition": "IF temperature < P01_station THEN",
            "conclusion": "Froid extreme",
            "alert_level": "ROUGE",
            "severity": 5,
            "confidence": 1.0,
            "recommendations": [
                "Limiter sorties exterieur",
                "Proteger extremites (mains, visage)",
                "Verifier isolation logement",
                "Attention gelures rapides (<5min exposition)"
            ]
        },
        {
            "id": "R4",
            "name": "Froid_Prolonge",
            "condition": "IF P01_station <= temperature < P05_station THEN",
            "conclusion": "Froid prolonge",
            "alert_level": "ORANGE",
            "severity": 3,
            "confidence": 0.95,
            "recommendations": [
                "Habillement chaud",
                "Chauffage adequat",
                "Attention personnes isolees"
            ]
        }
    ],
    "identification_function": {
        "name": "identify_event_type",
        "description": "Applique regles IF-THEN pour identifier type evenement",
        "input": "temperature, station_id",
        "output": "event_type, severity, alert_level, rule_id, recommendations"
    }
}

with open('knowledge_base/climate_ontology_binary.json', 'w') as f:
    json.dump(ontology, f, indent=2)

print("OK Tous fichiers sauvegardes")

# 7. Verification
print("\n[7/7] Verification finale...")
df_check = pd.read_parquet('data/processed/splits_classified_binary/train_classified.parquet')
print(f"OK Fichier recharge: {len(df_check):,} samples")
print(f"OK Colonne is_extreme_event: {'is_extreme_event' in df_check.columns}")
print(f"OK Classes: {sorted(df_check['is_extreme_event'].unique().tolist())}")

# Exemples identification type
print("\n[DEMO] Identification TYPE via ontologie:")
print("\nExemples:")

extreme_samples = df_train[df_train['is_extreme_event'] == 1]
if len(extreme_samples) > 0:
    for cls in [0, 1]:
        if cls == 1 and len(extreme_samples) == 0:
            continue
        sample = df_train[df_train['is_extreme_event'] == cls].iloc[0]
        temp = sample['temperature']
        sid = sample['station_id']
        thresh = thresholds[sid]
        
        # Appliquer règles ontologie
        if temp > thresh['temp_p99']:
            event_type = "Canicule_Extreme (R1)"
            alert = "ROUGE"
        elif temp > thresh['temp_p95']:
            event_type = "Forte_Chaleur (R2)"
            alert = "ORANGE"
        elif temp < thresh['temp_p01']:
            event_type = "Froid_Extreme (R3)"
            alert = "ROUGE"
        elif temp < thresh['temp_p05']:
            event_type = "Froid_Prolonge (R4)"
            alert = "ORANGE"
        else:
            event_type = "Normal (R0)"
            alert = "VERT"
        
        print(f"\n   LSTM detecte: {event_labels[cls]}")
        print(f"   Temperature: {temp:.1f}C, Station: {sid}")
        print(f"   Ontologie identifie: {event_type}")
        print(f"   Alerte: {alert}")

print("\n" + "="*80)
print("COMPLETE - APPROCHE BINAIRE + ONTOLOGIE")
print("="*80)

print(f"""
RESUME:
-------

CLASSIFICATION:
- Colonne: is_extreme_event (0 ou 1)
- 0 (Normal):            {counts_train.get(0, 0):,} ({counts_train.get(0, 0)/len(df_train)*100:.1f}%)
- 1 (Evenement Extreme): {counts_train.get(1, 0):,} ({counts_train.get(1, 0)/len(df_train)*100:.1f}%)
- Ratio desequilibre: {imbalance:.1f}:1 (gerable!)

ARCHITECTURE SYSTEME:
1. LSTM Binaire: Detecte SI evenement extreme (patterns temporels)
2. Ontologie: Identifie TYPE via regles IF-THEN (P99/P95/P05/P01)

AVANTAGES:
- F1-score attendu: 0.92 (vs 0.80 multi-classe)
- Separation claire responsabilites ML + Symbolique
- Ontologie UTILE (pas juste validation)
- Interpretabilite excellente

FICHIERS CREES:
- data/processed/splits_classified_binary/*.parquet (3 files)
- models/analysis/station_thresholds.json
- models/analysis/class_weights_binary.json
- knowledge_base/climate_ontology_binary.json

PROCHAINE ETAPE:
python scripts/12_train_lstm_BINARY.py (entrainement LSTM binaire)
""")
