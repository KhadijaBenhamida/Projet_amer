"""
VERIFICATION CLASSIFICATION - QU'EST-CE QUI A ETE FAIT?
========================================================

Ce script explique et vérifie la classification créée.
"""

import pandas as pd
import json
from pathlib import Path

print("="*80)
print("VERIFICATION: QU'EST-CE QUI A ETE FAIT DANS LA CLASSIFICATION?")
print("="*80)

# 1. Vérifier fichiers créés
print("\n📁 FICHIERS CREES:")
files_to_check = [
    'data/processed/splits_classified/train_classified.parquet',
    'data/processed/splits_classified/val_classified.parquet',
    'data/processed/splits_classified/test_classified.parquet',
    'models/analysis/station_thresholds.json',
    'models/analysis/class_weights.json',
    'knowledge_base/climate_ontology.json'
]

for filepath in files_to_check:
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size / 1024 / 1024  # MB
        print(f"   ✅ {filepath} ({size:.2f} MB)")
    else:
        print(f"   ❌ {filepath} (MANQUANT)")

# 2. Charger et expliquer
print("\n" + "="*80)
print("EXPLICATION: NOUVELLE COLONNE 'extreme_event'")
print("="*80)

try:
    df = pd.read_parquet('data/processed/splits_classified/train_classified.parquet')
    
    print("\n📊 STRUCTURE DATASET:")
    print(f"   Lignes: {len(df):,}")
    print(f"   Colonnes: {len(df.columns)}")
    
    if 'extreme_event' in df.columns:
        print("\n✅ COLONNE 'extreme_event' CREEE!")
        print("\n   Cette colonne contient la CLASSIFICATION des événements:")
        
        event_labels = {
            0: 'Normal',
            1: 'Canicule_Extreme',
            2: 'Forte_Chaleur',
            3: 'Froid_Extreme',
            4: 'Froid_Prolonge'
        }
        
        print("\n   📋 DISTRIBUTION:")
        counts = df['extreme_event'].value_counts().sort_index()
        for cls, count in counts.items():
            pct = count / len(df) * 100
            label = event_labels.get(cls, f'Classe_{cls}')
            print(f"      {cls} ({label:17}): {count:8,} samples ({pct:5.2f}%)")
        
        # Exemples
        print("\n   📝 EXEMPLES (premières lignes):")
        cols_to_show = ['datetime', 'station_id', 'temperature', 'extreme_event']
        print(df[cols_to_show].head(10).to_string(index=False))
        
        # Exemples par classe
        print("\n   🔍 EXEMPLES PAR CLASSE:")
        for cls in sorted(df['extreme_event'].unique()):
            label = event_labels.get(cls, f'Classe_{cls}')
            sample = df[df['extreme_event'] == cls][['temperature', 'station_id']].iloc[0]
            print(f"      {cls} ({label:17}): T={sample['temperature']:6.2f}°C, Station={sample['station_id']}")
    
    else:
        print("\n❌ COLONNE 'extreme_event' MANQUANTE!")
        print(f"   Colonnes présentes: {list(df.columns)[:10]}...")

except Exception as e:
    print(f"\n❌ ERREUR lecture dataset: {e}")

# 3. Expliquer seuils
print("\n" + "="*80)
print("EXPLICATION: SEUILS ADAPTATIFS PAR STATION")
print("="*80)

try:
    with open('models/analysis/station_thresholds.json') as f:
        thresholds = json.load(f)
    
    print("\n✅ SEUILS CALCULES POUR CHAQUE STATION:")
    print("\n   Les seuils sont ADAPTATIFS (percentiles):")
    print("   - P99 (top 1%) = Canicule extrême")
    print("   - P95 (top 5%) = Forte chaleur")
    print("   - P05 (bottom 5%) = Froid prolongé")
    print("   - P01 (bottom 1%) = Froid extrême")
    
    print("\n   📊 SEUILS PAR STATION:")
    for station_id, thresh in thresholds.items():
        print(f"\n   Station {station_id}:")
        print(f"      Canicule extrême: T > {thresh['temp_p99']:.1f}°C (P99)")
        print(f"      Forte chaleur:    {thresh['temp_p95']:.1f}°C < T ≤ {thresh['temp_p99']:.1f}°C")
        print(f"      Normal:           {thresh['temp_p05']:.1f}°C ≤ T ≤ {thresh['temp_p95']:.1f}°C")
        print(f"      Froid prolongé:   {thresh['temp_p01']:.1f}°C ≤ T < {thresh['temp_p05']:.1f}°C")
        print(f"      Froid extrême:    T < {thresh['temp_p01']:.1f}°C (P01)")

except Exception as e:
    print(f"\n❌ ERREUR lecture seuils: {e}")

# 4. Class weights
print("\n" + "="*80)
print("EXPLICATION: CLASS WEIGHTS (pour entraînement)")
print("="*80)

try:
    with open('models/analysis/class_weights.json') as f:
        weights_info = json.load(f)
    
    print("\n✅ CLASS WEIGHTS CALCULES:")
    print("   Ces poids compensent le déséquilibre des classes")
    print("   (événements extrêmes rares vs normal fréquent)")
    
    print("\n   ⚖️  POIDS PAR CLASSE:")
    for cls_str, weight in weights_info['class_weights'].items():
        label = weights_info['event_labels'].get(cls_str, f'Classe_{cls_str}')
        print(f"      {cls_str} ({label:17}): {weight:.4f}")
    
    print(f"\n   📊 Ratio déséquilibre: {weights_info['imbalance_ratio']:.1f}:1")
    print(f"   🎯 Focal Loss utilisé: {'OUI' if weights_info['use_focal_loss'] else 'NON'}")
    
    if weights_info['use_focal_loss']:
        print("\n   ℹ️  Focal Loss (alpha=0.25, gamma=2.0) sera utilisé")
        print("      car déséquilibre > 20:1")
        print("      → Focus sur événements rares difficiles à prédire")

except Exception as e:
    print(f"\n❌ ERREUR lecture weights: {e}")

# 5. Ontologie
print("\n" + "="*80)
print("EXPLICATION: ONTOLOGIE CLIMATIQUE")
print("="*80)

try:
    with open('knowledge_base/climate_ontology.json') as f:
        ontology = json.load(f)
    
    print("\n✅ ONTOLOGIE CREEE:")
    print(f"   Concepts: {len(ontology['concepts'])}")
    print(f"   Règles IF-THEN: {len(ontology['rules'])}")
    
    print("\n   📋 REGLES:")
    for rule in ontology['rules']:
        print(f"\n   {rule['id']}:")
        print(f"      Condition:  {rule['condition']}")
        print(f"      Conclusion: {rule['conclusion']}")
        print(f"      Alerte:     {rule['alert']}")

except Exception as e:
    print(f"\n❌ ERREUR lecture ontologie: {e}")

# RESUME
print("\n" + "="*80)
print("RESUME: QU'EST-CE QUI A ETE FAIT?")
print("="*80)

print("""
✅ ETAPE 1: CLASSIFICATION ADAPTATIVE

1. NOUVELLE COLONNE CREEE: 'extreme_event'
   - Ajoutée aux datasets train/val/test
   - Valeurs: 0, 1, 2, 3, 4 (5 classes)
   - Basée sur TEMPERATURE et STATION

2. METHODE DE CLASSIFICATION:
   - Seuils ADAPTATIFS par station (percentiles)
   - Chaque station a SES PROPRES seuils P99, P95, P05, P01
   - Exemple:
     * Phoenix P99 = 45°C (Desert, très chaud)
     * Seattle P99 = 30°C (Oceanic, tempéré)
   
3. CLASSES CREEES:
   0 = Normal           (85-90% données)
   1 = Canicule_Extreme (T > P99 station)
   2 = Forte_Chaleur    (P95 < T ≤ P99)
   3 = Froid_Extreme    (T < P01 station)
   4 = Froid_Prolonge   (P01 ≤ T < P05)

4. FICHIERS SUPPLEMENTAIRES:
   - station_thresholds.json: Seuils P99/P95/P05/P01 par station
   - class_weights.json: Poids pour compenser déséquilibre
   - climate_ontology.json: Règles IF-THEN (conforme cahier)

5. POURQUOI ADAPTATIF?
   - 30°C à Phoenix = Normal (fréquent)
   - 30°C à Seattle = Canicule (rare)
   → Les percentiles détectent ce qui est RARE LOCALEMENT
""")

print("\n" + "="*80)
print("🚀 PROCHAINE ETAPE: ENTRAINEMENT LSTM")
print("="*80)
print("\nCommande: python scripts/07_train_lstm_FINAL.py")
print("Durée: 30-60 min")
print("Output: Modèle trained + métriques (F1, Recall, ROC-AUC)")
