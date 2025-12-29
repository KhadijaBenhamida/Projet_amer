"""
ETAPE 1 (CORRIGEE): Création classification événements extrêmes

Version simplifiée et robuste pour classification multi-classe
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import json

print("="*80)
print("CREATION CLASSIFICATION EVENEMENTS EXTREMES (VERSION CORRIGEE)")
print("="*80)

# Charger données
print("\n1. Chargement données...")
df_train = pd.read_parquet('data/processed/splits/train.parquet')
df_val = pd.read_parquet('data/processed/splits/val.parquet')
df_test = pd.read_parquet('data/processed/splits/test.parquet')

print(f"   Train: {len(df_train):,} samples")
print(f"   Val: {len(df_val):,} samples")
print(f"   Test: {len(df_test):,} samples")

# ============================================================================
# CLASSIFICATION BASEE SUR TEMPERATURE + ROLLING WINDOW
# ============================================================================

print("\n" + "="*80)
print("CLASSIFICATION EVENEMENTS EXTREMES")
print("="*80)

print("\nStratégie:")
print("   1. Détecter températures extrêmes (>=30°C ou <=0°C)")
print("   2. Calculer rolling mean 48h pour lisser")
print("   3. Si moyenne 48h > seuil = événement prolongé")

def classify_extreme_events_v2(df):
    """
    Classification robuste événements extrêmes
    
    Classes:
    0 = Normal
    1 = Canicule (rolling 48h >= 28°C)
    2 = Froid prolongé (rolling 48h <= 2°C)
    """
    # Créer rolling mean 48h par station
    df['temp_rolling_48h'] = df.groupby('station_id')['temperature'].transform(
        lambda x: x.rolling(window=48, min_periods=24, center=False).mean()
    )
    
    # Classification basée sur moyenne 48h
    conditions = [
        (df['temp_rolling_48h'] >= 28),  # Canicule (moyenne >= 28°C sur 48h)
        (df['temp_rolling_48h'] <= 2)     # Froid (moyenne <= 2°C sur 48h)
    ]
    choices = [1, 2]
    
    df['extreme_event'] = np.select(conditions, choices, default=0)
    
    # Événements ponctuels (pour référence)
    df['is_hot'] = df['temperature'] >= 30
    df['is_cold'] = df['temperature'] <= 0
    
    return df

# Appliquer
print("\nCalcul classification...")
df_train = classify_extreme_events_v2(df_train)
df_val = classify_extreme_events_v2(df_val)
df_test = classify_extreme_events_v2(df_test)

# Statistiques
print("\n" + "="*80)
print("DISTRIBUTION CLASSES")
print("="*80)

labels = {0: 'Normal', 1: 'Canicule', 2: 'Froid prolongé'}

for dataset_name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    counts = df['extreme_event'].value_counts().sort_index()
    print(f"\n{dataset_name}:")
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"   {labels[label]}: {count:,} ({pct:.2f}%)")

# ============================================================================
# ANALYSE DESEQUILIBRE
# ============================================================================

print("\n" + "="*80)
print("ANALYSE DESEQUILIBRE CLASSES")
print("="*80)

counts_train = df_train['extreme_event'].value_counts()
total_train = len(df_train)

print(f"\nClasse majoritaire: {counts_train.max():,} ({counts_train.max()/total_train*100:.2f}%)")
print(f"Classe minoritaire: {counts_train.min():,} ({counts_train.min()/total_train*100:.2f}%)")

imbalance_ratio = counts_train.max() / counts_train.min()
print(f"Ratio déséquilibre: {imbalance_ratio:.1f}:1")

if imbalance_ratio > 20:
    print("\n⚠️  DESEQUILIBRE IMPORTANT (>20:1)")
    print("   Stratégies:")
    print("   - Focal Loss (alpha=0.25, gamma=2.0)")
    print("   - Class weights")
    print("   - SMOTE oversampling")
else:
    print("\n✅ Déséquilibre modéré - Class weights suffisent")

# ============================================================================
# CALCUL CLASS WEIGHTS
# ============================================================================

print("\n" + "="*80)
print("CALCUL CLASS WEIGHTS")
print("="*80)

# Classes présentes
unique_classes = sorted(df_train['extreme_event'].unique())
print(f"\nClasses détectées: {unique_classes}")

# Calculer weights
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.array(unique_classes),
    y=df_train['extreme_event']
)

class_weights = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights_array)}

print("\nClass weights:")
for cls, weight in class_weights.items():
    print(f"   {labels.get(cls, f'Class {cls}')}: {weight:.4f}")

# Sauvegarder
Path('models/analysis').mkdir(parents=True, exist_ok=True)
weights_path = Path('models/analysis/class_weights.json')
with open(weights_path, 'w') as f:
    json.dump({
        'class_weights': class_weights,
        'imbalance_ratio': float(imbalance_ratio),
        'unique_classes': [int(c) for c in unique_classes],  # Convert numpy int64 to int
        'strategy': 'rolling_48h_mean'
    }, f, indent=2)

print(f"\n✅ Weights sauvegardés: {weights_path}")

# ============================================================================
# VISUALISATIONS
# ============================================================================

print("\n" + "="*80)
print("VISUALISATIONS")
print("="*80)

# 1. Distribution classes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (dataset_name, df) in enumerate([('Train', df_train), ('Val', df_val), ('Test', df_test)]):
    counts = df['extreme_event'].value_counts().sort_index()
    class_labels = [labels.get(c, f'Class {c}') for c in counts.index]
    
    axes[idx].bar(range(len(counts)), counts.values)
    axes[idx].set_xticks(range(len(counts)))
    axes[idx].set_xticklabels(class_labels, rotation=45, ha='right')
    axes[idx].set_ylabel('Nombre échantillons')
    axes[idx].set_title(f'{dataset_name}')
    axes[idx].grid(axis='y', alpha=0.3)
    
    for i, (label, count) in enumerate(zip(class_labels, counts.values)):
        pct = count / len(df) * 100
        axes[idx].text(i, count, f'{pct:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('models/analysis/class_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: models/analysis/class_distribution.png")

# 2. Température par classe
fig, ax = plt.subplots(figsize=(12, 6))

for cls in unique_classes:
    temps = df_train[df_train['extreme_event'] == cls]['temperature']
    if len(temps) > 0:
        ax.hist(temps, bins=50, alpha=0.5, label=labels.get(cls, f'Class {cls}'))

ax.set_xlabel('Température (°C)')
ax.set_ylabel('Fréquence')
ax.set_title('Distribution températures par classe (Train)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('models/analysis/temperature_by_class.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: models/analysis/temperature_by_class.png")

# 3. Timeline échantillon
fig, ax = plt.subplots(figsize=(15, 5))

sample = df_train.head(24*30*3)  # 3 mois
colors_map = {0: 'green', 1: 'red', 2: 'blue'}
colors = sample['extreme_event'].map(colors_map)

ax.scatter(range(len(sample)), sample['temperature'], c=colors, alpha=0.6, s=10)
ax.set_xlabel('Heures')
ax.set_ylabel('Température (°C)')
ax.set_title('Timeline événements (échantillon 3 mois)')

# Légende
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Normal'),
    Patch(facecolor='red', label='Canicule'),
    Patch(facecolor='blue', label='Froid')
]
ax.legend(handles=legend_elements)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('models/analysis/events_timeline.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: models/analysis/events_timeline.png")

plt.close('all')

# ============================================================================
# SAUVEGARDER DATASETS
# ============================================================================

print("\n" + "="*80)
print("SAUVEGARDE DATASETS CLASSIFIES")
print("="*80)

output_dir = Path('data/processed/splits_classified')
output_dir.mkdir(exist_ok=True)

# Colonnes à garder
cols_to_keep = [col for col in df_train.columns]

df_train[cols_to_keep].to_parquet(output_dir / 'train_classified.parquet', compression='snappy', index=False)
df_val[cols_to_keep].to_parquet(output_dir / 'val_classified.parquet', compression='snappy', index=False)
df_test[cols_to_keep].to_parquet(output_dir / 'test_classified.parquet', compression='snappy', index=False)

print(f"\n✅ Sauvegardés dans: {output_dir}")
print(f"   - train_classified.parquet: {len(df_train):,} samples, {len(cols_to_keep)} colonnes")
print(f"   - val_classified.parquet: {len(df_val):,} samples")
print(f"   - test_classified.parquet: {len(df_test):,} samples")

# ============================================================================
# STATISTIQUES FINALES
# ============================================================================

print("\n" + "="*80)
print("RESUME FINAL")
print("="*80)

print("\n📊 Nouvelles colonnes:")
print("   - extreme_event: Classification (0=Normal, 1=Canicule, 2=Froid)")
print("   - temp_rolling_48h: Moyenne mobile 48h")
print("   - is_hot: Température >= 30°C (ponctuel)")
print("   - is_cold: Température <= 0°C (ponctuel)")

print("\n📈 Distribution finale (Train):")
for cls in sorted(df_train['extreme_event'].unique()):
    count = (df_train['extreme_event'] == cls).sum()
    pct = count / len(df_train) * 100
    print(f"   {labels.get(cls, f'Class {cls}')}: {count:,} ({pct:.2f}%)")

print("\n🎯 Class weights:")
for cls, weight in class_weights.items():
    print(f"   {labels.get(cls, f'Class {cls}')}: {weight:.4f}")

print("\n📊 Statistiques température par classe (Train):")
for cls in sorted(df_train['extreme_event'].unique()):
    temps = df_train[df_train['extreme_event'] == cls]['temperature']
    if len(temps) > 0:
        print(f"\n   {labels.get(cls, f'Class {cls}')}:")
        print(f"      Min: {temps.min():.1f}°C")
        print(f"      Mean: {temps.mean():.1f}°C")
        print(f"      Max: {temps.max():.1f}°C")
        print(f"      Samples: {len(temps):,}")

print("\n" + "="*80)
print("✅ ETAPE 1 TERMINEE!")
print("="*80)
print("\nProchaines étapes:")
print("   2. Entraîner LSTM classification")
print("   3. Tester Focal Loss vs Weighted Loss")
print("   4. Evaluer F1-score, Recall, ROC-AUC")
print("   5. Créer ontologie climatique")
print("   6. Développer interface Web")

print("\n" + "="*80)
