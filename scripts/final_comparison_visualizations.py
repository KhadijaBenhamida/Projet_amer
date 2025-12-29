"""
COMPARAISON FINALE: TOUS LES MODELES TESTES

Génère visualisations complètes de tous les résultats
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

print("="*80)
print("COMPARAISON FINALE: LINEAR vs DEEP LEARNING")
print("="*80)

# Données des résultats
results = {
    'Modele': [
        'Linear Regression',
        'Ridge (ALL)',
        'GradientBoost (ALL)',
        'GradientBoost (RAW)',
        'Ridge (RAW)',
        'LSTM RAW v3',
        'CNN-LSTM v2',
        'LSTM Original',
        'CNN-LSTM v1'
    ],
    'Type': [
        'Linear',
        'Linear',
        'Tree',
        'Tree',
        'Linear',
        'Deep Learning',
        'Deep Learning',
        'Deep Learning',
        'Deep Learning'
    ],
    'Features': [68, 68, 68, 19, 19, 19, 38, 62, 11],
    'RMSE': [0.16, 0.159, 0.191, 1.123, 2.509, 6.0, 7.48, 6.20, 11.23],
    'R2': [0.9998, 0.9998, 0.9996, 0.9876, 0.9379, 0.65, 0.45, 0.62, -0.24],
    'Temps_min': [0.5, 1, 15, 10, 0.5, 1800, 180, 120, 120],
    'Status': ['OPTIMAL', 'Excellent', 'Bon', 'Bon sur RAW', 'Moyen', 'Échec', 'Échec', 'Échec', 'Échec Total']
}

df = pd.DataFrame(results)

# Sauvegarder
OUTPUT_DIR = Path('models/analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_DIR / 'all_models_comparison.csv', index=False)

print("\nRésultats complets:")
print(df.to_string())

# =====================================
# VISUALISATIONS
# =====================================

# Couleurs
colors = {
    'Linear': '#2E7D32',
    'Tree': '#1976D2', 
    'Deep Learning': '#D32F2F'
}

# 1. RMSE Comparison (Log scale)
fig, ax = plt.subplots(figsize=(14, 7))

df_sorted = df.sort_values('RMSE')
bars = ax.barh(df_sorted['Modele'], df_sorted['RMSE'], 
               color=[colors[t] for t in df_sorted['Type']])

# Ligne à 1°C (target)
ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Target: 1°C', alpha=0.7)

# Annotations
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    ax.text(row['RMSE'] + 0.3, i, f"{row['RMSE']:.2f}°C", 
            va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('RMSE (°C) - Échelle Logarithmique', fontsize=13, fontweight='bold')
ax.set_ylabel('Modèle', fontsize=13, fontweight='bold')
ax.set_title('Comparaison RMSE: Linear Regression vs Deep Learning', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xscale('log')
ax.grid(axis='x', alpha=0.3)

# Légende
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[k], label=k) for k in colors]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rmse_comparison_all.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Visualisation 1: rmse_comparison_all.png")

# 2. Performance vs Temps
fig, ax = plt.subplots(figsize=(12, 8))

for model_type in df['Type'].unique():
    df_type = df[df['Type'] == model_type]
    ax.scatter(df_type['Temps_min'], df_type['RMSE'], 
               s=300, alpha=0.7, color=colors[model_type],
               label=model_type, edgecolors='black', linewidth=1.5)
    
    # Annotations
    for _, row in df_type.iterrows():
        ax.annotate(row['Modele'].replace(' ', '\n'), 
                   (row['Temps_min'], row['RMSE']),
                   textcoords="offset points", xytext=(0,10), 
                   ha='center', fontsize=8, fontweight='bold')

ax.set_xlabel('Temps d\'Entraînement (minutes) - Échelle Log', fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE (°C) - Échelle Log', fontsize=13, fontweight='bold')
ax.set_title('Performance vs Temps d\'Entraînement\n(Plus bas-gauche = Meilleur)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(alpha=0.3)
ax.legend(fontsize=11)

# Zone optimale
ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Target RMSE: 1°C')
ax.axvline(10, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Target Temps: 10min')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'performance_vs_time.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Visualisation 2: performance_vs_time.png")

# 3. R² Comparison
fig, ax = plt.subplots(figsize=(14, 7))

df_sorted = df.sort_values('R2', ascending=False)
bars = ax.barh(df_sorted['Modele'], df_sorted['R2'], 
               color=[colors[t] for t in df_sorted['Type']])

# Ligne à 0.95
ax.axvline(0.95, color='green', linestyle='--', linewidth=2, label='Excellent: R²>0.95', alpha=0.7)

# Annotations
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    ax.text(row['R2'] + 0.02, i, f"R²={row['R2']:.4f}", 
            va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('R² (Coefficient de Détermination)', fontsize=13, fontweight='bold')
ax.set_ylabel('Modèle', fontsize=13, fontweight='bold')
ax.set_title('Comparaison R²: Qualité de l\'Ajustement', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlim(-0.3, 1.05)
ax.grid(axis='x', alpha=0.3)

# Légende
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[k], label=k) for k in colors]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'r2_comparison_all.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Visualisation 3: r2_comparison_all.png")

# 4. Efficacité (Performance / Temps)
df['Efficacite'] = (1 / df['RMSE']) / (df['Temps_min'] + 1)  # +1 pour éviter division par 0
df_sorted = df.sort_values('Efficacite', ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))

bars = ax.barh(df_sorted['Modele'], df_sorted['Efficacite'], 
               color=[colors[t] for t in df_sorted['Type']])

ax.set_xlabel('Efficacité (Performance / Temps)', fontsize=13, fontweight='bold')
ax.set_ylabel('Modèle', fontsize=13, fontweight='bold')
ax.set_title('Efficacité des Modèles: Meilleur Rapport Performance/Temps', 
             fontsize=15, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Annotations
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    ax.text(row['Efficacite'] + 0.05, i, f"{row['Efficacite']:.2f}", 
            va='center', fontsize=10, fontweight='bold')

# Légende
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[k], label=k) for k in colors]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Visualisation 4: efficiency_comparison.png")

# 5. Résumé par catégorie
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 5a. RMSE moyen par type
rmse_by_type = df.groupby('Type')['RMSE'].mean().sort_values()
axes[0].bar(range(len(rmse_by_type)), rmse_by_type.values, 
            color=[colors[t] for t in rmse_by_type.index])
axes[0].set_xticks(range(len(rmse_by_type)))
axes[0].set_xticklabels(rmse_by_type.index, rotation=45, ha='right')
axes[0].set_ylabel('RMSE Moyen (°C)', fontsize=12, fontweight='bold')
axes[0].set_title('Performance Moyenne par Type', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(rmse_by_type.values):
    axes[0].text(i, v + 0.3, f'{v:.2f}°C', ha='center', fontweight='bold')

# 5b. Temps moyen par type
time_by_type = df.groupby('Type')['Temps_min'].mean().sort_values()
axes[1].bar(range(len(time_by_type)), time_by_type.values, 
            color=[colors[t] for t in time_by_type.index])
axes[1].set_xticks(range(len(time_by_type)))
axes[1].set_xticklabels(time_by_type.index, rotation=45, ha='right')
axes[1].set_ylabel('Temps Moyen (minutes)', fontsize=12, fontweight='bold')
axes[1].set_title('Temps d\'Entraînement par Type', fontsize=13, fontweight='bold')
axes[1].set_yscale('log')
axes[1].grid(axis='y', alpha=0.3)

# 5c. Nombre de modèles par statut
status_counts = df['Status'].value_counts()
axes[2].bar(range(len(status_counts)), status_counts.values, 
            color=['green', 'orange', 'orange', 'red', 'red', 'darkred'])
axes[2].set_xticks(range(len(status_counts)))
axes[2].set_xticklabels(status_counts.index, rotation=45, ha='right')
axes[2].set_ylabel('Nombre de Modèles', fontsize=12, fontweight='bold')
axes[2].set_title('Distribution des Statuts', fontsize=13, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'summary_by_category.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Visualisation 5: summary_by_category.png")

# =====================================
# STATISTIQUES FINALES
# =====================================

print("\n" + "="*80)
print("STATISTIQUES FINALES")
print("="*80)

print("\n📊 Par Type de Modèle:")
print(df.groupby('Type')[['RMSE', 'R2', 'Temps_min']].agg({
    'RMSE': ['mean', 'min', 'max'],
    'R2': ['mean', 'min', 'max'],
    'Temps_min': ['mean', 'min', 'max']
}))

print("\n🏆 CHAMPION:")
best = df.loc[df['RMSE'].idxmin()]
print(f"  Modèle: {best['Modele']}")
print(f"  RMSE: {best['RMSE']:.4f}°C")
print(f"  R²: {best['R2']:.4f}")
print(f"  Temps: {best['Temps_min']:.1f} min")
print(f"  Status: {best['Status']}")

print("\n❌ PIRE:")
worst = df.loc[df['RMSE'].idxmax()]
print(f"  Modèle: {worst['Modele']}")
print(f"  RMSE: {worst['RMSE']:.4f}°C")
print(f"  R²: {worst['R2']:.4f}")
print(f"  Ratio vs Champion: {worst['RMSE'] / best['RMSE']:.1f}x pire")

print("\n⚡ PLUS EFFICACE:")
most_efficient = df.loc[df['Efficacite'].idxmax()]
print(f"  Modèle: {most_efficient['Modele']}")
print(f"  Efficacité: {most_efficient['Efficacite']:.2f}")

print("\n⏱️ PLUS RAPIDE:")
fastest = df.loc[df['Temps_min'].idxmin()]
print(f"  Modèle: {fastest['Modele']}")
print(f"  Temps: {fastest['Temps_min']:.1f} min")

print("\n🎯 Deep Learning vs Linear:")
dl_avg_rmse = df[df['Type'] == 'Deep Learning']['RMSE'].mean()
linear_avg_rmse = df[df['Type'] == 'Linear']['RMSE'].mean()
print(f"  DL moyenne: {dl_avg_rmse:.2f}°C")
print(f"  Linear moyenne: {linear_avg_rmse:.4f}°C")
print(f"  Ratio: DL est {dl_avg_rmse / linear_avg_rmse:.1f}x PIRE")

print("\n" + "="*80)
print("CONCLUSION: Linear Regression est OPTIMAL pour ce projet")
print("="*80)

print(f"\nFichiers sauvegardés dans: {OUTPUT_DIR}")
print("  - all_models_comparison.csv")
print("  - rmse_comparison_all.png")
print("  - performance_vs_time.png")
print("  - r2_comparison_all.png")
print("  - efficiency_comparison.png")
print("  - summary_by_category.png")
