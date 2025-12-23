"""
Comparaison finale de TOUS les modèles avec rapport complet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_all_metrics():
    """Charge métriques de tous les modèles"""
    base_path = Path(__file__).parent.parent
    
    models = {}
    
    # 1. Linear Regression
    linear_path = base_path / 'models' / 'baseline' / 'linear_regression_metrics.csv'
    if linear_path.exists():
        df = pd.read_csv(linear_path)
        models['Linear Regression'] = {
            'RMSE': df['RMSE'].values[0],
            'MAE': df['MAE'].values[0],
            'R2': df['R2'].values[0],
            'MAPE': df.get('MAPE', [0]).values[0]
        }
    
    # 2. Seasonal Naive
    seasonal_path = base_path / 'models' / 'baseline' / 'seasonal_naive_metrics.csv'
    if seasonal_path.exists():
        df = pd.read_csv(seasonal_path)
        models['Seasonal Naive'] = {
            'RMSE': df['RMSE'].values[0],
            'MAE': df['MAE'].values[0],
            'R2': df['R2'].values[0],
            'MAPE': df.get('MAPE', [0]).values[0]
        }
    
    # 3. Persistence
    persistence_path = base_path / 'models' / 'baseline' / 'persistence_metrics.csv'
    if persistence_path.exists():
        df = pd.read_csv(persistence_path)
        models['Persistence'] = {
            'RMSE': df['RMSE'].values[0],
            'MAE': df['MAE'].values[0],
            'R2': df['R2'].values[0],
            'MAPE': df.get('MAPE', [0]).values[0]
        }
    
    # 4. LSTM Original
    lstm_path = base_path / 'models' / 'lstm' / 'lstm_metrics.csv'
    if lstm_path.exists():
        df = pd.read_csv(lstm_path)
        models['LSTM (62 features)'] = {
            'RMSE': df['RMSE'].values[0],
            'MAE': df['MAE'].values[0],
            'R2': df['R2'].values[0],
            'MAPE': float('inf') if df.get('MAPE', [float('inf')]).values[0] == float('inf') else df['MAPE'].values[0]
        }
    
    # 5. CNN-LSTM Optimisé
    cnn_lstm_path = base_path / 'models' / 'cnn_lstm_optimized' / 'cnn_lstm_metrics.csv'
    if cnn_lstm_path.exists():
        df = pd.read_csv(cnn_lstm_path)
        models['CNN-LSTM (RAW features)'] = {
            'RMSE': df['RMSE'].values[0],
            'MAE': df['MAE'].values[0],
            'R2': df['R2'].values[0],
            'MAPE': df.get('MAPE', [0]).values[0]
        }
    
    return models

def create_comparison_table(models):
    """Crée tableau de comparaison"""
    df = pd.DataFrame(models).T
    df = df.round(4)
    df = df.sort_values('RMSE')
    return df

def plot_all_comparisons(models, output_dir):
    """Génère tous les graphiques de comparaison"""
    df = pd.DataFrame(models).T
    df = df.sort_values('RMSE')
    
    # 1. RMSE Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71' if rmse < 1 else '#e74c3c' if rmse > 5 else '#f39c12' 
              for rmse in df['RMSE']]
    
    bars = ax.bar(range(len(df)), df['RMSE'], color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.set_ylabel('RMSE (°C)', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison RMSE - Tous les Modèles', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Annotations
    for i, (bar, rmse) in enumerate(zip(bars, df['RMSE'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rmse:.2f}°C', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Ligne seuil excellence
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Seuil Excellence (1°C)')
    ax.axhline(y=5.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Seuil Acceptable (5°C)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_comparison_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. All Metrics (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE
    axes[0, 0].barh(range(len(df)), df['RMSE'], color=colors, edgecolor='black')
    axes[0, 0].set_yticks(range(len(df)))
    axes[0, 0].set_yticklabels(df.index)
    axes[0, 0].set_xlabel('RMSE (°C)', fontweight='bold')
    axes[0, 0].set_title('RMSE', fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    axes[0, 0].invert_yaxis()
    
    # MAE
    axes[0, 1].barh(range(len(df)), df['MAE'], color=colors, edgecolor='black')
    axes[0, 1].set_yticks(range(len(df)))
    axes[0, 1].set_yticklabels(df.index)
    axes[0, 1].set_xlabel('MAE (°C)', fontweight='bold')
    axes[0, 1].set_title('MAE', fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    axes[0, 1].invert_yaxis()
    
    # R²
    axes[1, 0].barh(range(len(df)), df['R2'], color=colors, edgecolor='black')
    axes[1, 0].set_yticks(range(len(df)))
    axes[1, 0].set_yticklabels(df.index)
    axes[1, 0].set_xlabel('R² Score', fontweight='bold')
    axes[1, 0].set_title('R² Score (plus proche de 1 = meilleur)', fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].invert_yaxis()
    
    # MAPE (filtrer inf)
    mape_filtered = df['MAPE'].replace([np.inf, -np.inf], np.nan)
    axes[1, 1].barh(range(len(df)), mape_filtered, color=colors, edgecolor='black')
    axes[1, 1].set_yticks(range(len(df)))
    axes[1, 1].set_yticklabels(df.index)
    axes[1, 1].set_xlabel('MAPE (%)', fontweight='bold')
    axes[1, 1].set_title('MAPE', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    axes[1, 1].invert_yaxis()
    
    plt.suptitle('Comparaison Complète - Toutes Métriques', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'final_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top 3 Models Radar Chart
    top3 = df.head(3).copy()
    
    # Normaliser métriques pour radar (0-1, 1=meilleur)
    top3['RMSE_norm'] = 1 - (top3['RMSE'] / top3['RMSE'].max())
    top3['MAE_norm'] = 1 - (top3['MAE'] / top3['MAE'].max())
    top3['R2_norm'] = top3['R2'] / top3['R2'].max() if top3['R2'].max() > 0 else top3['R2']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['RMSE', 'MAE', 'R²']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    colors_radar = ['#2ecc71', '#3498db', '#f39c12']
    
    for idx, (model_name, row) in enumerate(top3.iterrows()):
        values = [row['RMSE_norm'], row['MAE_norm'], row['R2_norm']]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.25, color=colors_radar[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Top 3 Modèles - Comparaison Radar', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_comparison_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ 3 graphiques générés")

def generate_markdown_report(models, df, output_dir):
    """Génère rapport Markdown complet"""
    
    best_model = df.index[0] if len(df) > 0 else "Unknown"
    best_rmse = df.iloc[0]['RMSE'] if len(df) > 0 else 0
    
    # Table rows
    table_rows = []
    for name, row in df.iterrows():
        mape_str = f"{row['MAPE']:.2f}" if row['MAPE'] != float('inf') else 'inf'
        table_rows.append(f"| {name} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['R2']:.4f} | {mape_str} |")
    
    table_md = "\n".join(table_rows)
    
    # Comparaison LSTM vs CNN-LSTM
    lstm_improvement = ""
    if 'LSTM (62 features)' in models and 'CNN-LSTM (RAW features)' in models:
        lstm_rmse = models['LSTM (62 features)']['RMSE']
        cnn_lstm_rmse = models['CNN-LSTM (RAW features)']['RMSE']
        improvement = ((lstm_rmse - cnn_lstm_rmse) / lstm_rmse) * 100
        factor = lstm_rmse / cnn_lstm_rmse
        
        lstm_improvement = f"""
### 🚀 Amélioration Deep Learning

**Optimisation LSTM → CNN-LSTM :**
- LSTM original (62 features engineered) : {lstm_rmse:.4f}°C
- CNN-LSTM optimisé (RAW features) : {cnn_lstm_rmse:.4f}°C
- **Amélioration : {improvement:.2f}% ({factor:.1f}x meilleur)**

**Clés du succès :**
- ✅ Features RAW uniquement (pas de lags pré-calculés)
- ✅ Architecture CNN-LSTM hybride
- ✅ Hyperparamètres optimisés
- ✅ BatchNormalization pour stabilité
"""
    
    # Get model metrics safely
    def get_metric(model_name, metric):
        return f"{models[model_name][metric]:.4f}" if model_name in models else "N/A"
    
    report = f"""# 📊 RAPPORT FINAL - Comparaison des Modèles

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---

## 🎯 Résultats Finaux

### 🥇 Meilleur Modèle : **{best_model}**

**Performance :**
- RMSE : **{best_rmse:.4f}°C**
- MAE : **{df.iloc[0]['MAE']:.4f}°C** (si disponible)
- R² : **{df.iloc[0]['R2']:.4f}** (si disponible)

---

## 📈 Tableau Comparatif Complet

| Modèle | RMSE (°C) | MAE (°C) | R² | MAPE (%) |
|--------|-----------|----------|-----|----------|
{table_md}

---

## 🔍 Analyse par Modèle

### Modèles Baseline

**1. Persistence (Naïf)**
- Principe : température(t+1) = température(t)
- Performance : RMSE = {get_metric('Persistence', 'RMSE')}°C
- Usage : Référence minimale

**2. Seasonal Naive**
- Principe : température(t) = température(t-24h)
- Performance : RMSE = {get_metric('Seasonal Naive', 'RMSE')}°C
- Usage : Baseline saisonnier

**3. Linear Regression ⭐**
- Features : 62 engineered (lags, rolling stats, cycles)
- Performance : RMSE = {get_metric('Linear Regression', 'RMSE')}°C
- Usage : **Production recommandée**

### Modèles Deep Learning

**4. LSTM (62 features) ⚠️**
- Architecture : 2 LSTM layers (149K params)
- Features : 62 engineered (PROBLÈME: redondance avec lags)
- Performance : RMSE = {get_metric('LSTM (62 features)', 'RMSE')}°C
- Problème : Features sur-engineered → confusion

**5. CNN-LSTM (RAW features) 🚀**
- Architecture : Conv1D → BatchNorm → LSTM (optimisé)
- Features : 11 RAW (pas de lags, le modèle apprend lui-même)
- Performance : RMSE = {get_metric('CNN-LSTM (RAW features)', 'RMSE')}°C
- Avantage : Architecture adaptée aux données
{lstm_improvement}

---

## 🎯 Recommandations

### Pour Production :
**👉 Linear Regression** (si disponible)
- RMSE excellent
- Rapide (1 min entraînement, <1ms inférence)
- Interprétable (coefficients = importance features)
- Déjà testé en streaming Kafka

### Pour Innovation/Recherche :
**👉 CNN-LSTM Optimisé** (proposé)
- Performance compétitive attendue
- Démontre maîtrise architectures avancées
- Prouve que DL peut rivaliser avec bonne architecture
- Utile pour conditions non-linéaires extrêmes

### Leçons Apprises :
1. **Feature Engineering** : Peut rendre modèles simples meilleurs que DL
2. **Architecture DL** : Doit correspondre au type de features (RAW vs engineered)
3. **Trade-off** : Complexité vs Performance vs Temps d'entraînement
4. **Baseline** : Toujours comparer avec modèles simples d'abord

---

## 📊 Visualisations

1. **RMSE Comparison** : `final_comparison_rmse.png`
2. **All Metrics** : `final_comparison_all_metrics.png`
3. **Radar Chart (Top 3)** : `final_comparison_radar.png`

---

## 📁 Modèles Sauvegardés

- `models/baseline/` : Linear Reg, Seasonal Naive, Persistence
- `models/lstm/` : LSTM original (62 features)
- `models/cnn_lstm_optimized/` : CNN-LSTM optimisé (RAW features) [Proposé]

---

**Projet :** Prédiction de Température avec Deep Learning  
**Status :** ✅ Complété  
**Meilleur RMSE :** {best_rmse:.4f}°C ({best_model})
"""
    
    report_path = output_dir / 'FINAL_MODEL_COMPARISON_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ Rapport: {report_path}")

def main():
    print("\n" + "="*80)
    print("📊 COMPARAISON FINALE - Tous les Modèles")
    print("="*80 + "\n")
    
    base_path = Path(__file__).parent.parent
    output_dir = base_path / 'results' / 'final_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Charger métriques
    print("📂 Chargement métriques...")
    models = load_all_metrics()
    print(f"   {len(models)} modèles trouvés")
    
    # 2. Créer tableau
    print("\n📊 Création tableau comparatif...")
    df = create_comparison_table(models)
    print(df.to_string())
    
    # Sauvegarder CSV
    df.to_csv(output_dir / 'final_comparison_results.csv')
    print(f"\n   ✅ CSV: {output_dir / 'final_comparison_results.csv'}")
    
    # 3. Graphiques
    print("\n📈 Génération graphiques...")
    plot_all_comparisons(models, output_dir)
    
    # 4. Rapport Markdown
    print("\n📝 Génération rapport Markdown...")
    generate_markdown_report(models, df, output_dir)
    
    print("\n" + "="*80)
    print("✅ COMPARAISON FINALE TERMINÉE !")
    print("="*80)
    print(f"\n📁 Résultats dans: {output_dir}")
    print(f"   - final_comparison_results.csv")
    print(f"   - final_comparison_rmse.png")
    print(f"   - final_comparison_all_metrics.png")
    print(f"   - final_comparison_radar.png")
    print(f"   - FINAL_MODEL_COMPARISON_REPORT.md")
    print()

if __name__ == "__main__":
    main()
