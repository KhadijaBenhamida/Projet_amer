"""
Script de comparaison complète de tous les modèles

Compare les performances de :
- Persistence Model
- Seasonal Naive
- Linear Regression
- XGBoost (si disponible)
- LSTM (si disponible)

Génère :
- Tableau comparatif avec toutes les métriques (RMSE, MAE, R², MAPE)
- Visualisations (bar charts, radar chart, time series)
- Rapport détaillé en Markdown

Author: Climate Prediction Team
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Style pour les graphiques
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class ModelComparator:
    """
    Classe pour comparer les performances de tous les modèles.
    """
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.results = {}
        self.models_info = {}
        
    def load_baseline_results(self):
        """Charge les résultats des baselines."""
        logger.info("📂 Chargement des résultats baselines...")
        
        baseline_csv = self.base_path / 'models' / 'baselines' / 'baseline_comparison.csv'
        if baseline_csv.exists():
            df = pd.read_csv(baseline_csv)
            for _, row in df.iterrows():
                model_name = row['Model']
                self.results[model_name] = {
                    'MSE': row['MSE'],
                    'RMSE': row['RMSE'],
                    'MAE': row['MAE'],
                    'R2': row['R2'],
                    'MAPE': 0.0  # Non calculé pour baselines
                }
            logger.info(f"   ✅ {len(df)} baselines chargés")
        else:
            logger.warning("   ⚠️  Baseline results non trouvés")
    
    def load_lstm_results(self):
        """Charge les résultats LSTM."""
        logger.info("📂 Chargement des résultats LSTM...")
        
        lstm_metrics = self.base_path / 'models' / 'lstm' / 'lstm_metrics.csv'
        if lstm_metrics.exists():
            df = pd.read_csv(lstm_metrics)
            self.results['LSTM'] = {
                'MSE': df['MSE'].values[0],
                'RMSE': df['RMSE'].values[0],
                'MAE': df['MAE'].values[0],
                'R2': df['R2'].values[0],
                'MAPE': df['MAPE'].values[0]
            }
            logger.info("   ✅ LSTM chargé")
        else:
            logger.warning("   ⚠️  LSTM results non trouvés")
    
    def load_xgboost_results(self):
        """Charge les résultats XGBoost."""
        logger.info("📂 Chargement des résultats XGBoost...")
        
        xgb_metrics = self.base_path / 'models' / 'xgboost' / 'xgboost_metrics.csv'
        if xgb_metrics.exists():
            df = pd.read_csv(xgb_metrics)
            self.results['XGBoost'] = {
                'MSE': df['MSE'].values[0],
                'RMSE': df['RMSE'].values[0],
                'MAE': df['MAE'].values[0],
                'R2': df['R2'].values[0],
                'MAPE': df['MAPE'].values[0]
            }
            logger.info("   ✅ XGBoost chargé")
        else:
            logger.warning("   ⚠️  XGBoost results non trouvés")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Crée le tableau de comparaison."""
        logger.info("📊 Création du tableau comparatif...")
        
        df = pd.DataFrame(self.results).T
        df = df.sort_values('RMSE')
        
        logger.info(f"   ✅ {len(df)} modèles comparés")
        return df
    
    def plot_rmse_comparison(self, df: pd.DataFrame, save_path: Path):
        """Bar chart de comparaison des RMSE."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#2ecc71' if i == df['RMSE'].idxmin() else '#3498db' for i in df.index]
        df['RMSE'].plot(kind='bar', ax=ax, color=colors, alpha=0.8)
        
        ax.set_ylabel('RMSE (°C)', fontsize=12)
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_title('Comparaison RMSE - Tous les modèles', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, (idx, val) in enumerate(df['RMSE'].items()):
            ax.text(i, val + 0.5, f'{val:.3f}°C', ha='center', fontsize=10, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"   ✅ RMSE chart sauvegardé: {save_path}")
        plt.close()
    
    def plot_all_metrics_comparison(self, df: pd.DataFrame, save_path: Path):
        """Bar chart groupé pour toutes les métriques."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        titles = ['RMSE (°C)', 'MAE (°C)', 'R² Score', 'MAPE (%)']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            colors = ['#2ecc71' if i == df[metric].idxmin() else '#3498db' for i in df.index]
            if metric == 'R2':  # Pour R², on veut le max
                colors = ['#2ecc71' if i == df[metric].idxmax() else '#3498db' for i in df.index]
            
            df[metric].plot(kind='bar', ax=ax, color=colors, alpha=0.8)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Valeur', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Ajouter les valeurs
            for i, (idx, val) in enumerate(df[metric].items()):
                ax.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"   ✅ All metrics chart sauvegardé: {save_path}")
        plt.close()
    
    def plot_radar_chart(self, df: pd.DataFrame, save_path: Path):
        """Radar chart pour comparaison multi-dimensionnelle."""
        # Normaliser les métriques pour le radar chart (0-1)
        df_norm = df.copy()
        df_norm['RMSE_norm'] = 1 - (df_norm['RMSE'] / df_norm['RMSE'].max())
        df_norm['MAE_norm'] = 1 - (df_norm['MAE'] / df_norm['MAE'].max())
        df_norm['R2_norm'] = df_norm['R2']
        df_norm['MAPE_norm'] = 1 - (df_norm['MAPE'] / df_norm['MAPE'].max())
        
        # Sélectionner top 3 modèles
        top_models = df.nsmallest(3, 'RMSE').index
        
        categories = ['RMSE\n(inversé)', 'MAE\n(inversé)', 'R²', 'MAPE\n(inversé)']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        for model, color in zip(top_models, colors):
            values = [
                df_norm.loc[model, 'RMSE_norm'],
                df_norm.loc[model, 'MAE_norm'],
                df_norm.loc[model, 'R2_norm'],
                df_norm.loc[model, 'MAPE_norm']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Comparaison Multi-Métrique (Top 3 Modèles)', 
                     fontsize=14, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"   ✅ Radar chart sauvegardé: {save_path}")
        plt.close()
    
    def generate_markdown_report(self, df: pd.DataFrame, save_path: Path):
        """Génère un rapport Markdown détaillé."""
        logger.info("📄 Génération du rapport Markdown...")
        
        report = f"""# 📊 Rapport de Comparaison des Modèles
## Projet: Prédiction de Température Climatique

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🏆 Résultats Globaux

### Tableau Comparatif Complet

| Modèle | RMSE (°C) | MAE (°C) | R² | MAPE (%) | Rang |
|--------|-----------|----------|-----|----------|------|
"""
        
        for rank, (model, row) in enumerate(df.iterrows(), 1):
            report += f"| **{model}** | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['R2']:.4f} | {row['MAPE']:.2f} | {rank} |\n"
        
        best_model = df['RMSE'].idxmin()
        best_rmse = df.loc[best_model, 'RMSE']
        
        report += f"""
---

## 🥇 Meilleur Modèle: **{best_model}**

### Performances:
- **RMSE**: {best_rmse:.4f}°C
- **MAE**: {df.loc[best_model, 'MAE']:.4f}°C
- **R²**: {df.loc[best_model, 'R2']:.4f}
- **MAPE**: {df.loc[best_model, 'MAPE']:.2f}%

### Interprétation:
- Le modèle **{best_model}** atteint une précision de **±{best_rmse:.2f}°C**
- Il explique **{df.loc[best_model, 'R2']*100:.2f}%** de la variance
- Erreur moyenne absolue de **{df.loc[best_model, 'MAE']:.2f}°C**

---

## 📈 Analyse Comparative

### Baselines vs Machine Learning

"""
        
        # Comparer baselines vs ML
        baselines = [m for m in df.index if 'Persistence' in m or 'Seasonal' in m or 'Linear Regression' in m]
        ml_models = [m for m in df.index if m not in baselines]
        
        if ml_models:
            best_baseline = df.loc[baselines, 'RMSE'].min()
            best_ml = df.loc[ml_models, 'RMSE'].min()
            improvement = ((best_baseline - best_ml) / best_baseline) * 100
            
            report += f"""
**Amélioration ML vs Baselines**: {improvement:.2f}%
- Meilleur baseline: {best_baseline:.4f}°C
- Meilleur ML: {best_ml:.4f}°C
- Gain de précision: {best_baseline - best_ml:.4f}°C

"""
        
        report += f"""
---

## 🎯 Recommandations

### Pour la Production:
1. **Modèle recommandé**: {best_model}
2. **Précision attendue**: ±{best_rmse:.2f}°C
3. **Cas d'usage**: Prédiction température climatique en temps réel

### Pour l'Amélioration:
- Feature Engineering supplémentaire (interactions, polynomial features)
- Ensemble methods (stacking, voting)
- Hyperparameter tuning avancé
- Données météo supplémentaires (satellite, radar)

---

## 📁 Fichiers Générés

- `model_comparison_rmse.png`: Comparaison RMSE
- `model_comparison_all_metrics.png`: Toutes les métriques
- `model_comparison_radar.png`: Radar chart
- `model_comparison_results.csv`: Données complètes
- `model_comparison_report.md`: Ce rapport

---

*Rapport généré automatiquement par complete_model_comparison.py*
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"   ✅ Rapport sauvegardé: {save_path}")
    
    def run_comparison(self):
        """Exécute la comparaison complète."""
        logger.info("=" * 80)
        logger.info("📊 COMPARAISON COMPLÈTE DES MODÈLES")
        logger.info("=" * 80)
        
        # Charger tous les résultats
        self.load_baseline_results()
        self.load_lstm_results()
        self.load_xgboost_results()
        
        if not self.results:
            logger.error("❌ Aucun résultat trouvé!")
            return
        
        # Créer tableau
        df = self.create_comparison_table()
        
        # Créer dossier de sortie
        output_dir = self.base_path / 'results' / 'model_comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder CSV
        csv_path = output_dir / 'model_comparison_results.csv'
        df.to_csv(csv_path)
        logger.info(f"✅ Résultats sauvegardés: {csv_path}")
        
        # Afficher résultats
        logger.info("\n" + "=" * 80)
        logger.info("📊 RÉSULTATS DE COMPARAISON")
        logger.info("=" * 80)
        print(df.to_string())
        
        # Générer visualisations
        logger.info("\n" + "=" * 80)
        logger.info("📊 GÉNÉRATION DES VISUALISATIONS")
        logger.info("=" * 80)
        
        self.plot_rmse_comparison(df, output_dir / 'model_comparison_rmse.png')
        self.plot_all_metrics_comparison(df, output_dir / 'model_comparison_all_metrics.png')
        self.plot_radar_chart(df, output_dir / 'model_comparison_radar.png')
        
        # Générer rapport
        self.generate_markdown_report(df, output_dir / 'model_comparison_report.md')
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ COMPARAISON TERMINÉE")
        logger.info("=" * 80)
        logger.info(f"📁 Résultats dans: {output_dir}")
        
        # Meilleur modèle
        best_model = df['RMSE'].idxmin()
        best_rmse = df.loc[best_model, 'RMSE']
        logger.info(f"\n🏆 MEILLEUR MODÈLE: {best_model}")
        logger.info(f"   RMSE: {best_rmse:.4f}°C")
        logger.info(f"   R²: {df.loc[best_model, 'R2']:.4f}")


def main():
    """Fonction principale."""
    base_path = Path(__file__).parent.parent
    
    comparator = ModelComparator(base_path)
    comparator.run_comparison()


if __name__ == "__main__":
    main()
