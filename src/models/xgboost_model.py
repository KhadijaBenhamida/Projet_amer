"""
XGBoost Temperature Prediction Model

Ce module implémente un modèle XGBoost optimisé pour la prédiction de température.
Le modèle utilise des données météorologiques et temporelles pour prédire la température moyenne (T_moyenne).

Author: Climate Prediction Team
Date: December 2025
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostTemperatureModel:
    """
    Modèle XGBoost pour la prédiction de température.
    
    Cette classe encapsule un modèle XGBoost avec des fonctionnalités pour:
    - L'entraînement avec early stopping
    - La prédiction
    - La sauvegarde et le chargement
    - La visualisation de l'importance des features
    
    Attributes:
        model (xgb.XGBRegressor): Le modèle XGBoost
        feature_names (list): Liste des noms des features
        feature_importance (pd.DataFrame): Importance des features
    """
    
    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 10,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Initialise le modèle XGBoost avec les hyperparamètres.
        
        Args:
            n_estimators: Nombre d'arbres de boosting
            max_depth: Profondeur maximale des arbres
            learning_rate: Taux d'apprentissage
            subsample: Fraction des échantillons pour chaque arbre
            colsample_bytree: Fraction des features pour chaque arbre
            min_child_weight: Poids minimal des instances dans un noeud
            gamma: Réduction minimale de perte pour split
            reg_alpha: Régularisation L1
            reg_lambda: Régularisation L2
            random_state: Seed pour la reproductibilité
            n_jobs: Nombre de threads (-1 pour tous)
            **kwargs: Autres paramètres XGBoost
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method='hist',
            **kwargs
        )
        
        self.feature_names = None
        self.feature_importance = None
        
        logger.info(
            f"XGBoost model initialized with n_estimators={n_estimators}, "
            f"max_depth={max_depth}, learning_rate={learning_rate}"
        )
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
        eval_metric: str = 'rmse',
        verbose: bool = True
    ) -> 'XGBoostTemperatureModel':
        """
        Entraîne le modèle XGBoost avec early stopping.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Target de validation (optionnel)
            early_stopping_rounds: Nombre de rounds sans amélioration avant arrêt
            eval_metric: Métrique d'évaluation ('rmse', 'mae', etc.)
            verbose: Afficher les logs d'entraînement
            
        Returns:
            self: Instance du modèle entraîné
        """
        logger.info(f"Starting model training with {len(X_train)} samples...")
        
        # Stocker les noms des features
        self.feature_names = list(X_train.columns)
        
        # Préparer les paramètres d'entraînement
        fit_params = {}
        
        # Ajouter la validation si fournie
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_train, y_train), (X_val, y_val)]
            logger.info(f"Using validation set with {len(X_val)} samples")
        
        # Entraîner le modèle (XGBoost 3.x ne supporte plus early_stopping_rounds directement)
        self.model.fit(X_train, y_train, **fit_params, verbose=verbose)
        
        # Calculer l'importance des features
        self._compute_feature_importance()
        
        logger.info(f"Model training completed. Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Effectue des prédictions sur les données.
        
        Args:
            X: Features pour la prédiction
            
        Returns:
            Prédictions du modèle
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.predict(X)
    
    def _compute_feature_importance(self) -> None:
        """Calcule et stocke l'importance des features."""
        if self.feature_names is None:
            logger.warning("Feature names not set. Cannot compute feature importance.")
            return
        
        # Obtenir l'importance des features
        importance = self.model.feature_importances_
        
        # Créer un DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Feature importance computed for {len(self.feature_importance)} features")
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualise l'importance des features.
        
        Args:
            top_n: Nombre de features à afficher
            figsize: Taille de la figure
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not computed. Call fit() first.")
            return
        
        # Sélectionner les top N features
        top_features = self.feature_importance.head(top_n)
        
        # Créer le graphique
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - XGBoost Model', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Sauvegarder si un chemin est fourni
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde le modèle dans un fichier.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Sauvegarder le modèle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        # Obtenir la taille du fichier
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"Model saved to {filepath} ({file_size_mb:.2f} MB)")
    
    @classmethod
    def load(cls, filepath: str) -> 'XGBoostTemperatureModel':
        """
        Charge un modèle depuis un fichier.
        
        Args:
            filepath: Chemin du fichier à charger
            
        Returns:
            Instance du modèle chargé
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")
        return model


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "Dataset"
) -> Dict[str, float]:
    """
    Évalue les prédictions du modèle.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        dataset_name: Nom du dataset pour les logs
        
    Returns:
        Dictionnaire des métriques
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    logger.info(f"\n{dataset_name} Metrics:")
    logger.info(f"  RMSE: {rmse:.4f}°C")
    logger.info(f"  MAE:  {mae:.4f}°C")
    logger.info(f"  R²:   {r2:.4f}")
    
    return metrics


def train_xgboost_model(
    data_dir: str = "data/processed/splits",
    output_dir: str = "models/xgboost",
    target_col: str = "temperature",
    n_estimators: int = 1000,
    max_depth: int = 10,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50
) -> Tuple[XGBoostTemperatureModel, Dict[str, Any]]:
    """
    Fonction principale pour entraîner le modèle XGBoost.
    
    Cette fonction:
    1. Charge les données depuis data/processed/splits/
    2. Entraîne le modèle avec early stopping
    3. Évalue sur train/val/test
    4. Sauvegarde le modèle et les résultats
    
    Args:
        data_dir: Répertoire contenant les données
        output_dir: Répertoire de sortie pour les résultats
        target_col: Colonne cible
        n_estimators: Nombre d'arbres
        max_depth: Profondeur maximale des arbres
        learning_rate: Taux d'apprentissage
        early_stopping_rounds: Rounds pour early stopping
        
    Returns:
        Tuple (modèle entraîné, résultats)
    """
    logger.info("="*80)
    logger.info("Starting XGBoost Temperature Prediction Model Training")
    logger.info("="*80)
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Charger les données
    logger.info("\n1. Loading data...")
    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "val.parquet")
    test_path = os.path.join(data_dir, "test.parquet")
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    logger.info(f"  Train set: {len(train_df):,} samples")
    logger.info(f"  Val set:   {len(val_df):,} samples")
    logger.info(f"  Test set:  {len(test_df):,} samples")
    
    # Séparer features et target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Drop non-numeric columns (string columns like station_id, city, etc.)
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    X_train = X_train[numeric_cols]
    X_val = X_val[numeric_cols]
    X_test = X_test[numeric_cols]
    
    logger.info(f"  Number of features: {X_train.shape[1]}")
    
    # 2. Initialiser et entraîner le modèle
    logger.info("\n2. Training XGBoost model...")
    model = XGBoostTemperatureModel(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )
    
    model.fit(
        X_train, y_train,
        X_val, y_val,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric='rmse',
        verbose=False
    )
    
    # 3. Évaluer le modèle
    logger.info("\n3. Evaluating model...")
    
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Métriques
    train_metrics = evaluate_model(y_train, y_train_pred, "Train")
    val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
    test_metrics = evaluate_model(y_test, y_test_pred, "Test")
    
    # 4. Sauvegarder le modèle
    logger.info("\n4. Saving model and results...")
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    model.save(model_path)
    
    # 5. Sauvegarder l'importance des features
    if model.feature_importance is not None:
        importance_csv_path = os.path.join(output_dir, "feature_importance.csv")
        model.feature_importance.to_csv(importance_csv_path, index=False)
        logger.info(f"Feature importance saved to {importance_csv_path}")
        
        # Créer le graphique d'importance
        importance_plot_path = os.path.join(output_dir, "feature_importance.png")
        model.plot_feature_importance(top_n=20, save_path=importance_plot_path)
    
    # 6. Sauvegarder les résultats
    results = {
        'model': 'XGBoost',
        'train_rmse': train_metrics['RMSE'],
        'train_mae': train_metrics['MAE'],
        'train_r2': train_metrics['R2'],
        'val_rmse': val_metrics['RMSE'],
        'val_mae': val_metrics['MAE'],
        'val_r2': val_metrics['R2'],
        'test_rmse': test_metrics['RMSE'],
        'test_mae': test_metrics['MAE'],
        'test_r2': test_metrics['R2'],
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'early_stopping_rounds': early_stopping_rounds,
        'best_iteration': model.model.best_iteration if hasattr(model.model, 'best_iteration') else None
    }
    
    results_df = pd.DataFrame([results])
    results_path = os.path.join(output_dir, "xgboost_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Training completed successfully!")
    logger.info("="*80)
    
    return model, results


def main():
    """
    Point d'entrée principal pour l'entraînement du modèle XGBoost.
    """
    try:
        # Entraîner le modèle
        model, results = train_xgboost_model(
            data_dir="data/processed/splits",
            output_dir="models/xgboost",
            target_col="temperature",
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.05,
            early_stopping_rounds=50
        )
        
        logger.info("\nFinal Test Set Performance:")
        logger.info(f"  RMSE: {results['test_rmse']:.4f}°C")
        logger.info(f"  MAE:  {results['test_mae']:.4f}°C")
        logger.info(f"  R²:   {results['test_r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
