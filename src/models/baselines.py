"""
Modèles de baseline pour la prédiction de température climatique.

Ce module implémente trois modèles de baseline:
1. Persistence Model: Prédit T(t) = T(t-1)
2. Seasonal Naive: Utilise la valeur de la même saison l'année précédente
3. Linear Regression: Régression linéaire simple

Auteur: Climate Prediction Team
Date: 2025
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class BaselineModel:
    """
    Classe de base pour tous les modèles de baseline.
    
    Attributs:
        name (str): Nom du modèle
        model: Le modèle entraîné (varie selon le type de baseline)
    """
    
    def __init__(self, name: str):
        """
        Initialise le modèle de baseline.
        
        Args:
            name: Nom du modèle
        """
        self.name = name
        self.model = None
        logger.info(f"Initialisation du modèle: {self.name}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaselineModel':
        """
        Entraîne le modèle sur les données d'entraînement.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            
        Returns:
            self: Instance du modèle entraîné
        """
        raise NotImplementedError("La méthode fit() doit être implémentée par les sous-classes")
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions sur les données de test.
        
        Args:
            X_test: Features de test
            
        Returns:
            Prédictions sous forme de numpy array
        """
        raise NotImplementedError("La méthode predict() doit être implémentée par les sous-classes")
    
    def save(self, filepath: Path) -> None:
        """
        Sauvegarde le modèle en pickle.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Modèle {self.name} sauvegardé: {filepath}")
    
    @staticmethod
    def load(filepath: Path) -> 'BaselineModel':
        """
        Charge un modèle depuis un fichier pickle.
        
        Args:
            filepath: Chemin du fichier à charger
            
        Returns:
            Instance du modèle chargé
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Modèle chargé: {filepath}")
        return model


class PersistenceModel(BaselineModel):
    """
    Modèle de persistence: Prédit T(t) = T(t-1)
    
    Ce modèle simple utilise la valeur de température précédente
    comme prédiction pour la valeur actuelle.
    """
    
    def __init__(self):
        """Initialise le modèle de persistence."""
        super().__init__("Persistence Model")
        self.last_value = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'PersistenceModel':
        """
        "Entraîne" le modèle de persistence.
        
        Pour ce modèle, l'entraînement consiste simplement à stocker
        la dernière valeur observée.
        
        Args:
            X_train: Features d'entraînement (non utilisé)
            y_train: Target d'entraînement
            
        Returns:
            self: Instance du modèle
        """
        # Gestion des valeurs manquantes
        y_clean = y_train.dropna()
        if len(y_clean) == 0:
            logger.warning("Aucune valeur valide dans y_train")
            self.last_value = 0.0
        else:
            self.last_value = y_clean.iloc[-1]
        
        logger.info(f"{self.name} entraîné. Dernière valeur: {self.last_value:.2f}")
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions en utilisant la persistence.
        
        Pour chaque prédiction, utilise la valeur précédente.
        
        Args:
            X_test: Features de test
            
        Returns:
            Array de prédictions
        """
        predictions = np.full(len(X_test), self.last_value)
        logger.info(f"{self.name}: {len(predictions)} prédictions générées")
        return predictions


class SeasonalNaiveModel(BaselineModel):
    """
    Modèle Seasonal Naive: Utilise la valeur de la même saison l'année précédente.
    
    Ce modèle exploite la saisonnalité en utilisant la valeur observée
    à la même période l'année précédente.
    """
    
    def __init__(self, seasonal_period: int = 365):
        """
        Initialise le modèle seasonal naive.
        
        Args:
            seasonal_period: Période saisonnière (défaut: 365 jours)
        """
        super().__init__("Seasonal Naive Model")
        self.seasonal_period = seasonal_period
        self.seasonal_data = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'SeasonalNaiveModel':
        """
        "Entraîne" le modèle seasonal naive.
        
        Stocke les valeurs historiques pour les utiliser comme prédictions.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            
        Returns:
            self: Instance du modèle
        """
        # Stocke toutes les valeurs historiques
        self.seasonal_data = y_train.copy()
        
        # Calcule la moyenne pour les valeurs de fallback
        self.global_mean = y_train.mean()
        
        logger.info(f"{self.name} entraîné avec {len(self.seasonal_data)} observations")
        logger.info(f"Période saisonnière: {self.seasonal_period} jours")
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions en utilisant les valeurs saisonnières.
        
        Pour chaque point, utilise la valeur de la même saison l'année précédente.
        Si cette valeur n'est pas disponible, utilise la moyenne globale.
        
        Args:
            X_test: Features de test
            
        Returns:
            Array de prédictions
        """
        predictions = []
        
        for i in range(len(X_test)):
            # Calcule l'index de la valeur saisonnière
            seasonal_index = len(self.seasonal_data) - self.seasonal_period + i
            
            if 0 <= seasonal_index < len(self.seasonal_data):
                # Utilise la valeur saisonnière si disponible
                pred = self.seasonal_data.iloc[seasonal_index]
                if pd.isna(pred):
                    pred = self.global_mean
            else:
                # Utilise la moyenne globale comme fallback
                pred = self.global_mean
            
            predictions.append(pred)
        
        logger.info(f"{self.name}: {len(predictions)} prédictions générées")
        return np.array(predictions)


class LinearRegressionBaseline(BaselineModel):
    """
    Modèle de régression linéaire simple.
    
    Utilise une régression linéaire standard de scikit-learn
    pour faire des prédictions basées sur les features.
    """
    
    def __init__(self):
        """Initialise le modèle de régression linéaire."""
        super().__init__("Linear Regression Baseline")
        self.model = LinearRegression()
        self.feature_names = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'LinearRegressionBaseline':
        """
        Entraîne le modèle de régression linéaire.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            
        Returns:
            self: Instance du modèle entraîné
        """
        # Gestion des valeurs manquantes
        mask = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_clean = X_train[mask]
        y_clean = y_train[mask]
        
        if len(X_clean) == 0:
            logger.error("Aucune donnée valide pour l'entraînement")
            raise ValueError("Pas de données valides après nettoyage")
        
        # Stocke les noms des features
        self.feature_names = X_clean.columns.tolist()
        
        # Entraîne le modèle
        self.model.fit(X_clean, y_clean)
        
        logger.info(f"{self.name} entraîné sur {len(X_clean)} échantillons")
        logger.info(f"Features utilisées: {len(self.feature_names)}")
        logger.info(f"Coefficients R²: {self.model.score(X_clean, y_clean):.4f}")
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Fait des prédictions avec la régression linéaire.
        
        Args:
            X_test: Features de test
            
        Returns:
            Array de prédictions
        """
        # Gestion des valeurs manquantes
        X_test_clean = X_test.fillna(X_test.mean())
        
        predictions = self.model.predict(X_test_clean)
        logger.info(f"{self.name}: {len(predictions)} prédictions générées")
        
        return predictions


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
    """
    Évalue un modèle avec plusieurs métriques.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        model_name: Nom du modèle
        
    Returns:
        Dictionnaire contenant les métriques d'évaluation
    """
    # Gestion des valeurs manquantes
    mask = ~pd.isna(y_true)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        logger.warning(f"Aucune valeur valide pour l'évaluation de {model_name}")
        return {
            'Model': model_name,
            'MSE': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'R2': np.nan
        }
    
    # Calcul des métriques
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Évaluation: {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"MSE:  {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"R²:   {r2:.4f}")
    logger.info(f"{'='*60}\n")
    
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les données d'entraînement, validation et test.
    
    Args:
        data_dir: Répertoire contenant les fichiers de données
        
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    logger.info("Chargement des données...")
    
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"
    
    # Vérification de l'existence des fichiers
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {path}")
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    logger.info(f"Train: {len(train_df)} échantillons")
    logger.info(f"Val:   {len(val_df)} échantillons")
    logger.info(f"Test:  {len(test_df)} échantillons")
    
    return train_df, val_df, test_df


def prepare_features(df: pd.DataFrame, target_col: str = 'temperature') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare les features et la target à partir d'un DataFrame.
    
    Args:
        df: DataFrame contenant les données
        target_col: Nom de la colonne target
        
    Returns:
        Tuple (X, y) avec les features et la target
    """
    # Sépare la target des features
    if target_col not in df.columns:
        raise ValueError(f"Colonne target '{target_col}' introuvable")
    
    y = df[target_col].copy()
    
    # Sélectionne les features (toutes les colonnes sauf la target)
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    
    # Sélectionne uniquement les colonnes numériques
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    logger.info(f"Features: {X.shape[1]} colonnes")
    logger.info(f"Target: {target_col}")
    
    return X, y


def train_and_evaluate_baselines(
    data_dir: Path = Path("data/processed/splits"),
    output_dir: Path = Path("models/baselines"),
    target_col: str = 'T_moyenne'
) -> pd.DataFrame:
    """
    Entraîne et évalue tous les modèles de baseline.
    
    Cette fonction:
    1. Charge les données d'entraînement, validation et test
    2. Entraîne les trois modèles de baseline
    3. Évalue chaque modèle sur l'ensemble de test
    4. Sauvegarde les résultats et les modèles
    
    Args:
        data_dir: Répertoire contenant les données
        output_dir: Répertoire pour sauvegarder les résultats
        target_col: Nom de la colonne target
        
    Returns:
        DataFrame contenant les résultats de comparaison
    """
    logger.info("\n" + "="*80)
    logger.info("ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES DE BASELINE")
    logger.info("="*80 + "\n")
    
    # Crée le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charge les données
    train_df, val_df, test_df = load_data(data_dir)
    
    # Prépare les features et targets
    X_train, y_train = prepare_features(train_df, target_col)
    X_val, y_val = prepare_features(val_df, target_col)
    X_test, y_test = prepare_features(test_df, target_col)
    
    # Initialise les modèles
    models = [
        PersistenceModel(),
        SeasonalNaiveModel(seasonal_period=365),
        LinearRegressionBaseline()
    ]
    
    # Entraîne et évalue chaque modèle
    results = []
    
    for model in models:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# Traitement du modèle: {model.name}")
        logger.info(f"{'#'*80}\n")
        
        try:
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions sur le test set
            y_pred = model.predict(X_test)
            
            # Évaluation
            metrics = evaluate_model(y_test, y_pred, model.name)
            results.append(metrics)
            
            # Sauvegarde du modèle
            model_path = output_dir / f"{model.name.lower().replace(' ', '_')}.pkl"
            model.save(model_path)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {model.name}: {str(e)}")
            logger.exception(e)
            continue
    
    # Crée un DataFrame avec les résultats
    results_df = pd.DataFrame(results)
    
    # Sauvegarde les résultats
    results_path = output_dir / "baseline_comparison.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nRésultats sauvegardés: {results_path}")
    
    # Affiche le tableau récapitulatif
    logger.info("\n" + "="*80)
    logger.info("RÉSULTATS FINAUX - COMPARAISON DES MODÈLES")
    logger.info("="*80)
    logger.info("\n" + results_df.to_string(index=False))
    logger.info("\n" + "="*80 + "\n")
    
    return results_df


def main():
    """
    Fonction principale pour exécuter l'entraînement et l'évaluation des baselines.
    """
    try:
        # Définition des chemins
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data" / "processed" / "splits"
        output_dir = project_root / "models" / "baselines"
        
        # Exécute l'entraînement et l'évaluation
        results_df = train_and_evaluate_baselines(
            data_dir=data_dir,
            output_dir=output_dir,
            target_col='temperature'
        )
        
        logger.info("✓ Processus terminé avec succès!")
        return results_df
        
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()
