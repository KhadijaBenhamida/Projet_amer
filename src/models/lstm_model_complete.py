"""
LSTM Model for Temperature Prediction - COMPLETE IMPLEMENTATION

Ce module implémente un modèle LSTM complet pour la prédiction de température.
Architecture optimisée pour séries temporelles climatiques.

Author: Climate Prediction Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import pickle
import json
from typing import Tuple, Optional, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMTemperatureModel:
    """
    Modèle LSTM pour prédiction de température avec séquences temporelles.
    
    Architecture : LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
    """
    
    def __init__(self, sequence_length=24, n_features=62):
        """
        Initialise le modèle LSTM.
        
        Args:
            sequence_length (int): Longueur de la fenêtre temporelle (24h par défaut)
            n_features (int): Nombre de features en input (62 features numériques)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        self.scaler = None
        self.imputer = None
        logger.info(f"✅ LSTM initialisé (seq_length={sequence_length}, features={n_features})")
        
    def build_model(self):
        """
        Construit l'architecture LSTM.
        
        Architecture:
            - LSTM(128, return_sequences=True) : Première couche LSTM avec 128 unités
            - Dropout(0.2) : Régularisation pour éviter overfitting
            - LSTM(64, return_sequences=False) : Seconde couche LSTM avec 64 unités
            - Dropout(0.2) : Régularisation
            - Dense(32, relu) : Couche dense intermédiaire
            - Dense(1) : Output layer (température prédite)
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            self.model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            optimizer = Adam(learning_rate=0.001)
            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            logger.info("✅ Architecture LSTM construite")
            logger.info(f"   Total params: {self.model.count_params():,}")
            
        except ImportError:
            logger.error("❌ TensorFlow non installé. Installation: pip install tensorflow")
            raise
            
    def prepare_sequences(self, X, y):
        """
        Prépare des séquences 3D pour le LSTM à partir de données 2D.
        
        Transforme (n_samples, n_features) → (n_samples - seq_length, seq_length, n_features)
        Utilise une fenêtre glissante (sliding window) pour créer les séquences.
        
        Args:
            X (np.ndarray): Features (n_samples, n_features)
            y (np.ndarray): Target values (n_samples,)
            
        Returns:
            tuple: (X_sequences, y_sequences)
                - X_sequences: shape (n_sequences, seq_length, n_features)
                - y_sequences: shape (n_sequences,)
                
        Example:
            Si X.shape = (1000, 62) et seq_length = 24:
            X_sequences.shape = (976, 24, 62)  # 976 = 1000 - 24
            y_sequences.shape = (976,)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.sequence_length):
            # Fenêtre glissante : prend les 24 dernières heures
            X_sequences.append(X[i:i + self.sequence_length])
            # Target : température à prédire (heure suivante)
            y_sequences.append(y[i + self.sequence_length])
            
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences)
        
        logger.info(f"   Séquences créées : X={X_seq.shape}, y={y_seq.shape}")
        return X_seq, y_seq
        
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
        """
        Entraîne le modèle LSTM avec Early Stopping et Learning Rate Reduction.
        
        Args:
            X_train: Features d'entraînement (n_train, n_features)
            y_train: Target d'entraînement (n_train,)
            X_val: Features de validation (n_val, n_features)
            y_val: Target de validation (n_val,)
            epochs: Nombre maximum d'epochs (default: 50)
            batch_size: Taille des batches (default: 256)
            
        Returns:
            history: Objet History de Keras avec les métriques d'entraînement
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        if self.model is None:
            self.build_model()
            
        # Préparer les séquences 3D
        logger.info("📊 Préparation des séquences temporelles...")
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        logger.info(f"🚀 Entraînement LSTM sur {len(X_train_seq):,} séquences...")
        logger.info(f"   Epochs max: {epochs}, Batch size: {batch_size}")
        
        # Entraînement
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        logger.info("✅ Entraînement terminé")
        return self.history
        
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """
        Évalue le modèle sur le jeu de test.
        
        Args:
            X_test: Features de test
            y_test: Target de test
            
        Returns:
            dict: Métriques de performance (RMSE, MAE, R2, MAPE)
        """
        logger.info("📊 Évaluation du modèle...")
        
        # Prédire
        predictions = self.predict(X_test)
        
        # Ajuster la taille de y_test (car séquences réduisent la taille)
        y_test_adjusted = y_test[self.sequence_length:]
        predictions_adjusted = predictions[:len(y_test_adjusted)]
        
        # Calcul des métriques
        mse = mean_squared_error(y_test_adjusted, predictions_adjusted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_adjusted, predictions_adjusted)
        r2 = r2_score(y_test_adjusted, predictions_adjusted)
        mape = np.mean(np.abs((y_test_adjusted - predictions_adjusted) / y_test_adjusted)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        logger.info("✅ Résultats LSTM:")
        for metric, value in metrics.items():
            logger.info(f"   {metric}: {value:.4f}")
            
        return metrics
    
    def predict(self, X):
        """
        Fait des prédictions avec le modèle.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            np.ndarray: Prédictions (n_samples - seq_length,)
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné. Appelez fit() d'abord.")
            
        # Créer séquences (y est dummy ici)
        X_seq, _ = self.prepare_sequences(X, np.zeros(len(X)))
        
        # Prédire
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()
        
    def save(self, path):
        """
        Sauvegarde le modèle LSTM.
        
        Args:
            path: Chemin où sauvegarder le modèle (.h5 ou .keras)
        """
        if self.model:
            self.model.save(path)
            logger.info(f"✅ Modèle sauvegardé: {path}")
            
            # Sauvegarder l'historique si disponible
            if self.history:
                history_path = Path(path).parent / 'lstm_history.json'
                with open(history_path, 'w') as f:
                    json.dump(self.history.history, f, indent=2)
                logger.info(f"✅ Historique sauvegardé: {history_path}")
        else:
            logger.warning("⚠️  Aucun modèle à sauvegarder")
            
    def load(self, path):
        """
        Charge un modèle LSTM sauvegardé.
        
        Args:
            path: Chemin du modèle à charger
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        logger.info(f"✅ Modèle chargé: {path}")
        
    def plot_history(self, save_path=None):
        """
        Trace les courbes d'apprentissage (loss et MAE).
        
        Args:
            save_path: Chemin où sauvegarder le graphique (optionnel)
        """
        if not self.history:
            logger.warning("⚠️  Pas d'historique disponible")
            return
            
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Train MAE')
        ax2.plot(self.history.history['val_mae'], label='Val MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (°C)')
        ax2.set_title('Model MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Graphique sauvegardé: {save_path}")
        else:
            plt.show()


def main():
    """
    Fonction principale : Entraîne et évalue le modèle LSTM.
    """
    logger.info("=" * 80)
    logger.info("🧠 LSTM TRAINING - Climate Temperature Prediction")
    logger.info("=" * 80)
    
    # Chemins
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / 'data' / 'processed' / 'splits'
    model_path = base_path / 'models' / 'lstm'
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Charger les données
    logger.info("📂 Chargement des données...")
    train = pd.read_parquet(data_path / 'train.parquet')
    val = pd.read_parquet(data_path / 'val.parquet')
    test = pd.read_parquet(data_path / 'test.parquet')
    
    logger.info(f"   Train: {train.shape}")
    logger.info(f"   Val: {val.shape}")
    logger.info(f"   Test: {test.shape}")
    
    # Charger scaler et imputer
    logger.info("📂 Chargement du preprocessing...")
    with open(data_path / 'scaler_new.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(data_path / 'imputer_new.pkl', 'rb') as f:
        imputer = pickle.load(f)
    
    # Séparation X et y
    X_train = train.drop('temperature', axis=1)
    y_train = train['temperature'].values
    X_val = val.drop('temperature', axis=1)
    y_val = val['temperature'].values
    X_test = test.drop('temperature', axis=1)
    y_test = test['temperature'].values
    
    # Sélectionner colonnes numériques
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    logger.info(f"   Features numériques: {len(numeric_cols)}")
    
    X_train_num = X_train[numeric_cols].values
    X_val_num = X_val[numeric_cols].values
    X_test_num = X_test[numeric_cols].values
    
    # Preprocessing
    logger.info("🔧 Preprocessing...")
    X_train_imputed = imputer.transform(X_train_num)
    X_train_scaled = scaler.transform(X_train_imputed)
    
    X_val_imputed = imputer.transform(X_val_num)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    X_test_imputed = imputer.transform(X_test_num)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Initialiser et entraîner LSTM
    logger.info("\n" + "=" * 80)
    logger.info("🚀 ENTRAÎNEMENT LSTM")
    logger.info("=" * 80)
    
    lstm_model = LSTMTemperatureModel(sequence_length=24, n_features=len(numeric_cols))
    history = lstm_model.fit(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=50,
        batch_size=256
    )
    
    # Évaluation
    logger.info("\n" + "=" * 80)
    logger.info("📊 ÉVALUATION")
    logger.info("=" * 80)
    
    metrics = lstm_model.evaluate(X_test_scaled, y_test)
    
    # Sauvegarder
    logger.info("\n" + "=" * 80)
    logger.info("💾 SAUVEGARDE")
    logger.info("=" * 80)
    
    lstm_model.save(model_path / 'lstm_model.h5')
    lstm_model.plot_history(save_path=model_path / 'training_curves.png')
    
    # Sauvegarder métriques
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(model_path / 'lstm_metrics.csv', index=False)
    logger.info(f"✅ Métriques sauvegardées: {model_path / 'lstm_metrics.csv'}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ LSTM TRAINING TERMINÉ")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
