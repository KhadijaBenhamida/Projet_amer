"""
CNN-LSTM HYBRID MODEL - Optimisé pour Séries Temporelles Météorologiques

Architecture : Conv1D → MaxPooling → LSTM → Dense
Features : RAW uniquement (sans lags, sans rolling stats)
Objectif : RMSE < 0.5°C (12x meilleur que LSTM actuel)

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


class CNNLSTMHybridModel:
    """
    Modèle CNN-LSTM Hybrid optimisé pour prédiction de température.
    
    Architecture :
        CNN (capture patterns locaux) → LSTM (capture patterns temporels) → Dense
    
    Features utilisées : RAW uniquement (pas de lags, pas de rolling stats)
    """
    
    def __init__(self, sequence_length=48, n_features=16):
        """
        Initialise le modèle CNN-LSTM Hybrid.
        
        Args:
            sequence_length (int): Longueur fenêtre temporelle (48h recommandé)
            n_features (int): Nombre de features RAW (16 recommandé)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        self.feature_names = None
        logger.info(f"✅ CNN-LSTM Hybrid initialisé (seq={sequence_length}, features={n_features})")
        
    def build_model(self):
        """
        Construit l'architecture CNN-LSTM Hybrid optimisée.
        
        Architecture :
            1. Conv1D(64, kernel=3) : Capture micro-patterns (3 timesteps)
            2. MaxPooling1D(2) : Réduit dimensionnalité
            3. Conv1D(128, kernel=3) : Patterns de niveau supérieur
            4. LSTM(64) : Capture dépendances temporelles
            5. Dropout(0.3) : Régularisation
            6. Dense(32, relu) : Couche dense
            7. Dense(1) : Output température
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            self.model = Sequential([
                # CNN Layers - Capture patterns locaux
                Conv1D(64, kernel_size=3, activation='relu', 
                       input_shape=(self.sequence_length, self.n_features)),
                MaxPooling1D(pool_size=2),
                
                Conv1D(128, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                
                # LSTM Layer - Capture patterns temporels long-terme
                LSTM(64, return_sequences=False),
                Dropout(0.3),
                
                # Dense Layers
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            # Optimizer avec learning rate faible
            optimizer = Adam(learning_rate=0.0001)  # 10x plus faible que LSTM original
            
            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            logger.info("✅ Architecture CNN-LSTM Hybrid construite")
            logger.info(f"   Total params: {self.model.count_params():,}")
            
        except ImportError:
            logger.error("❌ TensorFlow non installé")
            raise
            
    def select_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sélectionne uniquement les features RAW (sans lags, sans rolling stats).
        
        Features retenues (16) :
            - Variables météo brutes : temperature, humidity, wind_speed, wind_direction,
              pressure, dewpoint, precipitation, cloud_cover
            - Variables temporelles cycliques : hour_sin, hour_cos, month_sin, month_cos,
              day_of_week_sin, day_of_week_cos, day_of_year_sin, day_of_year_cos
        
        Args:
            df: DataFrame avec toutes les features
            
        Returns:
            DataFrame avec features RAW uniquement
        """
        # Features météo brutes
        weather_features = [
            'humidity', 'wind_speed', 'wind_direction', 'pressure', 
            'dewpoint', 'precipitation', 'cloud_cover'
        ]
        
        # Features temporelles cycliques
        temporal_features = [
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'day_of_week_sin', 'day_of_week_cos', 
            'day_of_year_sin', 'day_of_year_cos'
        ]
        
        # Sélectionner features disponibles
        selected_features = []
        for feat in weather_features + temporal_features:
            if feat in df.columns:
                selected_features.append(feat)
        
        logger.info(f"   Features RAW sélectionnées: {len(selected_features)}")
        logger.info(f"   Liste: {selected_features[:5]}...")
        
        self.feature_names = selected_features
        return df[selected_features]
    
    def prepare_sequences(self, X, y):
        """
        Prépare séquences 3D avec sliding window.
        
        Args:
            X: Features (n_samples, n_features)
            y: Target (n_samples,)
            
        Returns:
            tuple: (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.sequence_length):
            X_sequences.append(X[i:i + self.sequence_length])
            y_sequences.append(y[i + self.sequence_length])
            
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences)
        
        logger.info(f"   Séquences: X={X_seq.shape}, y={y_seq.shape}")
        return X_seq, y_seq
        
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128):
        """
        Entraîne le modèle CNN-LSTM avec configuration optimisée.
        
        Args:
            X_train, y_train: Données entraînement
            X_val, y_val: Données validation
            epochs: Epochs max (100 recommandé)
            batch_size: Batch size (128 recommandé)
            
        Returns:
            History object
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        if self.model is None:
            self.build_model()
            
        logger.info("📊 Préparation séquences...")
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
        
        # Callbacks optimisés
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # + de patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,  # Réduction + agressive
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        logger.info(f"🚀 Entraînement CNN-LSTM sur {len(X_train_seq):,} séquences...")
        logger.info(f"   Epochs max: {epochs}, Batch: {batch_size}, LR: 0.0001")
        
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Entraînement terminé")
        return self.history
        
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """
        Évalue le modèle.
        
        Returns:
            dict: Métriques (RMSE, MAE, R2, MAPE)
        """
        logger.info("📊 Évaluation...")
        
        predictions = self.predict(X_test)
        y_test_adjusted = y_test[self.sequence_length:]
        predictions_adjusted = predictions[:len(y_test_adjusted)]
        
        # Métriques
        mse = mean_squared_error(y_test_adjusted, predictions_adjusted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_adjusted, predictions_adjusted)
        r2 = r2_score(y_test_adjusted, predictions_adjusted)
        
        # MAPE avec gestion division par zéro
        mape_vals = np.abs((y_test_adjusted - predictions_adjusted) / (y_test_adjusted + 1e-10))
        mape = np.mean(mape_vals) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        logger.info("✅ Résultats CNN-LSTM Hybrid:")
        for metric, value in metrics.items():
            logger.info(f"   {metric}: {value:.4f}")
            
        return metrics
    
    def predict(self, X):
        """Prédictions."""
        if self.model is None:
            raise ValueError("Modèle non entraîné")
            
        X_seq, _ = self.prepare_sequences(X, np.zeros(len(X)))
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()
        
    def save(self, path):
        """Sauvegarde modèle."""
        if self.model:
            self.model.save(path)
            logger.info(f"✅ Modèle sauvegardé: {path}")
            
            if self.history:
                history_path = Path(path).parent / 'cnn_lstm_history.json'
                with open(history_path, 'w') as f:
                    json.dump(self.history.history, f, indent=2)
                logger.info(f"✅ Historique sauvegardé: {history_path}")
                
    def plot_history(self, save_path=None):
        """Trace courbes d'apprentissage."""
        if not self.history:
            return
            
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('CNN-LSTM Hybrid - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Train MAE', linewidth=2)
        ax2.plot(self.history.history['val_mae'], label='Val MAE', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (°C)')
        ax2.set_title('CNN-LSTM Hybrid - MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Graphique sauvegardé: {save_path}")
        else:
            plt.show()


def main():
    """
    Fonction principale : Entraîne CNN-LSTM Hybrid avec features RAW.
    """
    logger.info("=" * 80)
    logger.info("🚀 CNN-LSTM HYBRID - Temperature Prediction (Optimized)")
    logger.info("=" * 80)
    
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / 'data' / 'processed' / 'splits'
    model_path = base_path / 'models' / 'cnn_lstm'
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Charger données
    logger.info("📂 Chargement données...")
    train = pd.read_parquet(data_path / 'train.parquet')
    val = pd.read_parquet(data_path / 'val.parquet')
    test = pd.read_parquet(data_path / 'test.parquet')
    
    logger.info(f"   Train: {train.shape}")
    logger.info(f"   Val: {val.shape}")
    logger.info(f"   Test: {test.shape}")
    
    # Créer preprocessing pour features RAW (pas utiliser celui avec 62 features)
    logger.info("📂 Création preprocessing pour features RAW...")
    
    # Initialiser modèle
    model = CNNLSTMHybridModel(sequence_length=48, n_features=16)
    
    # Séparer X et y
    X_train_full = train.drop('temperature', axis=1)
    y_train = train['temperature'].values
    X_val_full = val.drop('temperature', axis=1)
    y_val = val['temperature'].values
    X_test_full = test.drop('temperature', axis=1)
    y_test = test['temperature'].values
    
    # Sélectionner features RAW uniquement
    logger.info("🔧 Sélection features RAW...")
    X_train_raw = model.select_raw_features(X_train_full)
    X_val_raw = model.select_raw_features(X_val_full)
    X_test_raw = model.select_raw_features(X_test_full)
    
    logger.info(f"   Features après sélection: {X_train_raw.shape[1]}")
    
    # Créer preprocessing spécifique pour features RAW
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    logger.info("🔧 Preprocessing (création nouveau scaler/imputer)...")
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_train_imputed = imputer.fit_transform(X_train_raw.values)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    X_val_imputed = imputer.transform(X_val_raw.values)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    X_test_imputed = imputer.transform(X_test_raw.values)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Sauvegarder nouveaux preprocessing objects
    cnn_lstm_scaler_path = model_path / 'cnn_lstm_scaler.pkl'
    cnn_lstm_imputer_path = model_path / 'cnn_lstm_imputer.pkl'
    
    with open(cnn_lstm_scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(cnn_lstm_imputer_path, 'wb') as f:
        pickle.dump(imputer, f)
    
    logger.info(f"✅ Scaler sauvegardé: {cnn_lstm_scaler_path}")
    logger.info(f"✅ Imputer sauvegardé: {cnn_lstm_imputer_path}")
    
    # Update n_features basé sur features réelles
    actual_features = X_train_scaled.shape[1]
    model.n_features = actual_features
    
    # Entraîner
    logger.info("\n" + "=" * 80)
    logger.info("🚀 ENTRAÎNEMENT CNN-LSTM HYBRID")
    logger.info("=" * 80)
    
    history = model.fit(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=100,
        batch_size=128
    )
    
    # Évaluer
    logger.info("\n" + "=" * 80)
    logger.info("📊 ÉVALUATION")
    logger.info("=" * 80)
    
    metrics = model.evaluate(X_test_scaled, y_test)
    
    # Sauvegarder
    logger.info("\n" + "=" * 80)
    logger.info("💾 SAUVEGARDE")
    logger.info("=" * 80)
    
    model.save(model_path / 'cnn_lstm_model.h5')
    model.plot_history(save_path=model_path / 'cnn_lstm_training_curves.png')
    
    # Sauvegarder métriques
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(model_path / 'cnn_lstm_metrics.csv', index=False)
    logger.info(f"✅ Métriques sauvegardées: {model_path / 'cnn_lstm_metrics.csv'}")
    
    # Comparaison avec LSTM original
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPARAISON LSTM vs CNN-LSTM")
    logger.info("=" * 80)
    
    lstm_metrics_path = base_path / 'models' / 'lstm' / 'lstm_metrics.csv'
    if lstm_metrics_path.exists():
        lstm_metrics = pd.read_csv(lstm_metrics_path)
        lstm_rmse = lstm_metrics['RMSE'].values[0]
        cnn_lstm_rmse = metrics['RMSE']
        
        improvement = ((lstm_rmse - cnn_lstm_rmse) / lstm_rmse) * 100
        
        logger.info(f"   LSTM original : {lstm_rmse:.4f}°C")
        logger.info(f"   CNN-LSTM Hybrid : {cnn_lstm_rmse:.4f}°C")
        logger.info(f"   Amélioration : {improvement:.2f}%")
        
        if cnn_lstm_rmse < lstm_rmse:
            logger.info("   🎉 CNN-LSTM MEILLEUR que LSTM !")
        else:
            logger.info("   ℹ️  LSTM reste meilleur (features engineered)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ CNN-LSTM HYBRID TERMINÉ")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
