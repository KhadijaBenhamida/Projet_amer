"""
Script rapide: Entraîne CNN-LSTM sur échantillon réduit (test rapide)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import logging
import pickle
from src.models.cnn_lstm_hybrid import CNNLSTMHybridModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 CNN-LSTM Training (Échantillon réduit pour test rapide)")
    
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data' / 'processed' / 'splits'
    model_path = base_path / 'models' / 'cnn_lstm'
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Charger seulement 100K samples pour test rapide
    logger.info("📂 Chargement données (échantillon)...")
    train = pd.read_parquet(data_path / 'train.parquet').sample(n=50000, random_state=42)
    val = pd.read_parquet(data_path / 'val.parquet').sample(n=10000, random_state=42)
    test = pd.read_parquet(data_path / 'test.parquet').sample(n=5000, random_state=42)
    
    logger.info(f"   Train: {train.shape}")
    logger.info(f"   Val: {val.shape}")
    logger.info(f"   Test: {test.shape}")
    
    # Initialiser modèle
    model = CNNLSTMHybridModel(sequence_length=24, n_features=16)  # 24h au lieu de 48h
    
    # Séparer
    X_train_full = train.drop('temperature', axis=1)
    y_train = train['temperature'].values
    X_val_full = val.drop('temperature', axis=1)
    y_val = val['temperature'].values
    X_test_full = test.drop('temperature', axis=1)
    y_test = test['temperature'].values
    
    # Sélectionner RAW features
    logger.info("🔧 Sélection features RAW...")
    X_train_raw = model.select_raw_features(X_train_full)
    X_val_raw = model.select_raw_features(X_val_full)
    X_test_raw = model.select_raw_features(X_test_full)
    
    # Preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    logger.info("🔧 Preprocessing...")
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_raw.values))
    X_val_scaled = scaler.transform(imputer.transform(X_val_raw.values))
    X_test_scaled = scaler.transform(imputer.transform(X_test_raw.values))
    
    # Update n_features
    model.n_features = X_train_scaled.shape[1]
    
    # Entraîner (epochs réduits)
    logger.info("\n🚀 ENTRAÎNEMENT (50 epochs max)...")
    history = model.fit(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=50,
        batch_size=256  # Batch + gros pour accélérer
    )
    
    # Évaluer
    logger.info("\n📊 ÉVALUATION...")
    metrics = model.evaluate(X_test_scaled, y_test)
    
    # Sauvegarder
    logger.info("\n💾 SAUVEGARDE...")
    model.save(model_path / 'cnn_lstm_model.h5')
    model.plot_history(save_path=model_path / 'cnn_lstm_training_curves.png')
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(model_path / 'cnn_lstm_metrics.csv', index=False)
    
    # Comparaison
    logger.info("\n📊 COMPARAISON:")
    lstm_path = base_path / 'models' / 'lstm' / 'lstm_metrics.csv'
    if lstm_path.exists():
        lstm = pd.read_csv(lstm_path)
        logger.info(f"   LSTM original : {lstm['RMSE'].values[0]:.4f}°C")
        logger.info(f"   CNN-LSTM Hybrid : {metrics['RMSE']:.4f}°C")
        improvement = ((lstm['RMSE'].values[0] - metrics['RMSE']) / lstm['RMSE'].values[0]) * 100
        logger.info(f"   Amélioration : {improvement:.2f}%")
    
    logger.info("\n✅ TERMINÉ!")

if __name__ == "__main__":
    main()
