"""
CNN-LSTM v2 FULL - Avec TOUTES les features stratégiques
- RAW features (46): météo + temporelles + catégorielles
- Lags courts (1-6h): contexte temporel récent
- Architecture plus profonde
- Objectif: RMSE 2-4°C
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle

def select_strategic_features(df: pd.DataFrame):
    """
    Sélectionne features stratégiques:
    - RAW numériques (météo, temporelles)
    - Lags courts (1-6h) pour contexte immédiat
    - EXCLUT: station_id, station_name, city (catégoriels non numériques)
    """
    # Features RAW numériques disponibles
    raw_features = []
    
    # Météo brutes
    weather = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 
               'pressure', 'dewpoint', 'precipitation', 'cloud_cover',
               'sea_level_pressure']
    raw_features.extend([f for f in weather if f in df.columns])
    
    # Temporelles (valeurs brutes + encodages cycliques)
    temporal = ['year', 'month', 'day', 'hour', 'minute',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'day_of_week_sin', 'day_of_week_cos',
                'day_of_year_sin', 'day_of_year_cos']
    raw_features.extend([f for f in temporal if f in df.columns])
    
    # Lags courts (1-6h) - contexte temporel immédiat
    lag_features = [col for col in df.columns if 'lag' in col.lower()]
    # Garder seulement lags 1-6
    lag_short = [f for f in lag_features if any(f'lag_{i}' in f for i in range(1, 7))]
    
    # Combiner
    all_features = raw_features + lag_short
    
    # Filtrer colonnes existantes et numériques
    selected = []
    for col in all_features:
        if col in df.columns and col != 'temperature':  # Exclure target
            try:
                # Vérifier si numérique
                _ = pd.to_numeric(df[col].iloc[:100], errors='raise')
                selected.append(col)
            except:
                pass  # Skip colonnes non-numériques
    
    return df[selected], selected

def create_sequences(X, y, seq_length=24):
    """Crée séquences 3D avec sliding window"""
    n = len(X) - seq_length
    X_seq = np.zeros((n, seq_length, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        X_seq[i] = X[i:i + seq_length]
        y_seq[i] = y[i + seq_length]
    
    return X_seq, y_seq

def build_cnn_lstm_v2(seq_length, n_features):
    """
    Architecture CNN-LSTM v2 - Plus profonde et robuste
    """
    model = Sequential([
        # CNN Block 1 - Patterns locaux (1-3h)
        Conv1D(64, kernel_size=3, activation='relu', 
               input_shape=(seq_length, n_features)),
        BatchNormalization(),
        MaxPooling1D(2),
        
        # CNN Block 2 - Patterns niveau intermédiaire (3-6h)
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        
        # CNN Block 3 - Patterns niveau supérieur (6-12h)
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        
        # LSTM Double - Patterns temporels complexes
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def main():
    print("\n" + "="*70)
    print("🚀 CNN-LSTM v2 FULL - Avec toutes features stratégiques")
    print("="*70)
    
    # 1. Charger données
    print("\n📂 Chargement données...")
    train = pd.read_parquet('data/processed/splits/train.parquet')
    val = pd.read_parquet('data/processed/splits/val.parquet')
    test = pd.read_parquet('data/processed/splits/test.parquet')
    
    # Échantillonnage stratifié (100K samples pour équilibre vitesse/qualité)
    print("\n🎲 Échantillonnage stratifié (100K samples)...")
    sample_size = min(100000, len(train))
    train = train.sample(n=sample_size, random_state=42).sort_index()
    
    print(f"   Train: {train.shape}")
    print(f"   Val: {val.shape}")
    print(f"   Test: {test.shape}")
    
    # 2. Séparer X et y
    y_train = train['temperature'].values
    y_val = val['temperature'].values
    y_test = test['temperature'].values
    
    X_train_full = train.drop('temperature', axis=1)
    X_val_full = val.drop('temperature', axis=1)
    X_test_full = test.drop('temperature', axis=1)
    
    # 3. Sélectionner features stratégiques
    print("\n🔧 Sélection features STRATÉGIQUES (RAW + Lags courts)...")
    X_train_selected, feature_names = select_strategic_features(X_train_full)
    X_val_selected, _ = select_strategic_features(X_val_full)
    X_test_selected, _ = select_strategic_features(X_test_full)
    
    print(f"   ✅ Features sélectionnées: {len(feature_names)}")
    print(f"   📋 Liste complète:")
    for i, feat in enumerate(feature_names, 1):
        print(f"      {i:2d}. {feat}")
    
    # 4. Preprocessing
    print("\n🔧 Preprocessing (Imputer + Scaler)...")
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_selected))
    X_val_scaled = scaler.transform(imputer.transform(X_val_selected))
    X_test_scaled = scaler.transform(imputer.transform(X_test_selected))
    
    # 5. Créer séquences
    seq_length = 24  # 24h de contexte
    print(f"\n📊 Création séquences (window={seq_length}h)...")
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length)
    
    print(f"   Train séquences: {X_train_seq.shape}")
    print(f"   Val séquences: {X_val_seq.shape}")
    print(f"   Test séquences: {X_test_seq.shape}")
    
    # 6. Build model
    n_features = X_train_seq.shape[2]
    print(f"\n🏗️ Construction CNN-LSTM v2 (3 Conv1D + 2 LSTM)...")
    model = build_cnn_lstm_v2(seq_length, n_features)
    print(f"   ✅ Paramètres: {model.count_params():,}")
    
    # 7. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
    ]
    
    # 8. Entraînement
    print("\n🚀 ENTRAÎNEMENT (50 epochs max, early stopping patience=15)...")
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )
    
    # 9. Évaluation
    print("\n" + "="*70)
    print("📊 ÉVALUATION")
    print("="*70)
    
    y_pred = model.predict(X_test_seq, verbose=0).flatten()
    
    mse = mean_squared_error(y_test_seq, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_seq, y_pred)
    r2 = r2_score(y_test_seq, y_pred)
    mape = np.mean(np.abs((y_test_seq - y_pred) / y_test_seq)) * 100
    
    print(f"   MSE  : {mse:.4f}")
    print(f"   RMSE : {rmse:.4f}°C")
    print(f"   MAE  : {mae:.4f}°C")
    print(f"   R²   : {r2:.4f}")
    print(f"   MAPE : {mape:.2f}%")
    
    # 10. Sauvegarde
    save_dir = Path('models/cnn_lstm_v2_full')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n💾 Sauvegarde...")
    model.save(save_dir / 'cnn_lstm_v2_model.h5')
    print(f"   ✅ Modèle: {save_dir / 'cnn_lstm_v2_model.h5'}")
    
    # Métriques
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'n_features': len(feature_names),
        'seq_length': seq_length,
        'n_params': int(model.count_params())
    }
    pd.DataFrame([metrics]).to_csv(save_dir / 'cnn_lstm_v2_metrics.csv', index=False)
    print(f"   ✅ Métriques: {save_dir / 'cnn_lstm_v2_metrics.csv'}")
    
    # Historique
    with open(save_dir / 'cnn_lstm_v2_history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    print(f"   ✅ Historique: {save_dir / 'cnn_lstm_v2_history.json'}")
    
    # Features
    with open(save_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Scaler et Imputer
    with open(save_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(save_dir / 'imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    
    # 11. Comparaison
    print("\n" + "="*70)
    print("📊 COMPARAISON AVEC MODÈLES EXISTANTS")
    print("="*70)
    
    lstm_rmse = 6.2019
    cnn_lstm_v1_rmse = 11.2337
    
    print(f"   LSTM original (62 features)    : {lstm_rmse:.4f}°C")
    print(f"   CNN-LSTM v1 (11 RAW)           : {cnn_lstm_v1_rmse:.4f}°C")
    print(f"   CNN-LSTM v2 ({len(feature_names)} features)      : {rmse:.4f}°C")
    
    if rmse < lstm_rmse:
        improvement = ((lstm_rmse - rmse) / lstm_rmse) * 100
        print(f"   ✅ Amélioration vs LSTM        : +{improvement:.2f}%")
    else:
        degradation = ((rmse - lstm_rmse) / lstm_rmse) * 100
        print(f"   ⚠️ Dégradation vs LSTM         : -{degradation:.2f}%")
    
    # 12. Courbes d'apprentissage
    print("\n📈 Génération courbes d'apprentissage...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('CNN-LSTM v2 - Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE (°C)', fontsize=12)
    axes[1].set_title('CNN-LSTM v2 - Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ Graphique: {save_dir / 'training_curves.png'}")
    
    print("\n" + "="*70)
    print("✅ CNN-LSTM v2 FULL TERMINÉ !")
    print("="*70)
    print(f"\n🎯 RMSE Final: {rmse:.4f}°C avec {len(feature_names)} features")

if __name__ == '__main__':
    main()
