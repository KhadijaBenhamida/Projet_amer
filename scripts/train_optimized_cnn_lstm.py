"""
CNN-LSTM OPTIMISÉ - Version Production
Entraînement rapide avec échantillon stratifié + architecture optimisée
Target: RMSE < 0.5°C (15-30x meilleur que LSTM actuel)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import logging
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def select_raw_features(df: pd.DataFrame):
    """Sélectionne uniquement features RAW (pas de lags, pas de rolling stats)"""
    # Features météo brutes
    weather = ['humidity', 'wind_speed', 'wind_direction', 'pressure', 
               'dewpoint', 'precipitation', 'cloud_cover']
    
    # Features temporelles cycliques
    temporal = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'day_of_week_sin', 'day_of_week_cos', 
                'day_of_year_sin', 'day_of_year_cos']
    
    # Sélectionner features disponibles
    selected = [f for f in weather + temporal if f in df.columns]
    
    return df[selected], selected

def create_sequences(X, y, seq_length=24):
    """Crée séquences 3D avec sliding window optimisé"""
    n = len(X) - seq_length
    X_seq = np.zeros((n, seq_length, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        X_seq[i] = X[i:i + seq_length]
        y_seq[i] = y[i + seq_length]
    
    return X_seq, y_seq

def build_optimized_model(seq_length, n_features):
    """Architecture CNN-LSTM optimisée pour convergence rapide"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        # CNN Block 1 - Patterns locaux
        Conv1D(32, kernel_size=3, activation='relu', 
               input_shape=(seq_length, n_features)),
        BatchNormalization(),
        MaxPooling1D(2),
        
        # CNN Block 2 - Patterns niveau supérieur
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        
        # LSTM - Patterns temporels
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.0005)  # LR optimal
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def main():
    print("\n" + "="*80)
    print("🚀 CNN-LSTM OPTIMIZED - Temperature Prediction")
    print("="*80 + "\n")
    
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data' / 'processed' / 'splits'
    model_path = base_path / 'models' / 'cnn_lstm_optimized'
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Charger échantillon stratifié (plus rapide)
    print("📂 Chargement données (échantillon stratifié)...")
    
    # Charger et échantillonner stratifié par mois
    train_full = pd.read_parquet(data_path / 'train.parquet')
    val_full = pd.read_parquet(data_path / 'val.parquet')
    test_full = pd.read_parquet(data_path / 'test.parquet')
    
    # Échantillonnage stratifié si colonnes temporelles disponibles
    if 'month' in train_full.columns:
        train = train_full.groupby('month', group_keys=False).apply(
            lambda x: x.sample(min(len(x), 8000), random_state=42)
        ).reset_index(drop=True)
        val = val_full.groupby('month', group_keys=False).apply(
            lambda x: x.sample(min(len(x), 2000), random_state=42)
        ).reset_index(drop=True)
    else:
        train = train_full.sample(n=80000, random_state=42)
        val = val_full.sample(n=15000, random_state=42)
    
    test = test_full.sample(n=10000, random_state=42)
    
    print(f"   Train: {train.shape} (échantillon)")
    print(f"   Val: {val.shape}")
    print(f"   Test: {test.shape}")
    
    # 2. Séparer X et y
    X_train_full = train.drop('temperature', axis=1)
    y_train = train['temperature'].values
    X_val_full = val.drop('temperature', axis=1)
    y_val = val['temperature'].values
    X_test_full = test.drop('temperature', axis=1)
    y_test = test['temperature'].values
    
    # 3. Sélectionner features RAW
    print("\n🔧 Sélection features RAW (pas de lags, pas de rolling stats)...")
    X_train_raw, feature_names = select_raw_features(X_train_full)
    X_val_raw, _ = select_raw_features(X_val_full)
    X_test_raw, _ = select_raw_features(X_test_full)
    
    print(f"   Features RAW: {len(feature_names)}")
    print(f"   Liste: {feature_names}")
    
    # 4. Preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    print("\n🔧 Preprocessing...")
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_val_scaled = scaler.transform(imputer.transform(X_val_raw))
    X_test_scaled = scaler.transform(imputer.transform(X_test_raw))
    
    n_features = X_train_scaled.shape[1]
    
    # 5. Créer séquences (24h = 1 jour de contexte)
    print("\n📊 Création séquences temporelles (24h)...")
    seq_length = 24
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length)
    
    print(f"   Train sequences: {X_train_seq.shape}")
    print(f"   Val sequences: {X_val_seq.shape}")
    print(f"   Test sequences: {X_test_seq.shape}")
    
    # 6. Construire modèle
    print("\n🏗️ Construction CNN-LSTM optimisé...")
    model = build_optimized_model(seq_length, n_features)
    
    print(f"   Paramètres: {model.count_params():,}")
    print(f"   Architecture: Conv1D(32) → BN → Pool → Conv1D(64) → BN → Pool → LSTM(32) → Dense")
    
    # 7. Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # 8. Entraîner
    print("\n" + "="*80)
    print("🚀 ENTRAÎNEMENT (50 epochs max, early stopping)")
    print("="*80 + "\n")
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )
    
    # 9. Évaluer
    print("\n" + "="*80)
    print("📊 ÉVALUATION")
    print("="*80 + "\n")
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    predictions = model.predict(X_test_seq, verbose=0).flatten()
    
    mse = mean_squared_error(y_test_seq, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_seq, predictions)
    r2 = r2_score(y_test_seq, predictions)
    mape = np.mean(np.abs((y_test_seq - predictions) / (y_test_seq + 1e-10))) * 100
    
    print(f"   MSE  : {mse:.4f}")
    print(f"   RMSE : {rmse:.4f}°C")
    print(f"   MAE  : {mae:.4f}°C")
    print(f"   R²   : {r2:.4f}")
    print(f"   MAPE : {mape:.2f}%")
    
    # 10. Sauvegarder
    print("\n💾 Sauvegarde...")
    
    model.save(model_path / 'cnn_lstm_model.h5')
    print(f"   ✅ Modèle: {model_path / 'cnn_lstm_model.h5'}")
    
    # Métriques
    metrics = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'MAPE': float(mape)
    }
    
    pd.DataFrame([metrics]).to_csv(model_path / 'cnn_lstm_metrics.csv', index=False)
    print(f"   ✅ Métriques: {model_path / 'cnn_lstm_metrics.csv'}")
    
    # History
    with open(model_path / 'cnn_lstm_history.json', 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"   ✅ Historique: {model_path / 'cnn_lstm_history.json'}")
    
    # Preprocessing objects
    with open(model_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(model_path / 'imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    
    # Feature names
    with open(model_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # 11. Comparaison avec LSTM original
    print("\n" + "="*80)
    print("📊 COMPARAISON AVEC LSTM ORIGINAL")
    print("="*80 + "\n")
    
    lstm_path = base_path / 'models' / 'lstm' / 'lstm_metrics.csv'
    if lstm_path.exists():
        lstm_metrics = pd.read_csv(lstm_path)
        lstm_rmse = lstm_metrics['RMSE'].values[0]
        
        improvement = ((lstm_rmse - rmse) / lstm_rmse) * 100
        factor = lstm_rmse / rmse
        
        print(f"   LSTM original        : {lstm_rmse:.4f}°C")
        print(f"   CNN-LSTM optimisé    : {rmse:.4f}°C")
        print(f"   Amélioration         : {improvement:.2f}%")
        print(f"   Facteur              : {factor:.1f}x meilleur")
        
        if rmse < 1.0:
            print(f"\n   🎉 OBJECTIF ATTEINT ! RMSE < 1.0°C")
        if rmse < 0.5:
            print(f"   🏆 EXCELLENT ! RMSE < 0.5°C (compétitif avec Linear Reg)")
    
    # 12. Comparaison avec Linear Regression
    linear_path = base_path / 'models' / 'baseline' / 'linear_regression_metrics.csv'
    if linear_path.exists():
        linear_metrics = pd.read_csv(linear_path)
        linear_rmse = linear_metrics['RMSE'].values[0]
        
        print(f"\n   Linear Regression    : {linear_rmse:.4f}°C (baseline)")
        print(f"   CNN-LSTM optimisé    : {rmse:.4f}°C")
        
        if rmse < linear_rmse * 3:
            print(f"   ✅ CNN-LSTM compétitif (< 3x Linear Reg)")
        if rmse < linear_rmse * 2:
            print(f"   ⭐ CNN-LSTM très compétitif (< 2x Linear Reg)")
        if rmse < linear_rmse * 1.5:
            print(f"   🏆 CNN-LSTM excellent (< 1.5x Linear Reg)")
    
    # 13. Courbes d'apprentissage
    print("\n📈 Génération courbes d'apprentissage...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    # Loss
    ax1.plot(epochs_range, history.history['loss'], 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs_range, history.history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('CNN-LSTM Optimized - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # MAE
    ax2.plot(epochs_range, history.history['mae'], 'b-', linewidth=2, label='Train MAE')
    ax2.plot(epochs_range, history.history['val_mae'], 'r-', linewidth=2, label='Val MAE')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE (°C)', fontsize=12)
    ax2.set_title('CNN-LSTM Optimized - MAE', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Graphique: {model_path / 'training_curves.png'}")
    
    print("\n" + "="*80)
    print("✅ CNN-LSTM OPTIMISÉ TERMINÉ !")
    print("="*80 + "\n")
    
    return metrics

if __name__ == "__main__":
    main()
