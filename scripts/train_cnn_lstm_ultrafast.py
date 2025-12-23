"""
CNN-LSTM ULTRA-OPTIMISÉ - Version Production Ultra-Rapide
Entraînement en 10-15 minutes max
"""

# Imports TensorFlow en premier pour éviter ralentissements
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n" + "="*80)
print("🚀 CNN-LSTM ULTRA-OPTIMIZED - Fast Training")
print("="*80 + "\n")

# Paths
base_path = Path(__file__).parent.parent
data_path = base_path / 'data' / 'processed' / 'splits'
model_path = base_path / 'models' / 'cnn_lstm_optimized'
model_path.mkdir(parents=True, exist_ok=True)

# 1. Load data (small sample)
print("📂 Loading data (stratified sample)...")
train_full = pd.read_parquet(data_path / 'train.parquet')
val_full = pd.read_parquet(data_path / 'val.parquet')
test_full = pd.read_parquet(data_path / 'test.parquet')

# Sample: 50K train, 10K val, 5K test
np.random.seed(42)
train_idx = np.random.choice(len(train_full), size=min(50000, len(train_full)), replace=False)
val_idx = np.random.choice(len(val_full), size=min(10000, len(val_full)), replace=False)
test_idx = np.random.choice(len(test_full), size=min(5000, len(test_full)), replace=False)

train = train_full.iloc[train_idx].reset_index(drop=True)
val = val_full.iloc[val_idx].reset_index(drop=True)
test = test_full.iloc[test_idx].reset_index(drop=True)

print(f"   Train: {train.shape}")
print(f"   Val: {val.shape}")
print(f"   Test: {test.shape}")

# 2. Select RAW features
print("\n🔧 Selecting RAW features (no lags, no rolling)...")
weather_features = ['humidity', 'wind_speed', 'wind_direction', 'pressure', 
                   'dewpoint', 'precipitation', 'cloud_cover']
temporal_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'day_of_week_sin', 'day_of_week_cos', 
                    'day_of_year_sin', 'day_of_year_cos']

raw_features = [f for f in weather_features + temporal_features if f in train.columns]

X_train = train[raw_features].values
y_train = train['temperature'].values
X_val = val[raw_features].values
y_val = val['temperature'].values
X_test = test[raw_features].values
y_test = test['temperature'].values

print(f"   RAW features: {len(raw_features)}")
print(f"   Features: {raw_features}")

# 3. Preprocessing
print("\n🔧 Preprocessing...")
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train = scaler.fit_transform(imputer.fit_transform(X_train))
X_val = scaler.transform(imputer.transform(X_val))
X_test = scaler.transform(imputer.transform(X_test))

# 4. Create sequences
print("\n📊 Creating sequences (12h window for speed)...")
seq_length = 12  # Réduit pour vitesse

def create_sequences(X, y, seq_len):
    n = len(X) - seq_len
    X_seq = np.zeros((n, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        X_seq[i] = X[i:i + seq_len]
        y_seq[i] = y[i + seq_len]
    
    return X_seq, y_seq

X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

print(f"   Train: {X_train_seq.shape}")
print(f"   Val: {X_val_seq.shape}")
print(f"   Test: {X_test_seq.shape}")

# 5. Build lightweight model
print("\n🏗️ Building ultra-lightweight CNN-LSTM...")
n_features = X_train_seq.shape[2]

model = Sequential([
    # Single CNN block
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(seq_length, n_features)),
    BatchNormalization(),
    MaxPooling1D(2),
    
    # Single LSTM
    LSTM(32),
    Dropout(0.2),
    
    # Dense
    Dense(16, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

print(f"   Parameters: {model.count_params():,}")

# 6. Train
print("\n" + "="*80)
print("🚀 TRAINING (30 epochs max, fast convergence)")
print("="*80 + "\n")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=30,
    batch_size=512,  # Large batch for speed
    callbacks=callbacks,
    verbose=1
)

# 7. Evaluate
print("\n" + "="*80)
print("📊 EVALUATION")
print("="*80 + "\n")

predictions = model.predict(X_test_seq, verbose=0).flatten()

mse = mean_squared_error(y_test_seq, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_seq, predictions)
r2 = r2_score(y_test_seq, predictions)
mape = np.mean(np.abs((y_test_seq - predictions) / (y_test_seq + 1e-10))) * 100

print(f"   RMSE : {rmse:.4f}°C")
print(f"   MAE  : {mae:.4f}°C")
print(f"   R²   : {r2:.4f}")
print(f"   MAPE : {mape:.2f}%")

# 8. Save
print("\n💾 Saving...")

model.save(model_path / 'cnn_lstm_model.h5')
print(f"   ✅ Model: cnn_lstm_model.h5")

metrics = {'MSE': float(mse), 'RMSE': float(rmse), 'MAE': float(mae), 'R2': float(r2), 'MAPE': float(mape)}
pd.DataFrame([metrics]).to_csv(model_path / 'cnn_lstm_metrics.csv', index=False)
print(f"   ✅ Metrics: cnn_lstm_metrics.csv")

with open(model_path / 'cnn_lstm_history.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

with open(model_path / 'feature_names.json', 'w') as f:
    json.dump(raw_features, f, indent=2)

with open(model_path / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(model_path / 'imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)

# 9. Plot
print("\n📈 Generating plots...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

epochs = range(1, len(history.history['loss']) + 1)

ax1.plot(epochs, history.history['loss'], 'b-', lw=2, label='Train')
ax1.plot(epochs, history.history['val_loss'], 'r-', lw=2, label='Val')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('CNN-LSTM Optimized - Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, history.history['mae'], 'b-', lw=2, label='Train')
ax2.plot(epochs, history.history['val_mae'], 'r-', lw=2, label='Val')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('MAE (°C)', fontsize=12)
ax2.set_title('CNN-LSTM Optimized - MAE', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(model_path / 'training_curves.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Plot: training_curves.png")

# 10. Compare
print("\n" + "="*80)
print("📊 COMPARISON")
print("="*80 + "\n")

# LSTM original
lstm_path = base_path / 'models' / 'lstm' / 'lstm_metrics.csv'
if lstm_path.exists():
    lstm_df = pd.read_csv(lstm_path)
    lstm_rmse = lstm_df['RMSE'].values[0]
    improvement = ((lstm_rmse - rmse) / lstm_rmse) * 100
    factor = lstm_rmse / rmse
    
    print(f"   LSTM original (62 features) : {lstm_rmse:.4f}°C")
    print(f"   CNN-LSTM (RAW features)     : {rmse:.4f}°C")
    print(f"   Improvement                 : {improvement:.2f}%")
    print(f"   Factor                      : {factor:.1f}x better")
    
    if rmse < lstm_rmse:
        print(f"\n   🎉 SUCCESS! CNN-LSTM beats original LSTM")

# Linear Reg
linear_path = base_path / 'models' / 'baseline' / 'linear_regression_metrics.csv'
if linear_path.exists():
    linear_df = pd.read_csv(linear_path)
    linear_rmse = linear_df['RMSE'].values[0]
    
    print(f"\n   Linear Regression           : {linear_rmse:.4f}°C (baseline)")
    print(f"   CNN-LSTM                    : {rmse:.4f}°C")
    
    ratio = rmse / linear_rmse
    if ratio < 3:
        print(f"   ✅ Competitive ({ratio:.1f}x Linear Reg)")
    if ratio < 2:
        print(f"   ⭐ Very good ({ratio:.1f}x Linear Reg)")
    if ratio < 1.5:
        print(f"   🏆 Excellent ({ratio:.1f}x Linear Reg)")

print("\n" + "="*80)
print("✅ CNN-LSTM TRAINING COMPLETED!")
print("="*80 + "\n")

print(f"📁 Results in: {model_path}")
print(f"   Final RMSE: {rmse:.4f}°C")
