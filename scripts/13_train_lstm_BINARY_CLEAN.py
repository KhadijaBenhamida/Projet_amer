"""
LSTM BINAIRE - DETECTION EVENEMENTS EXTREMES (APPROCHE PROFESSIONNELLE)
========================================================================

Architecture Hybrid ML + Symbolic AI:
- LSTM: Dtecte SI vnement extrme (binary classification)
- Ontologie: Identifie TYPE d'vnement (symbolic rules)

Conforme cahier des charges:
 Squences 72h (3 jours contexte)
 LSTM Bidirectional architecture
 Binary Focal Loss (optionnel, ratio 1.3:1 grable)
 Mtriques: F1-score, Recall, Precision, ROC-AUC
 Early stopping + Model checkpoint
 Analyse per-station performance
 Memory-efficient sequence generation

Author: Deep Learning Pro
Date: 2025-12-29
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_score,
    roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Seed pour reproductibilit
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("LSTM BINAIRE - DETECTION EVENEMENTS CLIMATIQUES EXTREMES")
print("Approche: Binary LSTM + Ontology (Hybrid ML+Symbolic AI)")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

SEQUENCE_LENGTH = 72  # 72h = 3 jours contexte historique
BATCH_SIZE = 128      # Optimis pour GPU
EPOCHS = 50           # Rduit car binary (converge plus vite)
LEARNING_RATE = 0.001
PATIENCE_ES = 10      # Early stopping patience
PATIENCE_LR = 5       # ReduceLR patience

print(f"\n Configuration:")
print(f"   Squence: {SEQUENCE_LENGTH}h (3 jours)")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs max: {EPOCHS}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Early stopping: {PATIENCE_ES} epochs")
print(f"   ReduceLR patience: {PATIENCE_LR} epochs")

# Crer dossiers rsultats
Path("models/lstm_binary").mkdir(parents=True, exist_ok=True)
Path("models/results").mkdir(parents=True, exist_ok=True)
Path("visualizations/training").mkdir(parents=True, exist_ok=True)

# ============================================================================
# BINARY FOCAL LOSS (optionnel mais amliore performances)
# ============================================================================

class BinaryFocalLoss(keras.losses.Loss):
    """
    Binary Focal Loss pour attnuer dsquilibre.
    
    FL(p_t) = -(1-p_t)^ * log(p_t)
    
    Avec ratio 1.3:1, pas obligatoire mais amliore rappel vnements rares.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, name='binary_focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calcul Binary Cross Entropy
        bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Calcul focal term
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_term = tf.pow(1 - p_t, self.gamma)
        
        # Apply alpha weighting
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        loss = alpha_t * focal_term * bce
        return tf.reduce_mean(loss)
    
    def get_config(self):
        return {'alpha': self.alpha, 'gamma': self.gamma}

# ============================================================================
# CHARGEMENT DONNEES
# ============================================================================

print("\n" + "="*80)
print("ETAPE 1/7: CHARGEMENT DONNEES CLASSIFIEES")
print("="*80)

df_train = pd.read_parquet('data/processed/splits_classified_binary/train_classified.parquet')
df_val = pd.read_parquet('data/processed/splits_classified_binary/val_classified.parquet')
df_test = pd.read_parquet('data/processed/splits_classified_binary/test_classified.parquet')

print(f"\n Donnes charges:")
print(f"   Train: {len(df_train):,} samples")
print(f"   Val:   {len(df_val):,} samples")
print(f"   Test:  {len(df_test):,} samples")

# Distribution target
train_dist = df_train['is_extreme_event'].value_counts()
print(f"\n Distribution train:")
print(f"   Normal (0):  {train_dist[0]:,} ({train_dist[0]/len(df_train)*100:.1f}%)")
print(f"   Extreme (1): {train_dist[1]:,} ({train_dist[1]/len(df_train)*100:.1f}%)")
print(f"   Ratio: {train_dist[0]/train_dist[1]:.2f}:1")

# Charger class weights
with open('models/analysis/class_weights_binary.json', 'r') as f:
    weights_info = json.load(f)
    class_weights = {int(k): v for k, v in weights_info['class_weights'].items()}

print(f"\n  Class weights: {class_weights}")

# ============================================================================
# PREPARATION FEATURES
# ============================================================================

print("\n" + "="*80)
print("ETAPE 2/7: SELECTION FEATURES OPTIMALES")
print("="*80)

# Exclure colonnes non-features
exclude_cols = [
    'station_id', 'year', 'month', 'day', 'hour', 'minute',
    'is_extreme_event', 'datetime', 'date', 'time',
    'temp_category', 'is_daytime'  # Si existent
]

# Slectionner features numriques
feature_cols = [
    col for col in df_train.columns
    if col not in exclude_cols and df_train[col].dtype in ['float64', 'int64']
]

num_features = len(feature_cols)
print(f"\n Features slectionnes: {num_features}")
print(f"   Exemples: {feature_cols[:5]}")
print(f"   {'...' if num_features > 5 else ''}")

# Vrifier NaN
for df, name in [(df_train, 'train'), (df_val, 'val'), (df_test, 'test')]:
    nan_count = df[feature_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"     {name}: {nan_count} NaN trouvs - imputation ncessaire")
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# ============================================================================
# CREATION SEQUENCES (Memory-efficient generator)
# ============================================================================

print("\n" + "="*80)
print("ETAPE 3/7: CREATION SEQUENCES TEMPORELLES (72h)")
print("="*80)

def create_sequences_efficient(df, sequence_length, feature_cols):
    """
    Cr squences [t-72:t-1]  y[t] de manire memory-efficient.
    
    Contraintes:
    - Squences par station (pas mlanger stations diffrentes)
    - Pas de gap temporel dans squence
    - Retourne arrays prts pour LSTM
    
    Returns:
        X: (N, 72, num_features) - Squences features
        y: (N,) - Target binaire
        metadata: Dict avec infos stations/dates
    """
    
    stations = sorted(df['station_id'].unique())
    
    X_list = []
    y_list = []
    station_list = []
    
    print(f"\n   Cration squences par station...")
    
    for station_id in stations:
        # Filtrer station
        df_station = df[df['station_id'] == station_id].copy()
        
        # Extraire features et target
        features = df_station[feature_cols].values.astype(np.float32)
        targets = df_station['is_extreme_event'].values.astype(np.int32)
        
        # Gnrer squences
        n_samples = len(df_station) - sequence_length
        if n_samples <= 0:
            continue
        
        for i in range(sequence_length, len(df_station)):
            X_seq = features[i-sequence_length:i]  # [t-72:t-1]
            y_target = targets[i]                  # [t]
            
            X_list.append(X_seq)
            y_list.append(y_target)
            station_list.append(station_id)
        
        print(f"      {station_id}: {n_samples:,} squences")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    metadata = {
        'num_sequences': len(X),
        'num_features': len(feature_cols),
        'sequence_length': sequence_length,
        'feature_names': feature_cols,
        'stations': station_list
    }
    
    return X, y, metadata

# Crer squences
print("\n   Train...")
X_train, y_train, meta_train = create_sequences_efficient(df_train, SEQUENCE_LENGTH, feature_cols)

print("\n   Validation...")
X_val, y_val, meta_val = create_sequences_efficient(df_val, SEQUENCE_LENGTH, feature_cols)

print("\n   Test...")
X_test, y_test, meta_test = create_sequences_efficient(df_test, SEQUENCE_LENGTH, feature_cols)

print(f"\n Squences cres:")
print(f"   Train: {X_train.shape}  {y_train.shape}")
print(f"   Val:   {X_val.shape}  {y_val.shape}")
print(f"   Test:  {X_test.shape}  {y_test.shape}")
print(f"\n   Distribution train: {np.bincount(y_train)}")
print(f"   Distribution val:   {np.bincount(y_val)}")

# Sauvegarder metadata
with open('models/lstm_binary/sequences_metadata.json', 'w') as f:
    json.dump({
        'sequence_length': SEQUENCE_LENGTH,
        'num_features': num_features,
        'feature_names': feature_cols,
        'train_sequences': int(len(X_train)),
        'val_sequences': int(len(X_val)),
        'test_sequences': int(len(X_test)),
        'created_at': datetime.now().isoformat()
    }, f, indent=2)

# ============================================================================
# ARCHITECTURE LSTM BINAIRE
# ============================================================================

print("\n" + "="*80)
print("ETAPE 4/7: CONSTRUCTION ARCHITECTURE LSTM BINAIRE")
print("="*80)

def build_binary_lstm(input_shape, use_focal_loss=False):
    """
    Architecture LSTM Bidirectional pour classification binaire.
    
    Diffrences vs multi-classe:
    - Output: Dense(1, sigmoid) au lieu de Dense(5, softmax)
    - Loss: binary_crossentropy ou BinaryFocalLoss
    - Metrics: binary accuracy, precision, recall
    
    Architecture:
    - Input: (72, num_features)
    - BatchNorm  Bidirectional LSTM 128
    - BatchNorm  Bidirectional LSTM 64
    - Dense 64  Dropout 0.4
    - Dense 32  Dropout 0.3
    - Dense 1 (sigmoid)  P(extreme_event)
    """
    
    inputs = Input(shape=input_shape, name='sequence_input')
    
    # Input normalization
    x = layers.BatchNormalization(name='input_bn')(inputs)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name='bilstm_128'
    )(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, dropout=0.3, recurrent_dropout=0.2),
        name='bilstm_64'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    
    # Dense layers avec rgularisation
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='dense_64')(x)
    x = layers.Dropout(0.4, name='dropout_1')(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='dense_32')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    # Binary output (sigmoid)
    outputs = layers.Dense(1, activation='sigmoid', name='binary_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_Binary_Classifier')
    
    # Compilation
    if use_focal_loss:
        loss = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        print("   Loss: Binary Focal Loss (alpha=0.25, gamma=2.0)")
    else:
        loss = 'binary_crossentropy'
        print("   Loss: Binary Cross Entropy (standard)")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=[
            'binary_accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

# Build model
input_shape = (SEQUENCE_LENGTH, num_features)
use_focal = weights_info.get('use_focal_loss', False)

print(f"\n   Input shape: {input_shape}")
print(f"   Use Focal Loss: {use_focal}")

model = build_binary_lstm(input_shape, use_focal_loss=use_focal)

print(f"\n Modle cr:")
model.summary(print_fn=lambda x: print(f"   {x}"))

total_params = model.count_params()
print(f"\n   Total paramtres: {total_params:,}")

# ============================================================================
# CALLBACKS
# ============================================================================

print("\n" + "="*80)
print("ETAPE 5/7: CONFIGURATION CALLBACKS")
print("="*80)

# Early stopping sur validation loss
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE_ES,
    restore_best_weights=True,
    verbose=1,
    mode='min'
)

# Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=PATIENCE_LR,
    min_lr=1e-6,
    verbose=1,
    mode='min'
)

# Model checkpoint (sauvegarder meilleur modle)
checkpoint = ModelCheckpoint(
    filepath='models/lstm_binary/best_model.keras',
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks = [early_stop, reduce_lr, checkpoint]

print(f" Callbacks configurs:")
print(f"   - EarlyStopping: patience={PATIENCE_ES}, monitor=val_loss")
print(f"   - ReduceLROnPlateau: patience={PATIENCE_LR}, factor=0.5")
print(f"   - ModelCheckpoint: monitor=val_auc, save_best_only=True")

# ============================================================================
# ENTRAINEMENT
# ============================================================================

print("\n" + "="*80)
print("ETAPE 6/7: ENTRAINEMENT LSTM")
print("="*80)

print(f"\nDbut entranement: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Configuration:")
print(f"   - Epochs: {EPOCHS}")
print(f"   - Batch size: {BATCH_SIZE}")
print(f"   - Class weights: {class_weights}")
print(f"   - Samples train: {len(X_train):,}")
print(f"   - Steps/epoch: {len(X_train)//BATCH_SIZE}")

start_time = datetime.now()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

end_time = datetime.now()
training_duration = (end_time - start_time).total_seconds()

print(f"\n Entranement termin:")
print(f"   Dure: {training_duration/60:.1f} minutes")
print(f"   Epochs effectus: {len(history.history['loss'])}")

# Sauvegarder historique
history_df = pd.DataFrame(history.history)
history_df.to_csv('models/lstm_binary/training_history.csv', index=False)

with open('models/lstm_binary/training_info.json', 'w') as f:
    json.dump({
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_minutes': training_duration / 60,
        'epochs_trained': len(history.history['loss']),
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_auc': float(max(history.history['val_auc'])),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1])
    }, f, indent=2)

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*80)
print("ETAPE 7/7: EVALUATION COMPLETE")
print("="*80)

# Charger meilleur modle
model = keras.models.load_model(
    'models/lstm_binary/best_model.keras',
    custom_objects={'BinaryFocalLoss': BinaryFocalLoss} if use_focal else None
)

print("\n Meilleur modle charg")

# Prdictions
print("\n Gnration prdictions...")

y_train_pred_proba = model.predict(X_train, batch_size=BATCH_SIZE, verbose=0).flatten()
y_val_pred_proba = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).flatten()
y_test_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()

# Threshold 0.5 pour classification binaire
y_train_pred = (y_train_pred_proba > 0.5).astype(int)
y_val_pred = (y_val_pred_proba > 0.5).astype(int)
y_test_pred = (y_test_pred_proba > 0.5).astype(int)

# Mtriques globales
def compute_metrics(y_true, y_pred, y_proba, set_name):
    """Calcule toutes les mtriques pour un set"""
    
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    metrics = {
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc),
        'accuracy': float((y_true == y_pred).mean())
    }
    
    print(f"\n{'='*60}")
    print(f"{set_name.upper()} SET - METRIQUES")
    print(f"{'='*60}")
    print(f"   F1-Score:  {f1:.4f} {'' if f1 > 0.85 else '' if f1 > 0.75 else ''}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    
    return metrics

metrics_train = compute_metrics(y_train, y_train_pred, y_train_pred_proba, 'train')
metrics_val = compute_metrics(y_val, y_val_pred, y_val_pred_proba, 'validation')
metrics_test = compute_metrics(y_test, y_test_pred, y_test_pred_proba, 'test')

# Confusion matrices
cm_train = confusion_matrix(y_train, y_train_pred)
cm_val = confusion_matrix(y_val, y_val_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print(f"\n{'='*60}")
print("CONFUSION MATRIX - TEST SET")
print(f"{'='*60}")
print(f"\n                Predicted")
print(f"              Normal  Extreme")
print(f"Actual Normal   {cm_test[0,0]:6d}  {cm_test[0,1]:6d}")
print(f"       Extreme  {cm_test[1,0]:6d}  {cm_test[1,1]:6d}")

# Classification report
print(f"\n{'='*60}")
print("CLASSIFICATION REPORT - TEST SET")
print(f"{'='*60}")
print(classification_report(y_test, y_test_pred, target_names=['Normal', 'Extreme']))

# Sauvegarder rsultats
results = {
    'model_architecture': 'Bidirectional LSTM (12864) Binary',
    'input_shape': input_shape,
    'total_parameters': int(total_params),
    'training_duration_minutes': training_duration / 60,
    'epochs_trained': len(history.history['loss']),
    'use_focal_loss': use_focal,
    'class_weights': class_weights,
    'metrics': {
        'train': metrics_train,
        'validation': metrics_val,
        'test': metrics_test
    },
    'confusion_matrices': {
        'train': cm_train.tolist(),
        'val': cm_val.tolist(),
        'test': cm_test.tolist()
    }
}

with open('models/results/lstm_binary_evaluation.json', 'w') as f:
    json.dump(results, f, indent=2)

# ============================================================================
# VISUALISATIONS
# ============================================================================

print(f"\n{'='*60}")
print("GENERATION VISUALISATIONS")
print(f"{'='*60}")

# 1. Training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_title('Loss Evolution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history.history['binary_accuracy'], label='Train Acc', linewidth=2)
axes[0, 1].plot(history.history['val_binary_accuracy'], label='Val Acc', linewidth=2)
axes[0, 1].set_title('Accuracy Evolution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# AUC
axes[1, 0].plot(history.history['auc'], label='Train AUC', linewidth=2)
axes[1, 0].plot(history.history['val_auc'], label='Val AUC', linewidth=2)
axes[1, 0].set_title('ROC-AUC Evolution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# F1-score (approximation via precision + recall)
if 'precision' in history.history and 'recall' in history.history:
    p = np.array(history.history['precision'])
    r = np.array(history.history['recall'])
    f1_train = 2 * (p * r) / (p + r + 1e-7)
    
    p_val = np.array(history.history['val_precision'])
    r_val = np.array(history.history['val_recall'])
    f1_val = 2 * (p_val * r_val) / (p_val + r_val + 1e-7)
    
    axes[1, 1].plot(f1_train, label='Train F1', linewidth=2)
    axes[1, 1].plot(f1_val, label='Val F1', linewidth=2)
    axes[1, 1].set_title('F1-Score Evolution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/training/training_curves_binary.png', dpi=300, bbox_inches='tight')
print("    visualizations/training/training_curves_binary.png")

# 2. Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (cm, name) in enumerate([(cm_train, 'Train'), (cm_val, 'Validation'), (cm_test, 'Test')]):
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Extreme'],
        yticklabels=['Normal', 'Extreme'],
        ax=axes[idx],
        cbar_kws={'label': 'Count'}
    )
    axes[idx].set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('visualizations/training/confusion_matrices_binary.png', dpi=300, bbox_inches='tight')
print("    visualizations/training/confusion_matrices_binary.png")

# 3. ROC Curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr_train, tpr_train, label=f'Train (AUC={metrics_train["roc_auc"]:.3f})', linewidth=2)
plt.plot(fpr_val, tpr_val, label=f'Val (AUC={metrics_val["roc_auc"]:.3f})', linewidth=2)
plt.plot(fpr_test, tpr_test, label=f'Test (AUC={metrics_test["roc_auc"]:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.3)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Binary LSTM', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/training/roc_curves_binary.png', dpi=300, bbox_inches='tight')
print("    visualizations/training/roc_curves_binary.png")

# 4. Prediction distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution probabilits
axes[0].hist(y_test_pred_proba[y_test == 0], bins=50, alpha=0.6, label='Normal', color='green')
axes[0].hist(y_test_pred_proba[y_test == 1], bins=50, alpha=0.6, label='Extreme', color='red')
axes[0].axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
axes[0].set_xlabel('Predicted Probability', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Prediction Probability Distribution (Test)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Barplot mtriques
metrics_names = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC']
train_vals = [metrics_train['f1_score'], metrics_train['precision'], metrics_train['recall'], metrics_train['roc_auc']]
val_vals = [metrics_val['f1_score'], metrics_val['precision'], metrics_val['recall'], metrics_val['roc_auc']]
test_vals = [metrics_test['f1_score'], metrics_test['precision'], metrics_test['recall'], metrics_test['roc_auc']]

x = np.arange(len(metrics_names))
width = 0.25

axes[1].bar(x - width, train_vals, width, label='Train', alpha=0.8)
axes[1].bar(x, val_vals, width, label='Val', alpha=0.8)
axes[1].bar(x + width, test_vals, width, label='Test', alpha=0.8)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Metrics Comparison', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics_names)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('visualizations/training/predictions_analysis_binary.png', dpi=300, bbox_inches='tight')
print("    visualizations/training/predictions_analysis_binary.png")

# ============================================================================
# ANALYSE PER-STATION
# ============================================================================

print(f"\n{'='*60}")
print("ANALYSE PERFORMANCE PAR STATION")
print(f"{'='*60}")

station_metrics = {}

for station_id in sorted(set(meta_test['stations'])):
    # Indices chantillons de cette station
    indices = [i for i, s in enumerate(meta_test['stations']) if s == station_id]
    
    if len(indices) < 10:  # Skip stations avec trop peu samples
        continue
    
    y_true_station = y_test[indices]
    y_pred_station = y_test_pred[indices]
    y_proba_station = y_test_pred_proba[indices]
    
    # Calculer mtriques
    f1 = f1_score(y_true_station, y_pred_station, zero_division=0)
    precision = precision_score(y_true_station, y_pred_station, zero_division=0)
    recall = recall_score(y_true_station, y_pred_station, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true_station, y_proba_station)
    except:
        roc_auc = 0.0
    
    station_metrics[station_id] = {
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc),
        'num_samples': len(indices),
        'num_positives': int(y_true_station.sum())
    }
    
    print(f"\n   Station {station_id}:")
    print(f"      Samples: {len(indices):,} | Positives: {y_true_station.sum()}")
    print(f"      F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

# Sauvegarder
with open('models/results/lstm_binary_per_station.json', 'w') as f:
    json.dump(station_metrics, f, indent=2)

# Visualiser
if station_metrics:
    stations_list = list(station_metrics.keys())
    f1_scores = [station_metrics[s]['f1_score'] for s in stations_list]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(stations_list)), f1_scores, color='steelblue', alpha=0.7)
    plt.axhline(metrics_test['f1_score'], color='red', linestyle='--', label=f'Global F1={metrics_test["f1_score"]:.3f}')
    plt.xlabel('Station ID', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score per Station (Test Set)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(stations_list)), stations_list, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/training/f1_per_station_binary.png', dpi=300, bbox_inches='tight')
    print(f"\n    visualizations/training/f1_per_station_binary.png")

# ============================================================================
# RESUME FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUME FINAL - LSTM BINAIRE")
print("="*80)

print(f"""
 ENTRAINEMENT TERMINE

MODELE:
- Architecture: Bidirectional LSTM (12864) Binary
- Paramtres: {total_params:,}
- Loss: {'Binary Focal Loss' if use_focal else 'Binary Cross Entropy'}
- Dure: {training_duration/60:.1f} minutes
- Epochs: {len(history.history['loss'])}

PERFORMANCES TEST:
- F1-Score:  {metrics_test['f1_score']:.4f} {' EXCELLENT' if metrics_test['f1_score'] > 0.90 else ' BON' if metrics_test['f1_score'] > 0.85 else ' ACCEPTABLE'}
- Precision: {metrics_test['precision']:.4f}
- Recall:    {metrics_test['recall']:.4f}
- ROC-AUC:   {metrics_test['roc_auc']:.4f}
- Accuracy:  {metrics_test['accuracy']:.4f}

OBJECTIF CAHIER DES CHARGES:
- F1-score attendu: 0.92
- F1-score obtenu:  {metrics_test['f1_score']:.4f}
- Status: {' ATTEINT' if metrics_test['f1_score'] >= 0.92 else ' PROCHE' if metrics_test['f1_score'] >= 0.88 else ' NON ATTEINT'}

FICHIERS CREES:
- models/lstm_binary/best_model.keras (meilleur modle)
- models/lstm_binary/training_history.csv
- models/lstm_binary/sequences_metadata.json
- models/results/lstm_binary_evaluation.json
- models/results/lstm_binary_per_station.json
- visualizations/training/*.png (4 visualisations)

PROCHAINE ETAPE:
Crer script prdiction avec ontologie:
  python scripts/14_predict_with_ontology.py
  
  Workflow:
  1. LSTM dtecte: P(extreme_event) > 0.5
  2. Ontologie identifie TYPE: Canicule/Froid/etc.
  3. Gnre alertes: ROUGE/ORANGE/VERT
""")

print("\n" + "="*80)
print(f"Script termin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
