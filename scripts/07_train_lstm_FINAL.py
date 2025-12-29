"""
ENTRAINEMENT LSTM - CLASSIFICATION EVENEMENTS EXTREMES
=======================================================

Script professionnel conforme cahier des charges:
- Séquences temporelles 72h (3 jours contexte)
- LSTM Bidirectional + Focal Loss
- Métriques: F1-score, Recall, Precision, ROC-AUC
- Early stopping + Model checkpoint
- Analyse détaillée performances par classe
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, recall_score, precision_score, 
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*80)
print("ENTRAINEMENT LSTM - PREDICTION EVENEMENTS CLIMATIQUES EXTREMES")
print("Version professionnelle - Conforme cahier des charges")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

SEQUENCE_LENGTH = 72  # 72 heures = 3 jours de contexte historique
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

print(f"\n📋 Configuration:")
print(f"   Séquence: {SEQUENCE_LENGTH}h (3 jours)")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs max: {EPOCHS}")
print(f"   Learning rate: {LEARNING_RATE}")

# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(keras.losses.Loss):
    """Focal Loss pour classes déséquilibrées"""
    
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_term = tf.pow(1 - p_t, self.gamma)
        loss = self.alpha * focal_term * ce
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    def get_config(self):
        return {'alpha': self.alpha, 'gamma': self.gamma}

# ============================================================================
# CHARGEMENT DONNEES
# ============================================================================

print("\n" + "="*80)
print("ETAPE 1: CHARGEMENT DONNEES CLASSIFIEES")
print("="*80)

df_train = pd.read_parquet('data/processed/splits_classified/train_classified.parquet')
df_val = pd.read_parquet('data/processed/splits_classified/val_classified.parquet')
df_test = pd.read_parquet('data/processed/splits_classified/test_classified.parquet')

print(f"\n✅ Train: {len(df_train):,} samples")
print(f"✅ Val:   {len(df_val):,} samples")
print(f"✅ Test:  {len(df_test):,} samples")

# Charger class weights
with open('models/analysis/class_weights.json') as f:
    weights_info = json.load(f)

class_weights = weights_info['class_weights']
use_focal = weights_info['use_focal_loss']
event_labels = weights_info['event_labels']
num_classes = weights_info['num_classes']

print(f"\n📊 Classes: {num_classes}")
for cls_str, label in event_labels.items():
    print(f"   {cls_str}: {label}")

print(f"\n⚖️  Focal Loss: {'OUI' if use_focal else 'NON'}")
print(f"   Ratio déséquilibre: {weights_info['imbalance_ratio']:.1f}:1")

# ============================================================================
# CREATION SEQUENCES TEMPORELLES
# ============================================================================

print("\n" + "="*80)
print("ETAPE 2: CREATION SEQUENCES TEMPORELLES (72h)")
print("="*80)

def create_sequences(df, sequence_length=72):
    """
    Crée séquences temporelles pour LSTM.
    
    Pour chaque point t:
    - X: séquence [t-72h, ..., t-1h] (72 timesteps, 62 features)
    - y: classe au temps t (0-4)
    
    Args:
        df: DataFrame avec datetime index et colonnes features + extreme_event
        sequence_length: Longueur séquence (72h par défaut)
    
    Returns:
        X: array (n_sequences, 72, 62)
        y: array (n_sequences,)
        metadata: dict avec infos
    """
    
    # S'assurer index datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('datetime')
    
    # Trier par station + datetime
    df = df.sort_values(['station_id', df.index.name])
    
    # Features (exclure target + identifiers)
    exclude_cols = ['extreme_event', 'station_id', 'datetime']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X_list = []
    y_list = []
    station_list = []
    datetime_list = []
    
    # Par station (éviter séquences cross-station)
    for station_id in df['station_id'].unique():
        df_station = df[df['station_id'] == station_id].copy()
        
        # Convertir en arrays
        features = df_station[feature_cols].values
        targets = df_station['extreme_event'].values
        datetimes = df_station.index.values
        
        # Créer séquences
        for i in range(sequence_length, len(df_station)):
            X_seq = features[i-sequence_length:i]  # [t-72:t-1]
            y_target = targets[i]  # [t]
            
            X_list.append(X_seq)
            y_list.append(y_target)
            station_list.append(station_id)
            datetime_list.append(datetimes[i])
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    metadata = {
        'num_sequences': len(X),
        'num_features': len(feature_cols),
        'sequence_length': sequence_length,
        'feature_names': feature_cols,
        'stations': station_list,
        'datetimes': datetime_list
    }
    
    return X, y, metadata

print("\nCréation séquences train...")
X_train, y_train, meta_train = create_sequences(df_train, SEQUENCE_LENGTH)

print("\nCréation séquences val...")
X_val, y_val, meta_val = create_sequences(df_val, SEQUENCE_LENGTH)

print("\nCréation séquences test...")
X_test, y_test, meta_test = create_sequences(df_test, SEQUENCE_LENGTH)

print(f"\n✅ Séquences créées:")
print(f"   Train: {X_train.shape} → {y_train.shape}")
print(f"   Val:   {X_val.shape} → {y_val.shape}")
print(f"   Test:  {X_test.shape} → {y_test.shape}")

# One-hot encode targets
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print(f"\n✅ One-hot encoding:")
print(f"   Train: {y_train_cat.shape}")
print(f"   Val:   {y_val_cat.shape}")
print(f"   Test:  {y_test_cat.shape}")

# ============================================================================
# ARCHITECTURE LSTM
# ============================================================================

print("\n" + "="*80)
print("ETAPE 3: CONSTRUCTION ARCHITECTURE LSTM")
print("="*80)

def build_lstm_classifier(input_shape, num_classes, use_focal_loss=True):
    """
    Architecture LSTM Bidirectional professionnelle.
    
    Layers:
    - Input normalization
    - Bidirectional LSTM 128 → 64
    - Batch Normalization
    - Dense 128 → 64
    - Dropout régularisation
    - Softmax output
    """
    
    inputs = layers.Input(shape=input_shape, name='input_sequence')
    
    # Normalisation
    x = layers.BatchNormalization(name='input_bn')(inputs)
    
    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name='bilstm_1'
    )(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, dropout=0.3, recurrent_dropout=0.2),
        name='bilstm_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.4, name='dropout_1')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_Extreme_Events')
    
    # Loss
    if use_focal_loss:
        loss = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        loss = 'categorical_crossentropy'
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

input_shape = (SEQUENCE_LENGTH, meta_train['num_features'])
model = build_lstm_classifier(input_shape, num_classes, use_focal)

print("\n✅ Architecture créée:")
model.summary()

print(f"\n   Params: {model.count_params():,}")
print(f"   Loss: {'Focal Loss' if use_focal else 'CrossEntropy'}")

# ============================================================================
# CALLBACKS
# ============================================================================

print("\n" + "="*80)
print("ETAPE 4: CONFIGURATION CALLBACKS")
print("="*80)

Path('models/checkpoints').mkdir(exist_ok=True, parents=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    ),
    
    ModelCheckpoint(
        filepath='models/checkpoints/lstm_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

print("✅ Callbacks configurés:")
print("   - EarlyStopping (patience=15)")
print("   - ReduceLROnPlateau (patience=7)")
print("   - ModelCheckpoint (best model)")

# ============================================================================
# ENTRAINEMENT
# ============================================================================

print("\n" + "="*80)
print("ETAPE 5: ENTRAINEMENT LSTM")
print("="*80)

print(f"\n🚀 Démarrage entrainement...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Validation: {len(X_val):,} sequences")

start_time = datetime.now()

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds() / 60

print(f"\n✅ Entrainement terminé en {duration:.1f} minutes")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*80)
print("ETAPE 6: EVALUATION SUR TEST SET")
print("="*80)

# Charger meilleur modèle
model = keras.models.load_model(
    'models/checkpoints/lstm_best.keras',
    custom_objects={'FocalLoss': FocalLoss} if use_focal else None
)

print("✅ Meilleur modèle chargé")

# Predictions
print("\nPrédictions sur test set...")
y_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print("✅ Prédictions complétées")

# Métriques
print("\n" + "="*80)
print("RESULTATS FINAUX")
print("="*80)

# F1-score
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_per_class = f1_score(y_test, y_pred, average=None)

print(f"\n📊 F1-SCORE:")
print(f"   Macro:    {f1_macro:.4f}")
print(f"   Weighted: {f1_weighted:.4f}")
print(f"\n   Par classe:")
for cls in range(num_classes):
    print(f"      {event_labels[str(cls)]:17}: {f1_per_class[cls]:.4f}")

# Recall (crucial pour événements rares)
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_per_class = recall_score(y_test, y_pred, average=None)

print(f"\n📊 RECALL (détection événements):")
print(f"   Macro: {recall_macro:.4f}")
print(f"\n   Par classe:")
for cls in range(num_classes):
    print(f"      {event_labels[str(cls)]:17}: {recall_per_class[cls]:.4f}")

# Precision
precision_macro = precision_score(y_test, y_pred, average='macro')
precision_per_class = precision_score(y_test, y_pred, average=None)

print(f"\n📊 PRECISION:")
print(f"   Macro: {precision_macro:.4f}")
print(f"\n   Par classe:")
for cls in range(num_classes):
    print(f"      {event_labels[str(cls)]:17}: {precision_per_class[cls]:.4f}")

# ROC-AUC
y_test_bin = label_binarize(y_test, classes=range(num_classes))
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro')

print(f"\n📊 ROC-AUC (one-vs-rest):")
print(f"   Macro: {roc_auc:.4f}")

# Classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT COMPLET")
print("="*80)
print(classification_report(
    y_test, y_pred,
    target_names=[event_labels[str(i)] for i in range(num_classes)],
    digits=4
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 MATRICE DE CONFUSION:")
print(cm)

# ============================================================================
# VISUALISATIONS
# ============================================================================

print("\n" + "="*80)
print("ETAPE 7: VISUALISATIONS")
print("="*80)

Path('models/results').mkdir(exist_ok=True, parents=True)

# 1. Training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history.history['loss'], label='Train')
axes[0, 0].plot(history.history['val_loss'], label='Val')
axes[0, 0].set_title('Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history.history['accuracy'], label='Train')
axes[0, 1].plot(history.history['val_accuracy'], label='Val')
axes[0, 1].set_title('Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Val')
axes[1, 0].set_title('Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(history.history['recall'], label='Train')
axes[1, 1].plot(history.history['val_recall'], label='Val')
axes[1, 1].set_title('Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('models/results/training_history.png', dpi=150)
print("✅ Training history: models/results/training_history.png")

# 2. Confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=[event_labels[str(i)] for i in range(num_classes)],
    yticklabels=[event_labels[str(i)] for i in range(num_classes)]
)
plt.title('Matrice de Confusion')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.tight_layout()
plt.savefig('models/results/confusion_matrix.png', dpi=150)
print("✅ Confusion matrix: models/results/confusion_matrix.png")

# ============================================================================
# SAUVEGARDE RESULTATS
# ============================================================================

print("\n" + "="*80)
print("ETAPE 8: SAUVEGARDE RESULTATS")
print("="*80)

results = {
    'model_config': {
        'sequence_length': SEQUENCE_LENGTH,
        'num_features': meta_train['num_features'],
        'num_classes': num_classes,
        'batch_size': BATCH_SIZE,
        'epochs_trained': len(history.history['loss']),
        'use_focal_loss': use_focal,
        'learning_rate': LEARNING_RATE
    },
    
    'metrics': {
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_per_class': {event_labels[str(i)]: float(f1_per_class[i]) for i in range(num_classes)},
        
        'recall_macro': float(recall_macro),
        'recall_per_class': {event_labels[str(i)]: float(recall_per_class[i]) for i in range(num_classes)},
        
        'precision_macro': float(precision_macro),
        'precision_per_class': {event_labels[str(i)]: float(precision_per_class[i]) for i in range(num_classes)},
        
        'roc_auc_macro': float(roc_auc)
    },
    
    'training': {
        'duration_minutes': float(duration),
        'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
        'best_val_loss': float(np.min(history.history['val_loss'])),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1])
    },
    
    'data': {
        'train_sequences': int(len(X_train)),
        'val_sequences': int(len(X_val)),
        'test_sequences': int(len(X_test))
    },
    
    'timestamp': datetime.now().isoformat()
}

with open('models/results/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Résultats sauvegardés: models/results/training_results.json")

# Sauvegarder modèle final
model.save('models/lstm_final.keras')
print("✅ Modèle final: models/lstm_final.keras")

# ============================================================================
# CONFORMITE CAHIER DES CHARGES
# ============================================================================

print("\n" + "="*80)
print("CONFORMITE CAHIER DES CHARGES")
print("="*80)

requirements = {
    'Classification événements extrêmes': f1_macro >= 0.75,
    'Recall Canicule_Extreme >= 0.85': recall_per_class[1] >= 0.85 if num_classes > 1 else False,
    'Recall Froid_Extreme >= 0.85': recall_per_class[3] >= 0.85 if num_classes > 3 else False,
    'ROC-AUC >= 0.80': roc_auc >= 0.80,
    'Deep Learning LSTM séries temporelles': True,
    'Traitement déséquilibre (Focal Loss)': use_focal,
    'Ontologie climatique (règles IF-THEN)': Path('knowledge_base/climate_ontology.json').exists()
}

print("\n✅ Exigences:")
for req, status in requirements.items():
    symbol = "✅" if status else "❌"
    print(f"   {symbol} {req}")

all_passed = all(requirements.values())
if all_passed:
    print("\n🎉 TOUTES LES EXIGENCES RESPECTEES!")
else:
    print("\n⚠️  Certaines exigences non atteintes - ajustements nécessaires")

# ============================================================================
# RESUME FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUME FINAL")
print("="*80)

print(f"\n📊 Performances:")
print(f"   F1-score (macro): {f1_macro:.4f}")
print(f"   Recall (macro):   {recall_macro:.4f}")
print(f"   Precision (macro): {precision_macro:.4f}")
print(f"   ROC-AUC:          {roc_auc:.4f}")

print(f"\n⏱️  Temps entrainement: {duration:.1f} min")
print(f"📁 Fichiers créés:")
print(f"   - models/lstm_final.keras")
print(f"   - models/checkpoints/lstm_best.keras")
print(f"   - models/results/training_results.json")
print(f"   - models/results/training_history.png")
print(f"   - models/results/confusion_matrix.png")

print("\n🚀 Prochaines étapes:")
print("   1. Intégrer moteur inférence (ontologie + règles)")
print("   2. Créer API Node.js (endpoints predictions)")
print("   3. Développer interface React (dashboard alertes)")
print("   4. Tests end-to-end")
print("   5. Documentation utilisateur")

print("\n" + "="*80)
print("✅ ENTRAINEMENT TERMINE AVEC SUCCES")
print("="*80)
