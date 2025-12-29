"""
ETAPE 2: Modèle LSTM pour classification événements extrêmes

Architecture LSTM/GRU optimisée pour:
- Classification multi-classe (Normal/Canicule/Vague_froid)
- Traitement déséquilibre classes (Focal Loss, Weighted Loss)
- Métriques: F1-score, Recall, Precision, ROC-AUC

Conforme au cahier des charges: "Deep Learning pour séries temporelles"
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

print("="*80)
print("MODELE LSTM CLASSIFICATION EVENEMENTS EXTREMES")
print("="*80)

# ============================================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================================

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss pour traiter déséquilibre classes
    
    Formule: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Poids classe positive (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
            - gamma = 0: equivalent à CrossEntropy
            - gamma > 0: focus sur exemples difficiles
    
    Référence: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions pour stabilité
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Focal term: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_term = tf.pow(1 - p_t, self.gamma)
        
        # Focal loss
        loss = self.alpha * focal_term * ce
        
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    def get_config(self):
        return {
            'alpha': self.alpha,
            'gamma': self.gamma
        }

# ============================================================================
# WEIGHTED LOSS WRAPPER
# ============================================================================

def create_weighted_loss(class_weights):
    """
    Crée loss function avec poids classes
    
    Args:
        class_weights: Dict {class_id: weight}
    
    Returns:
        Loss function weighted
    """
    def weighted_categorical_crossentropy(y_true, y_pred):
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Get class weights
        weights = tf.constant([class_weights[i] for i in sorted(class_weights.keys())])
        
        # Apply weights
        weights = tf.reduce_sum(weights * y_true, axis=-1)
        
        # Cross entropy
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Weighted loss
        return tf.reduce_mean(weights * ce)
    
    return weighted_categorical_crossentropy

# ============================================================================
# ARCHITECTURE LSTM
# ============================================================================

def build_lstm_classifier(
    input_shape,
    num_classes=3,
    lstm_units=[128, 64],
    dropout_rate=0.3,
    use_bidirectional=True,
    use_attention=False
):
    """
    Construit modèle LSTM pour classification événements extrêmes
    
    Args:
        input_shape: Tuple (sequence_length, num_features)
        num_classes: Nombre classes (3: Normal/Canicule/Froid)
        lstm_units: Liste unités LSTM par couche
        dropout_rate: Taux dropout
        use_bidirectional: Utiliser Bidirectional LSTM
        use_attention: Ajouter attention layer
    
    Returns:
        Modèle Keras compilé
    """
    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs
    
    # Normalisation
    x = layers.BatchNormalization(name='batch_norm_input')(x)
    
    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1) or use_attention
        
        lstm_layer = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate * 0.5,
            name=f'lstm_{i+1}'
        )
        
        if use_bidirectional:
            x = layers.Bidirectional(lstm_layer, name=f'bidirectional_lstm_{i+1}')(x)
        else:
            x = lstm_layer(x)
        
        x = layers.BatchNormalization(name=f'batch_norm_lstm_{i+1}')(x)
    
    # Attention mechanism (optionnel)
    if use_attention:
        # Simple attention
        attention = layers.Dense(1, activation='tanh', name='attention_weights')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax', name='attention_softmax')(attention)
        attention = layers.RepeatVector(x.shape[-1])(attention)
        attention = layers.Permute([2, 1])(attention)
        
        x = layers.Multiply(name='attention_mul')([x, attention])
        x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_sum')(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu', name='dense_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    x = layers.BatchNormalization(name='batch_norm_dense')(x)
    
    x = layers.Dense(32, activation='relu', name='dense_2')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='dropout_2')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='LSTM_Classifier')
    
    return model

# ============================================================================
# GRU ALTERNATIVE
# ============================================================================

def build_gru_classifier(
    input_shape,
    num_classes=3,
    gru_units=[128, 64],
    dropout_rate=0.3
):
    """
    Architecture GRU (plus rapide que LSTM)
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    x = layers.BatchNormalization()(x)
    
    # GRU layers
    for i, units in enumerate(gru_units):
        return_sequences = i < len(gru_units) - 1
        x = layers.GRU(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate * 0.5
        )(x)
        x = layers.BatchNormalization()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='GRU_Classifier')
    
    return model

# ============================================================================
# CREATION SEQUENCES TEMPORELLES
# ============================================================================

def create_sequences(df, sequence_length=72, target_col='extreme_event', feature_cols=None):
    """
    Crée séquences temporelles pour LSTM
    
    Args:
        df: DataFrame avec features
        sequence_length: Longueur séquences (72h = 3 jours)
        target_col: Colonne target
        feature_cols: Liste colonnes features (None = auto-detect)
    
    Returns:
        X: Array (n_samples, sequence_length, n_features)
        y: Array (n_samples,) - labels
    """
    if feature_cols is None:
        # Auto-detect: exclure colonnes non-features
        exclude_cols = [
            'station_id', 'year', 'month', 'day', 'hour', 'minute',
            target_col, 'extreme_event_simple', 'is_heatwave', 'is_cold_wave', 'severity'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nCréation séquences:")
    print(f"   Sequence length: {sequence_length} heures")
    print(f"   Nombre features: {len(feature_cols)}")
    print(f"   Target: {target_col}")
    
    X = []
    y = []
    
    # Itérer par station pour préserver continuité temporelle
    stations = df['station_id'].unique()
    
    for station in stations:
        station_data = df[df['station_id'] == station].reset_index(drop=True)
        
        for i in range(len(station_data) - sequence_length):
            # Séquence features
            sequence = station_data[feature_cols].iloc[i:i+sequence_length].values
            
            # Label = dernier élément de la séquence
            label = station_data[target_col].iloc[i + sequence_length - 1]
            
            X.append(sequence)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   Séquences créées: {len(X):,}")
    print(f"   Shape X: {X.shape}")
    print(f"   Shape y: {y.shape}")
    
    return X, y, feature_cols

# ============================================================================
# METRIQUES EVALUATION
# ============================================================================

def evaluate_classifier(model, X, y_true, class_names=['Normal', 'Canicule', 'Vague froid']):
    """
    Evaluation complète avec métriques classification
    """
    print("\n" + "="*80)
    print("EVALUATION MODELE")
    print("="*80)
    
    # Prédictions
    y_pred_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # F1-scores par classe
    f1_scores = f1_score(y_true, y_pred, average=None)
    print("\nF1-Scores par classe:")
    for i, (name, score) in enumerate(zip(class_names, f1_scores)):
        print(f"   {name}: {score:.4f}")
    
    # Macro/Weighted F1
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"\nF1-Score Macro: {f1_macro:.4f}")
    print(f"F1-Score Weighted: {f1_weighted:.4f}")
    
    # ROC-AUC (one-vs-rest)
    try:
        # One-hot encode y_true
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        
        roc_auc = roc_auc_score(y_true_bin, y_pred_proba, average='macro')
        print(f"\nROC-AUC (macro): {roc_auc:.4f}")
    except Exception as e:
        print(f"\nROC-AUC: Non calculable ({e})")
        roc_auc = None
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Vraies valeurs')
    ax.set_title('Matrice de confusion')
    plt.tight_layout()
    
    return {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_scores': f1_scores,
        'roc_auc': roc_auc
    }

# ============================================================================
# FONCTION ENTRAINEMENT COMPLETE
# ============================================================================

def train_lstm_classifier(
    train_data,
    val_data,
    loss_type='focal',  # 'focal', 'weighted', 'categorical_crossentropy'
    class_weights=None,
    model_type='lstm',  # 'lstm' or 'gru'
    sequence_length=72,
    epochs=100,
    batch_size=256,
    learning_rate=0.001
):
    """
    Entraînement complet modèle LSTM/GRU classification
    
    Args:
        train_data: Tuple (X_train, y_train) ou DataFrame
        val_data: Tuple (X_val, y_val) ou DataFrame
        loss_type: 'focal', 'weighted', ou 'categorical_crossentropy'
        class_weights: Dict poids classes (pour weighted loss)
        model_type: 'lstm' ou 'gru'
        sequence_length: Longueur séquences (heures)
        epochs: Nombre epochs max
        batch_size: Taille batch
        learning_rate: Learning rate
    
    Returns:
        model, history, results_dict
    """
    print("\n" + "="*80)
    print(f"ENTRAINEMENT {model_type.upper()} - Loss: {loss_type}")
    print("="*80)
    
    # Préparer données si besoin
    if isinstance(train_data, tuple):
        X_train, y_train = train_data
        X_val, y_val = val_data
        feature_cols = None
    else:
        X_train, y_train, feature_cols = create_sequences(train_data, sequence_length)
        X_val, y_val, _ = create_sequences(val_data, sequence_length)
    
    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    
    print(f"\nDonnées:")
    print(f"   Train: {X_train.shape[0]:,} sequences")
    print(f"   Val: {X_val.shape[0]:,} sequences")
    print(f"   Input shape: {X_train.shape[1:]}")
    print(f"   Num classes: {num_classes}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    if model_type == 'lstm':
        model = build_lstm_classifier(
            input_shape,
            num_classes=num_classes,
            lstm_units=[128, 64],
            dropout_rate=0.3,
            use_bidirectional=True,
            use_attention=False
        )
    else:  # gru
        model = build_gru_classifier(
            input_shape,
            num_classes=num_classes,
            gru_units=[128, 64],
            dropout_rate=0.3
        )
    
    # Loss function
    if loss_type == 'focal':
        loss = FocalLoss(alpha=0.25, gamma=2.0)
        print("\n📍 Loss: Focal Loss (alpha=0.25, gamma=2.0)")
    elif loss_type == 'weighted':
        if class_weights is None:
            raise ValueError("class_weights requis pour weighted loss")
        loss = create_weighted_loss(class_weights)
        print(f"\n⚖️  Loss: Weighted CrossEntropy")
        print(f"   Weights: {class_weights}")
    else:
        loss = 'categorical_crossentropy'
        print("\n📊 Loss: Categorical CrossEntropy (baseline)")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    print(f"\nArchitecture:")
    model.summary()
    
    # Callbacks
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
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'models/checkpoints/{model_type}_{loss_type}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n🚀 Début entraînement...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Evaluation sur validation set:")
    results = evaluate_classifier(
        model,
        X_val,
        y_val,
        class_names=['Normal', 'Canicule', 'Vague froid']
    )
    
    return model, history, results

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("CHARGEMENT DONNEES")
    print("="*80)
    
    # Charger datasets classifiés
    train_path = Path('data/processed/splits_classified/train_classified.parquet')
    val_path = Path('data/processed/splits_classified/val_classified.parquet')
    
    if not train_path.exists():
        print("\n❌ ERREUR: Datasets classifiés non trouvés!")
        print("   Exécutez d'abord: python scripts/01_create_extreme_events_classification.py")
        exit(1)
    
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    
    print(f"✅ Train: {len(df_train):,} samples")
    print(f"✅ Val: {len(df_val):,} samples")
    
    # Charger class weights
    weights_path = Path('models/analysis/class_weights.json')
    with open(weights_path) as f:
        weights_data = json.load(f)
        class_weights = {int(k): v for k, v in weights_data['class_weights'].items()}
    
    print(f"\n✅ Class weights: {class_weights}")
    
    # Créer dossier checkpoints
    Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("ENTRAINEMENT MODELES")
    print("="*80)
    print("\nOn va entraîner 3 modèles:")
    print("   1. LSTM avec Focal Loss")
    print("   2. LSTM avec Weighted Loss")
    print("   3. LSTM avec CrossEntropy (baseline)")
    
    results_all = {}
    
    # 1. Focal Loss
    print("\n" + "="*80)
    print("MODELE 1/3: LSTM + FOCAL LOSS")
    print("="*80)
    
    model_focal, history_focal, results_focal = train_lstm_classifier(
        df_train,
        df_val,
        loss_type='focal',
        model_type='lstm',
        sequence_length=72,
        epochs=100,
        batch_size=256,
        learning_rate=0.001
    )
    
    results_all['focal'] = results_focal
    
    # Sauvegarder
    model_focal.save('models/lstm_focal_loss.keras')
    print("\n✅ Modèle sauvegardé: models/lstm_focal_loss.keras")
    
    # 2. Weighted Loss
    print("\n" + "="*80)
    print("MODELE 2/3: LSTM + WEIGHTED LOSS")
    print("="*80)
    
    model_weighted, history_weighted, results_weighted = train_lstm_classifier(
        df_train,
        df_val,
        loss_type='weighted',
        class_weights=class_weights,
        model_type='lstm',
        sequence_length=72,
        epochs=100,
        batch_size=256,
        learning_rate=0.001
    )
    
    results_all['weighted'] = results_weighted
    
    model_weighted.save('models/lstm_weighted_loss.keras')
    print("\n✅ Modèle sauvegardé: models/lstm_weighted_loss.keras")
    
    # 3. Baseline CrossEntropy
    print("\n" + "="*80)
    print("MODELE 3/3: LSTM + CROSSENTROPY (BASELINE)")
    print("="*80)
    
    model_baseline, history_baseline, results_baseline = train_lstm_classifier(
        df_train,
        df_val,
        loss_type='categorical_crossentropy',
        model_type='lstm',
        sequence_length=72,
        epochs=100,
        batch_size=256,
        learning_rate=0.001
    )
    
    results_all['baseline'] = results_baseline
    
    model_baseline.save('models/lstm_baseline.keras')
    print("\n✅ Modèle sauvegardé: models/lstm_baseline.keras")
    
    # Comparaison
    print("\n" + "="*80)
    print("COMPARAISON MODELES")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Model': ['Focal Loss', 'Weighted Loss', 'Baseline'],
        'F1-Macro': [
            results_all['focal']['f1_macro'],
            results_all['weighted']['f1_macro'],
            results_all['baseline']['f1_macro']
        ],
        'F1-Weighted': [
            results_all['focal']['f1_weighted'],
            results_all['weighted']['f1_weighted'],
            results_all['baseline']['f1_weighted']
        ],
        'ROC-AUC': [
            results_all['focal']['roc_auc'] or 0,
            results_all['weighted']['roc_auc'] or 0,
            results_all['baseline']['roc_auc'] or 0
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Meilleur modèle
    best_model = comparison.loc[comparison['F1-Macro'].idxmax(), 'Model']
    print(f"\n🏆 MEILLEUR MODELE: {best_model}")
    print(f"   F1-Macro: {comparison['F1-Macro'].max():.4f}")
    
    comparison.to_csv('models/analysis/models_comparison.csv', index=False)
    print("\n✅ Comparaison sauvegardée: models/analysis/models_comparison.csv")
    
    print("\n" + "="*80)
    print("✅ ETAPE 2 TERMINEE!")
    print("="*80)
