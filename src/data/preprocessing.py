"""
Data Preprocessing and Train/Val/Test Split Module

Prepares engineered features for model training by:
1. Chronological train/val/test split (prevents data leakage)
2. Missing value handling (from lag/rolling features)
3. Feature scaling (StandardScaler fitted on train only)
4. Saving preprocessed splits for reproducible training

Split strategy:
- Train: 2015-2021 (70%, ~728k records) - for model learning
- Validation: 2022-2023 (20%, ~207k records) - for hyperparameter tuning
- Test: 2024 (10%, ~108k records) - for final evaluation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses engineered features for time series modeling.
    
    Handles:
    - Chronological splitting (critical for time series - no shuffling!)
    - Missing value imputation (from lag features: first 168 hours)
    - Feature scaling (StandardScaler on numeric features only)
    - Categorical preservation (station codes, names, cities)
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'temperature'):
        """
        Initialize preprocessor with features dataset.
        
        Args:
            df: DataFrame with engineered features and datetime index
            target_col: Target variable for prediction (default: temperature)
        """
        self.df = df.copy()
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # Identify feature types
        self.numeric_cols = self._identify_numeric_features()
        self.categorical_cols = self._identify_categorical_features()
        
        logger.info(f"Initialized with {len(df):,} records")
        logger.info(f"Target variable: {target_col}")
        logger.info(f"Numeric features: {len(self.numeric_cols)}")
        logger.info(f"Categorical features: {len(self.categorical_cols)}")
        
    def _identify_numeric_features(self) -> List[str]:
        """Identify numeric columns excluding target."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from features
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
            
        # Remove temporal metadata (keep as features but don't scale)
        metadata_cols = ['year', 'month', 'day', 'hour', 'minute', 'station_id']
        numeric_cols = [c for c in numeric_cols if c not in metadata_cols]
        
        return numeric_cols
    
    def _identify_categorical_features(self) -> List[str]:
        """Identify categorical columns."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        return categorical_cols
    
    def train_val_test_split(self, 
                            train_end: str = '2021-12-31',
                            val_end: str = '2023-12-31') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (critical for time series).
        
        NO SHUFFLING - maintains temporal order to prevent data leakage.
        Future data must never influence past predictions.
        
        Args:
            train_end: Last date for training set (default: 2021-12-31)
            val_end: Last date for validation set (default: 2023-12-31)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("=" * 60)
        logger.info("Performing chronological train/val/test split")
        logger.info("=" * 60)
        
        # Ensure datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for temporal split")
        
        # Split by date
        train_df = self.df[self.df.index <= train_end].copy()
        val_df = self.df[(self.df.index > train_end) & (self.df.index <= val_end)].copy()
        test_df = self.df[self.df.index > val_end].copy()
        
        # Log split statistics
        total = len(self.df)
        logger.info(f"\nTrain set (2015-2021):")
        logger.info(f"  Records: {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
        logger.info(f"  Date range: {train_df.index.min()} to {train_df.index.max()}")
        
        logger.info(f"\nValidation set (2022-2023):")
        logger.info(f"  Records: {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
        logger.info(f"  Date range: {val_df.index.min()} to {val_df.index.max()}")
        
        logger.info(f"\nTest set (2024):")
        logger.info(f"  Records: {len(test_df):,} ({len(test_df)/total*100:.1f}%)")
        logger.info(f"  Date range: {test_df.index.min()} to {test_df.index.max()}")
        
        logger.info(f"\nTotal: {total:,} records")
        logger.info("=" * 60)
        
        return train_df, val_df, test_df
    
    def handle_missing_values(self, 
                             train_df: pd.DataFrame,
                             val_df: pd.DataFrame,
                             test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Handle missing values from lag and rolling features.
        
        Strategy:
        - First 168 hours (1 week) have missing lag_168h values
        - Use median imputation fitted on training data only
        - Apply same imputation to val/test (no leakage)
        - ALSO impute target variable if it has missing values
        
        Args:
            train_df, val_df, test_df: Split dataframes
            
        Returns:
            Tuple of imputed dataframes
        """
        logger.info("\nHandling missing values...")
        
        # All numeric columns including target
        all_numeric_cols = list(self.numeric_cols) + [self.target_col] if self.target_col not in self.numeric_cols else self.numeric_cols
        
        # Check missing values before
        train_missing_before = train_df[all_numeric_cols].isnull().sum().sum()
        val_missing_before = val_df[all_numeric_cols].isnull().sum().sum()
        test_missing_before = test_df[all_numeric_cols].isnull().sum().sum()
        
        logger.info(f"Missing values before imputation:")
        logger.info(f"  Train: {train_missing_before:,}")
        logger.info(f"  Val: {val_missing_before:,}")
        logger.info(f"  Test: {test_missing_before:,}")
        
        # Fit imputer on training data only
        self.imputer.fit(train_df[all_numeric_cols])
        
        # Transform all splits
        train_df[all_numeric_cols] = self.imputer.transform(train_df[all_numeric_cols])
        val_df[all_numeric_cols] = self.imputer.transform(val_df[all_numeric_cols])
        test_df[all_numeric_cols] = self.imputer.transform(test_df[all_numeric_cols])
        
        # Check missing values after
        train_missing_after = train_df[all_numeric_cols].isnull().sum().sum()
        val_missing_after = val_df[all_numeric_cols].isnull().sum().sum()
        test_missing_after = test_df[all_numeric_cols].isnull().sum().sum()
        
        logger.info(f"Missing values after imputation:")
        logger.info(f"  Train: {train_missing_after:,}")
        logger.info(f"  Val: {val_missing_after:,}")
        logger.info(f"  Test: {test_missing_after:,}")
        logger.info(f"✅ Imputed {train_missing_before + val_missing_before + test_missing_before:,} missing values")
        
        return train_df, val_df, test_df
    
    def scale_features(self,
                      train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale numeric features using StandardScaler.
        
        CRITICAL: Fit scaler on training data ONLY to prevent data leakage.
        Apply same transformation to val/test using train statistics.
        
        Formula: X_scaled = (X - mean_train) / std_train
        
        Args:
            train_df, val_df, test_df: Split dataframes
            
        Returns:
            Tuple of scaled dataframes
        """
        logger.info("\nScaling features with StandardScaler...")
        
        # Fit scaler on training data only
        self.scaler.fit(train_df[self.numeric_cols])
        
        # Transform all splits using train statistics
        train_df[self.numeric_cols] = self.scaler.transform(train_df[self.numeric_cols])
        val_df[self.numeric_cols] = self.scaler.transform(val_df[self.numeric_cols])
        test_df[self.numeric_cols] = self.scaler.transform(test_df[self.numeric_cols])
        
        logger.info(f"✅ Scaled {len(self.numeric_cols)} numeric features")
        logger.info(f"   Mean: {self.scaler.mean_[:5]} ...")
        logger.info(f"   Std: {self.scaler.scale_[:5]} ...")
        
        return train_df, val_df, test_df
    
    def prepare_model_inputs(self,
                           train_df: pd.DataFrame,
                           val_df: pd.DataFrame,
                           test_df: pd.DataFrame) -> dict:
        """
        Prepare final X and y arrays for model training.
        
        Separates features from target and converts to numpy arrays.
        
        Args:
            train_df, val_df, test_df: Preprocessed dataframes
            
        Returns:
            Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
        """
        logger.info("\nPreparing model inputs...")
        
        # Extract target variable
        y_train = train_df[self.target_col].values
        y_val = val_df[self.target_col].values
        y_test = test_df[self.target_col].values
        
        # Extract features (all numeric + categorical)
        feature_cols = self.numeric_cols + self.categorical_cols
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        X_test = test_df[feature_cols]
        
        logger.info(f"Training set:")
        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  y_train shape: {y_train.shape}")
        
        logger.info(f"Validation set:")
        logger.info(f"  X_val shape: {X_val.shape}")
        logger.info(f"  y_val shape: {y_val.shape}")
        
        logger.info(f"Test set:")
        logger.info(f"  X_test shape: {X_test.shape}")
        logger.info(f"  y_test shape: {y_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_cols
        }
    
    def run_full_pipeline(self) -> dict:
        """
        Execute complete preprocessing pipeline.
        
        Returns:
            Dictionary with all preprocessed data and metadata
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING FULL PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Split data chronologically
        train_df, val_df, test_df = self.train_val_test_split()
        
        # Step 2: Handle missing values
        train_df, val_df, test_df = self.handle_missing_values(train_df, val_df, test_df)
        
        # Step 3: Scale features
        train_df, val_df, test_df = self.scale_features(train_df, val_df, test_df)
        
        # Step 4: Prepare model inputs
        model_data = self.prepare_model_inputs(train_df, val_df, test_df)
        
        # Add metadata
        model_data['train_df'] = train_df
        model_data['val_df'] = val_df
        model_data['test_df'] = test_df
        model_data['scaler'] = self.scaler
        model_data['imputer'] = self.imputer
        model_data['numeric_cols'] = self.numeric_cols
        model_data['categorical_cols'] = self.categorical_cols
        
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return model_data


def save_preprocessed_data(model_data: dict, output_dir: str) -> None:
    """
    Save preprocessed data splits and fitted transformers.
    
    Saves:
    - train.parquet, val.parquet, test.parquet (full dataframes with datetime index)
    - scaler.pkl (fitted StandardScaler for production inference)
    - imputer.pkl (fitted SimpleImputer for production inference)
    - preprocessing_metadata.txt (human-readable summary)
    
    Args:
        model_data: Dictionary from run_full_pipeline()
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving preprocessed data to: {output_path}")
    
    # Save dataframes as parquet (compressed, fast)
    model_data['train_df'].to_parquet(output_path / 'train.parquet', compression='snappy')
    model_data['val_df'].to_parquet(output_path / 'val.parquet', compression='snappy')
    model_data['test_df'].to_parquet(output_path / 'test.parquet', compression='snappy')
    logger.info("✅ Saved train.parquet, val.parquet, test.parquet")
    
    # Save scaler and imputer for production use
    joblib.dump(model_data['scaler'], output_path / 'scaler.pkl')
    joblib.dump(model_data['imputer'], output_path / 'imputer.pkl')
    logger.info("✅ Saved scaler.pkl, imputer.pkl")
    
    # Save metadata
    with open(output_path / 'preprocessing_metadata.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PREPROCESSING METADATA\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET SPLITS:\n")
        f.write(f"  Train: {len(model_data['train_df']):,} records (2015-2021)\n")
        f.write(f"  Val: {len(model_data['val_df']):,} records (2022-2023)\n")
        f.write(f"  Test: {len(model_data['test_df']):,} records (2024)\n\n")
        
        f.write("FEATURES:\n")
        f.write(f"  Numeric: {len(model_data['numeric_cols'])}\n")
        f.write(f"  Categorical: {len(model_data['categorical_cols'])}\n")
        f.write(f"  Total: {len(model_data['feature_names'])}\n\n")
        
        f.write("PREPROCESSING STEPS:\n")
        f.write("  1. Chronological split (no shuffling)\n")
        f.write("  2. Median imputation (fitted on train)\n")
        f.write("  3. StandardScaler (fitted on train)\n\n")
        
        f.write("FEATURE NAMES:\n")
        for i, feat in enumerate(model_data['feature_names'], 1):
            f.write(f"  {i}. {feat}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Ready for model training!\n")
        f.write("=" * 60 + "\n")
    
    logger.info("✅ Saved preprocessing_metadata.txt")
    
    # Calculate and display file sizes
    total_size = sum(f.stat().st_size for f in output_path.glob('*') if f.is_file())
    logger.info(f"\nTotal size: {total_size / 1024**2:.2f} MB")


def load_preprocessed_data(data_dir: str) -> dict:
    """
    Load preprocessed data and fitted transformers.
    
    Args:
        data_dir: Directory containing preprocessed files
        
    Returns:
        Dictionary with all preprocessed data and transformers
    """
    data_path = Path(data_dir)
    
    logger.info(f"Loading preprocessed data from: {data_path}")
    
    # Load dataframes
    train_df = pd.read_parquet(data_path / 'train.parquet')
    val_df = pd.read_parquet(data_path / 'val.parquet')
    test_df = pd.read_parquet(data_path / 'test.parquet')
    
    # Load transformers
    scaler = joblib.load(data_path / 'scaler.pkl')
    imputer = joblib.load(data_path / 'imputer.pkl')
    
    logger.info(f"✅ Loaded {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test records")
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'scaler': scaler,
        'imputer': imputer
    }


if __name__ == "__main__":
    # Define paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_FILE = BASE_DIR / "data" / "processed" / "features_data.parquet"
    OUTPUT_DIR = BASE_DIR / "data" / "processed" / "splits"
    
    # Load features data
    logger.info(f"Loading features from: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    
    # Ensure datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
    
    # Run preprocessing pipeline
    preprocessor = DataPreprocessor(df, target_col='temperature')
    model_data = preprocessor.run_full_pipeline()
    
    # Save preprocessed data
    save_preprocessed_data(model_data, str(OUTPUT_DIR))
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA PREPROCESSING COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Train baseline models (persistence, seasonal naive, linear regression)")
    logger.info("2. Establish performance benchmarks (expected RMSE: 1.2-2.5°C)")
    logger.info("3. Train LSTM model (target RMSE: 0.8-1.0°C)")
    logger.info("4. Train XGBoost model (target RMSE: 0.9-1.3°C)")
    logger.info("5. Evaluate and compare models on test set (2024 data)")
