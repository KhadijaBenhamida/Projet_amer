"""
Feature Engineering Module for Climate Time Series Prediction

This module transforms raw NOAA ISD climate data into model-ready features by:
1. Temporal cyclical encoding (captures 18.7°C seasonal + 6.8°C diurnal patterns)
2. Lag features (temporal memory for historical context)
3. Rolling statistics (trend detection)
4. Interaction features (captures physical relationships like r=0.71 temp-dewpoint)
5. Station encoding (captures F=64,839 climate zone differences)

Expected improvement: 1.3-1.6°C RMSE reduction (50-60% better than raw data)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for climate time series data.
    
    Transforms 17 raw columns into ~57 model-ready features capturing:
    - Cyclical temporal patterns (seasonal sine wave, diurnal cycle)
    - Historical context (lag features for memory)
    - Trends (rolling statistics)
    - Physical relationships (interaction features)
    - Geographic differences (station encoding)
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer with processed data.
        
        Args:
            df: DataFrame with datetime index and climate variables
        """
        self.df = df.copy()
        self.original_columns = list(df.columns)
        logger.info(f"Initialized with {len(df)} records and {len(df.columns)} columns")
        
    def add_temporal_features(self) -> None:
        """
        Add cyclical temporal features using sin/cos encoding.
        
        Captures discovered patterns:
        - 18.7°C seasonal swing (annual cycle)
        - 6.8°C diurnal swing (daily cycle with peak at 21:00, low at 11:00)
        
        Creates 8 features:
        - hour_sin, hour_cos (24-hour cycle)
        - day_of_week_sin, day_of_week_cos (7-day cycle)
        - day_of_year_sin, day_of_year_cos (365-day seasonal cycle)
        - month_sin, month_cos (12-month cycle)
        """
        logger.info("Adding temporal cyclical features...")
        
        # Hour of day (0-23) - captures 6.8°C diurnal cycle
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df.index.hour / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df.index.hour / 24)
        
        # Day of week (0-6) - captures weekly patterns
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df.index.dayofweek / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df.index.dayofweek / 7)
        
        # Day of year (1-365) - captures 18.7°C seasonal swing
        self.df['day_of_year_sin'] = np.sin(2 * np.pi * self.df.index.dayofyear / 365.25)
        self.df['day_of_year_cos'] = np.cos(2 * np.pi * self.df.index.dayofyear / 365.25)
        
        # Month (1-12) - alternative seasonal encoding
        self.df['month_sin'] = np.sin(2 * np.pi * self.df.index.month / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df.index.month / 12)
        
        logger.info("Added 8 temporal features")
        
    def add_lag_features(self, variables: List[str] = None, lags: List[int] = None) -> None:
        """
        Add lag features for temporal memory.
        
        Provides historical context for autoregressive patterns.
        Critical for time series: temp(t) depends on temp(t-1), temp(t-24h), etc.
        
        Args:
            variables: List of variables to create lags for (default: temperature, dewpoint, pressure)
            lags: List of lag hours (default: [1, 3, 6, 12, 24, 48, 168])
        """
        if variables is None:
            # Focus on most predictive variables from EDA
            variables = ['temperature', 'dewpoint', 'sea_level_pressure']
            
        if lags is None:
            # Strategic lags: 1h, 3h, 6h, 12h, 24h (1 day), 48h (2 days), 168h (1 week)
            lags = [1, 3, 6, 12, 24, 48, 168]
        
        logger.info(f"Adding lag features for {len(variables)} variables with {len(lags)} lags...")
        
        for var in variables:
            if var in self.df.columns:
                for lag in lags:
                    self.df[f'{var}_lag_{lag}h'] = self.df[var].shift(lag)
        
        logger.info(f"Added {len(variables) * len(lags)} lag features")
        
    def add_rolling_features(self, variables: List[str] = None, 
                            windows: List[int] = None) -> None:
        """
        Add rolling statistics for trend detection.
        
        Captures moving averages, volatility, and range over time windows.
        Essential for detecting heating/cooling trends and stability.
        
        Args:
            variables: List of variables for rolling stats (default: temperature, dewpoint)
            windows: List of window sizes in hours (default: [24, 168])
        """
        if variables is None:
            variables = ['temperature', 'dewpoint']
            
        if windows is None:
            # 24h (1 day) and 168h (1 week) windows
            windows = [24, 168]
        
        logger.info(f"Adding rolling features for {len(variables)} variables with {len(windows)} windows...")
        
        for var in variables:
            if var in self.df.columns:
                for window in windows:
                    # Rolling mean (trend)
                    self.df[f'{var}_ma_{window}h'] = (
                        self.df[var].rolling(window=window, min_periods=1).mean()
                    )
                    
                    # Rolling std (volatility)
                    self.df[f'{var}_std_{window}h'] = (
                        self.df[var].rolling(window=window, min_periods=1).std()
                    )
                    
                    # Rolling min/max (range)
                    self.df[f'{var}_min_{window}h'] = (
                        self.df[var].rolling(window=window, min_periods=1).min()
                    )
                    self.df[f'{var}_max_{window}h'] = (
                        self.df[var].rolling(window=window, min_periods=1).max()
                    )
        
        features_added = len(variables) * len(windows) * 4  # mean, std, min, max
        logger.info(f"Added {features_added} rolling features")
        
    def add_interaction_features(self) -> None:
        """
        Add interaction features capturing physical relationships.
        
        Based on EDA findings:
        - Temperature-Dewpoint correlation: r=0.71 (strong thermodynamic relationship)
        - Temperature-Pressure correlation: r=-0.36 (atmospheric dynamics)
        - Hour×Month interaction (seasonal + diurnal combined effect)
        
        Creates features that capture non-linear relationships.
        """
        logger.info("Adding interaction features...")
        
        # Temperature-Dewpoint relationship (r=0.71)
        if 'temperature' in self.df.columns and 'dewpoint' in self.df.columns:
            # Difference (depression) - indicator of humidity
            self.df['temp_dewpoint_diff'] = self.df['temperature'] - self.df['dewpoint']
            
            # Product (interaction term)
            self.df['temp_dewpoint_product'] = self.df['temperature'] * self.df['dewpoint']
            
            # Ratio (relative relationship)
            self.df['temp_dewpoint_ratio'] = (
                self.df['temperature'] / (self.df['dewpoint'] + 273.15)  # +273.15 to avoid division issues
            )
        
        # Temperature-Pressure relationship (r=-0.36)
        if 'temperature' in self.df.columns and 'sea_level_pressure' in self.df.columns:
            self.df['temp_pressure_product'] = (
                self.df['temperature'] * self.df['sea_level_pressure']
            )
        
        # Hour-Month interaction (combined diurnal + seasonal effect)
        self.df['hour_month_interaction'] = self.df.index.hour * self.df.index.month
        
        # Wind chill approximation (if temperature and wind speed available)
        if 'temperature' in self.df.columns and 'wind_speed' in self.df.columns:
            # Simplified wind chill: T_felt = T - k * sqrt(wind_speed)
            self.df['wind_chill_approx'] = (
                self.df['temperature'] - 0.5 * np.sqrt(self.df['wind_speed'])
            )
        
        logger.info("Added 6-7 interaction features")
        
    def add_station_encoding(self) -> None:
        """
        Add station (airport) encoding to capture geographic differences.
        
        Based on EDA: ANOVA F=64,839 shows highly significant differences
        between climate zones/stations. Each station has unique characteristics.
        
        Uses one-hot encoding for interpretability.
        """
        if 'station' in self.df.columns:
            logger.info("Adding station one-hot encoding...")
            
            # One-hot encode station
            station_dummies = pd.get_dummies(self.df['station'], prefix='station')
            
            # Add to dataframe
            self.df = pd.concat([self.df, station_dummies], axis=1)
            
            logger.info(f"Added {len(station_dummies.columns)} station features")
        else:
            logger.warning("Station column not found - skipping station encoding")
            
    def add_all_features(self) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Returns:
            DataFrame with original + engineered features (~57 columns total)
        """
        logger.info("=" * 60)
        logger.info("Starting comprehensive feature engineering pipeline")
        logger.info("=" * 60)
        
        # Apply all transformations
        self.add_temporal_features()
        self.add_lag_features()
        self.add_rolling_features()
        self.add_interaction_features()
        self.add_station_encoding()
        
        # Summary
        original_count = len(self.original_columns)
        final_count = len(self.df.columns)
        engineered_count = final_count - original_count
        
        logger.info("=" * 60)
        logger.info(f"Feature engineering complete!")
        logger.info(f"Original features: {original_count}")
        logger.info(f"Engineered features: {engineered_count}")
        logger.info(f"Total features: {final_count}")
        logger.info(f"Records: {len(self.df):,}")
        logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info("=" * 60)
        
        return self.df
    
    def get_feature_info(self) -> Dict:
        """
        Get information about engineered features.
        
        Returns:
            Dictionary with feature statistics and categories
        """
        feature_info = {
            'total_features': len(self.df.columns),
            'original_features': len(self.original_columns),
            'engineered_features': len(self.df.columns) - len(self.original_columns),
            'missing_values': self.df.isnull().sum().sum(),
            'completeness': (1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'feature_categories': {
                'temporal': len([c for c in self.df.columns if any(x in c for x in ['_sin', '_cos', 'hour', 'day', 'month'])]),
                'lag': len([c for c in self.df.columns if '_lag_' in c]),
                'rolling': len([c for c in self.df.columns if any(x in c for x in ['_ma_', '_std_', '_min_', '_max_'])]),
                'interaction': len([c for c in self.df.columns if any(x in c for x in ['_diff', '_product', '_ratio', '_interaction', 'chill'])]),
                'station': len([c for c in self.df.columns if c.startswith('station_')])
            }
        }
        
        return feature_info


def load_and_engineer_features(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load processed data, engineer features, and save result.
    
    Args:
        input_path: Path to processed_data.parquet
        output_path: Path to save features_data.parquet
        
    Returns:
        DataFrame with engineered features
    """
    logger.info(f"Loading data from: {input_path}")
    
    # Load processed data
    df = pd.read_parquet(input_path)
    
    # Ensure datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
    
    # Engineer features
    engineer = FeatureEngineer(df)
    df_features = engineer.add_all_features()
    
    # Get feature info
    info = engineer.get_feature_info()
    logger.info("\nFeature Statistics:")
    logger.info(f"  Total features: {info['total_features']}")
    logger.info(f"  Completeness: {info['completeness']:.2f}%")
    logger.info(f"\nFeature Categories:")
    for category, count in info['feature_categories'].items():
        logger.info(f"  {category.capitalize()}: {count}")
    
    # Save to parquet
    logger.info(f"\nSaving engineered features to: {output_path}")
    df_features.to_parquet(output_path, compression='snappy', index=True)
    
    # Verify save
    saved_size = Path(output_path).stat().st_size / 1024**2
    logger.info(f"Saved successfully! File size: {saved_size:.2f} MB")
    
    return df_features


if __name__ == "__main__":
    # Define paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_FILE = BASE_DIR / "data" / "processed" / "processed_data.parquet"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "features_data.parquet"
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Run feature engineering pipeline
    logger.info("Starting feature engineering pipeline...")
    df_features = load_and_engineer_features(str(INPUT_FILE), str(OUTPUT_FILE))
    
    logger.info("\n" + "=" * 60)
    logger.info("Feature engineering pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"\nNext steps:")
    logger.info("1. Split data: train (2015-2021), val (2022-2023), test (2024)")
    logger.info("2. Scale features: StandardScaler (fit on train only)")
    logger.info("3. Train baseline models: persistence, seasonal naive, linear regression")
    logger.info("4. Train LSTM: expected RMSE 0.8-1.0°C")
    logger.info("5. Train XGBoost: expected RMSE 0.9-1.3°C")
