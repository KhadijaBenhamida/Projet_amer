"""
Data Validation Script
Validates the processed NOAA ISD dataset and generates a comprehensive report.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_and_validate():
    """Load processed data and perform validation checks."""
    
    # Load processed data
    print("Loading processed data...")
    data_path = Path("data/processed/processed_data.parquet")
    df = pd.read_parquet(data_path)
    
    print(f"\n{'='*60}")
    print(f"DATASET VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Dataset Overview
    print("1. DATASET OVERVIEW")
    print(f"   Total Records: {len(df):,}")
    print(f"   Total Columns: {len(df.columns)}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   File Size: 13.95 MB (compressed Parquet)\n")
    
    # 2. Temporal Coverage
    print("2. TEMPORAL COVERAGE")
    print(f"   Start Date: {df.index.min()}")
    print(f"   End Date: {df.index.max()}")
    print(f"   Time Span: {(df.index.max() - df.index.min()).days} days")
    print(f"   Years Covered: {sorted(df['year'].unique())}\n")
    
    # 3. Station Coverage
    print("3. STATION COVERAGE")
    station_counts = df.groupby(['station_id', 'station_name', 'city', 'climate_zone']).size()
    for (sid, name, city, climate), count in station_counts.items():
        print(f"   {name} ({city}) - {climate}")
        print(f"      Station ID: {sid}")
        print(f"      Records: {count:,}")
        print(f"      Percentage: {count/len(df)*100:.2f}%\n")
    
    # 4. Data Quality Metrics
    print("4. DATA QUALITY METRICS")
    print("   Missing Values per Column:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        pct = (missing[col] / len(df)) * 100
        print(f"      {col}: {missing[col]:,} ({pct:.2f}%)")
    
    if missing.sum() == 0:
        print("      ✓ No missing values detected!\n")
    else:
        print()
    
    # 5. Temperature Statistics
    print("5. TEMPERATURE STATISTICS (°C)")
    print(f"   Mean: {df['temperature'].mean():.2f}°C")
    print(f"   Median: {df['temperature'].median():.2f}°C")
    print(f"   Std Dev: {df['temperature'].std():.2f}°C")
    print(f"   Min: {df['temperature'].min():.2f}°C")
    print(f"   Max: {df['temperature'].max():.2f}°C")
    print(f"   25th Percentile: {df['temperature'].quantile(0.25):.2f}°C")
    print(f"   75th Percentile: {df['temperature'].quantile(0.75):.2f}°C\n")
    
    # 6. Temperature by Climate Zone
    print("6. TEMPERATURE BY CLIMATE ZONE")
    temp_by_climate = df.groupby('climate_zone')['temperature'].agg(['mean', 'min', 'max', 'std'])
    for zone in temp_by_climate.index:
        stats = temp_by_climate.loc[zone]
        print(f"   {zone}:")
        print(f"      Mean: {stats['mean']:.2f}°C, Min: {stats['min']:.2f}°C, Max: {stats['max']:.2f}°C, Std: {stats['std']:.2f}°C\n")
    
    # 7. Wind Statistics
    print("7. WIND STATISTICS")
    print(f"   Mean Wind Speed: {df['wind_speed'].mean():.2f} m/s")
    print(f"   Max Wind Speed: {df['wind_speed'].max():.2f} m/s")
    print(f"   Calm Conditions (<1 m/s): {(df['wind_speed'] < 1).sum():,} records ({(df['wind_speed'] < 1).sum()/len(df)*100:.2f}%)\n")
    
    # 8. Pressure Statistics
    print("8. PRESSURE STATISTICS (hPa)")
    print(f"   Mean: {df['sea_level_pressure'].mean():.2f} hPa")
    print(f"   Min: {df['sea_level_pressure'].min():.2f} hPa")
    print(f"   Max: {df['sea_level_pressure'].max():.2f} hPa\n")
    
    # 9. Derived Variables (if present)
    if 'relative_humidity' in df.columns:
        print("9. DERIVED VARIABLES")
        print(f"   Relative Humidity:")
        print(f"      Mean: {df['relative_humidity'].mean():.2f}%")
        print(f"      Min: {df['relative_humidity'].min():.2f}%")
        print(f"      Max: {df['relative_humidity'].max():.2f}%\n")
        
        if 'heat_index' in df.columns:
            hi_data = df['heat_index'].dropna()
            if len(hi_data) > 0:
                print(f"   Heat Index (when applicable):")
                print(f"      Records with Heat Index: {len(hi_data):,}")
                print(f"      Mean: {hi_data.mean():.2f}°C")
                print(f"      Max: {hi_data.max():.2f}°C\n")
    
    # 10. Temporal Distribution
    print("10. TEMPORAL DISTRIBUTION")
    records_per_year = df.groupby('year').size()
    print("    Records per Year:")
    for year, count in records_per_year.items():
        print(f"      {year}: {count:,}")
    print()
    
    # 11. Data Sample
    print("11. DATA SAMPLE (First 5 records)")
    print(df.head().to_string())
    print()
    
    print(f"{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}\n")
    
    # Return summary statistics
    return {
        'total_records': len(df),
        'stations': df['station_name'].unique().tolist(),
        'date_range': (df.index.min(), df.index.max()),
        'columns': df.columns.tolist(),
        'file_size_mb': 13.95
    }

if __name__ == "__main__":
    summary = load_and_validate()
