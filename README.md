# 🌡️ Climate Temperature Prediction - Data Preprocessing & Feature Engineering

## 📋 Project Overview

This repository contains the **data preprocessing and feature engineering pipeline** for climate temperature prediction using NOAA ISD (Integrated Surface Database) weather data from 8 US airports.

**Project Phase:** Pre-Training Data Preparation  
**Dataset:** 1,041,268 records from 2015-2024  
**Features:** 68 engineered features from 17 original variables  
**Coverage:** 8 US airports across 7 climate zones

---

## 📁 Project Structure

```
Projet_ALL/
├── data/
│   └── processed/
│       ├── features_data.parquet      # 68 engineered features
│       ├── processed_data.parquet     # Original processed data
│       └── splits/
│           ├── train.parquet          # Training set (2015-2021)
│           ├── val.parquet            # Validation set (2022-2023)
│           ├── test.parquet           # Test set (2024)
│           ├── scaler.pkl             # StandardScaler for production
│           ├── imputer.pkl            # SimpleImputer for production
│           └── preprocessing_metadata.txt
│
├── src/
│   ├── features/
│   │   └── engineering.py             # Feature engineering pipeline
│   └── data/
│       └── preprocessing.py           # Data preprocessing & splitting
│
├── notebooks/
│   └── 01_EDA_Climate_Analysis.ipynb # Exploratory Data Analysis
│
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/KhadijaBenhamida/Projet_ALL.git
cd Projet_ALL
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. **Download Preprocessed Data** 📥

⚠️ **Important:** The preprocessed `.parquet` files are too large for GitHub (~215 MB total).

**Download them separately:** See [DOWNLOAD_DATA.md](DOWNLOAD_DATA.md) for instructions.

After downloading, your structure should look like:
```
data/processed/
├── features_data.parquet ✅
├── processed_data.parquet ✅
└── splits/
    ├── train.parquet ✅
    ├── val.parquet ✅
    ├── test.parquet ✅
    ├── scaler.pkl (included in repo)
    ├── imputer.pkl (included in repo)
    └── preprocessing_metadata.txt (included in repo)
```

### 4. Load Preprocessed Data
```python
import pandas as pd
import joblib

# Load train/val/test splits
train = pd.read_parquet('data/processed/splits/train.parquet')
val = pd.read_parquet('data/processed/splits/val.parquet')
test = pd.read_parquet('data/processed/splits/test.parquet')

# Load preprocessing objects
scaler = joblib.load('data/processed/splits/scaler.pkl')
imputer = joblib.load('data/processed/splits/imputer.pkl')

print(f"Train: {train.shape}")  # (725176, 68)
print(f"Val: {val.shape}")      # (208005, 68)
print(f"Test: {test.shape}")    # (108087, 68)
```

---

## 📊 Data Overview

### Dataset Statistics
- **Total Records:** 1,041,268
- **Time Period:** 2015-01-01 to 2024-12-31
- **Temporal Resolution:** Hourly
- **Stations:** 8 US airports
- **Climate Zones:** 7 distinct zones
- **Data Quality:** 97.42% complete

### Train/Val/Test Split
| Split | Period | Records | Percentage |
|-------|--------|---------|------------|
| Train | 2015-2021 | 725,176 | 69.6% |
| Val | 2022-2023 | 208,005 | 20.0% |
| Test | 2024 | 108,087 | 10.4% |

**Note:** Chronological split (not random) to preserve temporal structure.

---

## 🛠️ Feature Engineering

### Original Features (17)
- Temperature, Dewpoint, Wind Direction, Wind Speed
- Sea Level Pressure, Station Information
- Temporal: Year, Month, Day, Hour, Minute
- Derived: Relative Humidity, Heat Index

### Engineered Features (51)

#### 1. Temporal Cyclical Features (12)
```python
hour_sin, hour_cos           # Hourly cycle
day_of_week_sin, day_of_week_cos  # Weekly cycle
day_of_year_sin, day_of_year_cos  # Yearly cycle
month_sin, month_cos         # Monthly cycle
```

#### 2. Lag Features (21)
```python
# Historical temperature values
temperature_lag_1h, 3h, 6h, 12h, 24h, 48h, 168h
dewpoint_lag_1h, 3h, 6h, 12h, 24h, 48h, 168h
sea_level_pressure_lag_1h, 3h, 6h, 12h, 24h, 48h, 168h
```

#### 3. Rolling Statistics (16)
```python
# 24-hour and 168-hour (1 week) windows
temperature_ma_24h, std_24h, min_24h, max_24h
temperature_ma_168h, std_168h, min_168h, max_168h
dewpoint_ma_24h, std_24h, min_24h, max_24h
dewpoint_ma_168h, std_168h, min_168h, max_168h
```

#### 4. Interaction Features (6)
```python
temp_dewpoint_diff = temp - dewpoint
temp_dewpoint_product = temp × dewpoint
temp_dewpoint_ratio = temp / dewpoint
temp_pressure_product = temp × pressure
hour_month_interaction
wind_chill_approx
```

#### 5. Station Encoding (3)
- One-hot encoding for 8 stations

**Total Features:** 17 original + 51 engineered = **68 features**

---

## 📈 Exploratory Data Analysis

### Key Insights from EDA Notebook

1. **Temporal Patterns**
   - Strong daily cycles in temperature
   - Seasonal variations across climate zones
   - Weekly patterns in urban stations

2. **Correlations**
   - Strong correlation: Temperature ↔ Heat Index (0.95)
   - Moderate correlation: Temperature ↔ Dewpoint (0.78)
   - Weak correlation: Temperature ↔ Wind Speed (0.12)

3. **Missing Data**
   - Original data: 2.58% missing (97.42% complete)
   - After feature engineering: 3.26% missing (due to lag features)
   - Imputation strategy: Median (SimpleImputer)

4. **Outliers**
   - Temperature range: -30°C to 45°C
   - Z-score threshold: ±3
   - Outliers retained (represent extreme weather events)

---

## 🔧 Preprocessing Pipeline

### Step 1: Feature Engineering
```python
from src.features.engineering import FeatureEngineer

engineer = FeatureEngineer('data/processed/processed_data.parquet')
df_featured = engineer.engineer_features()
# Output: data/processed/features_data.parquet
```

### Step 2: Train/Val/Test Split
```python
from src.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('data/processed/features_data.parquet')
train, val, test = preprocessor.preprocess_and_split()
# Output: data/processed/splits/*.parquet
```

### Step 3: Scaling & Imputation
```python
# StandardScaler (fitted on train only)
# SimpleImputer with median strategy
# Objects saved: scaler.pkl, imputer.pkl
```

---

## 📦 Data Files

### features_data.parquet (97.29 MB)
- Complete dataset with 68 engineered features
- Ready for model training

### splits/train.parquet (69.6 MB)
- Training data: 725,176 records
- Period: 2015-2021

### splits/val.parquet (20.0 MB)
- Validation data: 208,005 records
- Period: 2022-2023

### splits/test.parquet (10.4 MB)
- Test data: 108,087 records
- Period: 2024

### splits/scaler.pkl & imputer.pkl
- Preprocessing objects for production deployment
- Apply same transformations to new data

---

## 🎯 Usage Example

```python
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load data
train = pd.read_parquet('data/processed/splits/train.parquet')
val = pd.read_parquet('data/processed/splits/val.parquet')

# Separate features and target
X_train = train.drop('temperature', axis=1)
y_train = train['temperature']
X_val = val.drop('temperature', axis=1)
y_val = val['temperature']

# Train your model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_val, y_val)
print(f"R² Score: {score:.6f}")
```

---

## 📚 Dependencies

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyarrow>=12.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 👥 Contributors

- **Khadija Benhamida** - Data Engineering & Feature Engineering

---

## 📄 License

This project is for academic purposes.

---

## 📧 Contact

For questions or collaboration:
- GitHub: [@KhadijaBenhamida](https://github.com/KhadijaBenhamida)

---

## ✅ Data Quality Checklist

- ✅ No duplicate records
- ✅ Chronological split (temporal integrity preserved)
- ✅ Scaler fitted on train set only (no data leakage)
- ✅ Consistent feature engineering across all splits
- ✅ Missing values handled with median imputation
- ✅ All preprocessing objects saved for reproducibility

---

**Ready for Model Training!** 🚀
