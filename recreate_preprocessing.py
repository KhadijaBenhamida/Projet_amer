"""Recreate preprocessing objects from training data"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

print("Loading training data...")
df = pd.read_parquet('data/processed/splits/train.parquet')

# Get numeric columns (excluding target)
numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
numeric_cols = [c for c in numeric_cols if c != 'temperature']

print(f"Found {len(numeric_cols)} numeric features")

X = df[numeric_cols]

# Create and fit scaler
print("Fitting scaler...")
scaler = StandardScaler()
scaler.fit(X)

# Create and fit imputer
print("Fitting imputer...")
imputer = SimpleImputer(strategy='mean')
imputer.fit(X)

# Save
print("Saving...")
with open('data/processed/splits/scaler_new.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('data/processed/splits/imputer_new.pkl', 'wb') as f:
    pickle.dump(imputer, f)

print(f"✅ Scaler et Imputer recréés avec {len(numeric_cols)} features!")
print(f"Features: {list(numeric_cols)[:10]}...")
