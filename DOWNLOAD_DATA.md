# 📥 Download Preprocessed Data

## ⚠️ Large Files Not Included

The preprocessed data files (`.parquet` format) are **too large for GitHub** (total: ~215 MB).

These files must be **downloaded separately** and placed in the correct directories.

---

## 📦 Required Data Files

### File Structure After Download:
```
data/processed/
├── features_data.parquet (97.29 MB)    # 68 engineered features
├── processed_data.parquet (13.95 MB)   # Original processed
└── splits/
    ├── train.parquet (70.98 MB)        # Training set
    ├── val.parquet (21.45 MB)          # Validation set
    ├── test.parquet (11.57 MB)         # Test set
    ├── scaler.pkl ✅                    # Included in repo
    ├── imputer.pkl ✅                   # Included in repo
    └── preprocessing_metadata.txt ✅    # Included in repo
```

---

## 🔗 Download Options

### **Option 1: Google Drive (Recommended)**

**📥 Download Link:** *(Update with your Google Drive link)*

```
https://drive.google.com/drive/folders/YOUR_FOLDER_ID
```

**Steps:**
1. Click the link above
2. Download all `.parquet` files
3. Extract to `data/processed/` in your local project
4. Keep the folder structure intact

---

### **Option 2: Generate Data Yourself**

If you prefer to generate the data from scratch:

#### Step 1: Download Raw NOAA Data
```bash
# Download from NOAA ISD database
# (Instructions depend on your data source)
```

#### Step 2: Run Preprocessing Pipeline
```bash
# Parse raw data
python src/data/parser.py

# Engineer features
python -c "
from src.features.engineering import FeatureEngineer
engineer = FeatureEngineer('data/processed/processed_data.parquet')
engineer.engineer_features()
"

# Split and scale
python -c "
from src.data.preprocessing import DataPreprocessor
prep = DataPreprocessor('data/processed/features_data.parquet')
prep.preprocess_and_split()
"
```

**Time Required:** ~30-60 minutes (depending on your machine)

---

### **Option 3: OneDrive / Dropbox**

*(Alternative cloud storage)*

Contact the repository owner for alternative download links.

---

## ✅ Verification After Download

After downloading and placing files, verify:

```bash
# Check file sizes
python -c "
import os
from pathlib import Path

files = [
    'data/processed/features_data.parquet',
    'data/processed/processed_data.parquet',
    'data/processed/splits/train.parquet',
    'data/processed/splits/val.parquet',
    'data/processed/splits/test.parquet',
]

for f in files:
    if Path(f).exists():
        size_mb = Path(f).stat().st_size / (1024**2)
        print(f'✅ {f}: {size_mb:.2f} MB')
    else:
        print(f'❌ MISSING: {f}')
"
```

**Expected Output:**
```
✅ data/processed/features_data.parquet: 97.29 MB
✅ data/processed/processed_data.parquet: 13.95 MB
✅ data/processed/splits/train.parquet: 70.98 MB
✅ data/processed/splits/val.parquet: 21.45 MB
✅ data/processed/splits/test.parquet: 11.57 MB
```

---

## 📊 Data Checksums (Optional)

To verify data integrity:

```bash
# MD5 checksums
features_data.parquet:  [UPDATE_WITH_ACTUAL_MD5]
processed_data.parquet: [UPDATE_WITH_ACTUAL_MD5]
train.parquet:          [UPDATE_WITH_ACTUAL_MD5]
val.parquet:            [UPDATE_WITH_ACTUAL_MD5]
test.parquet:           [UPDATE_WITH_ACTUAL_MD5]
```

---

## 🆘 Troubleshooting

### "File not found" error
- Verify you placed files in `data/processed/` (not in root)
- Check folder structure matches exactly

### "Cannot read parquet" error
- Install pyarrow: `pip install pyarrow`
- Verify file is not corrupted (check size)

### Download speed is slow
- Try alternative cloud storage (OneDrive/Dropbox)
- Contact repository owner for direct transfer

---

## 📧 Contact

For download issues or alternative access:
- GitHub: [@KhadijaBenhamida](https://github.com/KhadijaBenhamida)
- Email: khadija.benhamida@example.com

---

**Note:** The `.pkl` files (scaler, imputer) and metadata are small and **already included** in the repository. You only need to download the `.parquet` files.
