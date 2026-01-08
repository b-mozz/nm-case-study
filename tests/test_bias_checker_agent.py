import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your custom agent files
from src.agents.bias_checker.bias_checker_agent import BiasChecker
from src.agents.bias_checker.feature_detector import SensitiveFeatureDetector, prepare_sensitive_features

def run_generic_audit(file_path, target_column):
    print(f"\n{'='*60}")
    print(f"🤖  GENERIC BIAS AUDIT RUNNER")
    print(f"    File:   {file_path}")
    print(f"    Target: {target_column}")
    print(f"{'='*60}\n")

    # 1. Load Data
    try:
        # We assume '?' or empty strings might be NaNs
        df = pd.read_csv(file_path, na_values=['?', '', ' '], low_memory=False)
        print(f"✅ Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # 2. Preprocessing
    if target_column not in df.columns:
        print(f"❌ Error: Column '{target_column}' not found.")
        print(f"   Available columns: {df.columns.tolist()}")
        return

    print("⚙️  Preprocessing data...")
    
    # Separate Features and Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- FIX: ROBUST BINARY CONVERSION ---
    # FIX: After testing LLCP2022, found that DIABETE4 uses numeric codes (1/2=yes, 3=no, 4=prediabetes)
    # Need to handle both string-based ("Yes"/"No") and numeric-coded targets

    # First, try numeric binarization (for codes like 1/2 vs 3/4)
    if pd.api.types.is_numeric_dtype(y):
        # For numeric codes, assume lower values (1, 2) = positive, higher values = negative
        # Works for: DIABETE4 (1/2 = yes, 3/4 = no), binary 0/1, etc.
        median_val = y.median()
        y_binary = np.where(y <= median_val, 1, 0)
        print(f"   - Binarized numeric target using median ({median_val}). ≤median → 1, >median → 0.")
    else:
        # String-based binarization
        y_str = y.astype(str).str.upper().str.strip()

        # Identify "Negative" class (0)
        # matches: "NO", "FALSE", "0", "N", "NEGATIVE"
        is_negative = y_str.isin(['NO', 'FALSE', '0', '0.0', 'N', 'NEGATIVE'])

        # Everything else is "Positive" (1)
        y_binary = np.where(is_negative, 0, 1)
        print(f"   - Binarized string target. Mapped 'NO'/'0' to 0. All others to 1.")

    y = y_binary
    unique_vals, counts = np.unique(y, return_counts=True)
    print(f"   - Final classes in y: {dict(zip(unique_vals, counts))}")

    if len(unique_vals) < 2:
        print(f"   ⚠️  WARNING: Only one class found! Cannot train model or check bias.")
        return
    # -------------------------------------

    # Handle Features
    X_model = X.copy()

    # FIX: NHANES - Remove completely empty columns (BMIHEAD is 100% NaN)
    empty_cols = X_model.columns[X_model.isnull().all()].tolist()
    if empty_cols:
        print(f"   - Dropping {len(empty_cols)} completely empty columns: {empty_cols}")
        X_model = X_model.drop(columns=empty_cols)

    # Simple Imputer for missing values
    # We use 'most_frequent' because it works for both text and numbers
    imputer = SimpleImputer(strategy='most_frequent')

    # Get column names back after imputation (imputer returns numpy array)
    X_model_imputed = pd.DataFrame(imputer.fit_transform(X_model), columns=X_model.columns)
    
    # Label Encode all object/text columns
    for col in X_model_imputed.columns:
        if X_model_imputed[col].dtype == 'object':
            le = LabelEncoder()
            # Convert to string to handle mixed types (like int/str in same column)
            X_model_imputed[col] = le.fit_transform(X_model_imputed[col].astype(str))

    # 3. Train Proxy Model
    print("🧠 Training a quick proxy model (Random Forest)...")
    X_train, X_test, y_train, y_test = train_test_split(X_model_imputed, y, test_size=0.3, random_state=42)
    
    # Use a small forest for speed
    model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("   - Predictions generated.")

    # 4. Run The Agent
    print("\n🕵️  RUNNING BIAS AGENT...")
    
    # Recover original data format for the detector
    # We grab the rows matching X_test from the ORIGINAL dataframe
    original_test_data = df.iloc[X_test.index].copy()

    # A. Detect
    detector = SensitiveFeatureDetector()
    detected_cols = detector.detect(original_test_data, exclude=[target_column])
    
    if not detected_cols:
        print("⚠️  No sensitive features detected! (Check if columns are named 'age', 'sex', etc.)")
        return

    print(f"   - Detected sensitive columns: {detected_cols}")

    # B. Prepare
    sensitive_features = prepare_sensitive_features(original_test_data, detected_cols)

    # C. Audit
    checker = BiasChecker()

    print(f"   - Running bias check on {len(y_test):,} test samples...")
    start_time = time.time()

    report = checker.check(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    execution_time = time.time() - start_time

    # 5. Print Results
    print(f"\n{'='*30}")
    print(f"📋  FINAL REPORT STATUS: {report.overall_status}")
    print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
    print(f"{'='*30}")
    
    if report.violations:
        print(f"\nFound {len(report.violations)} Violations:\n")
        for i, v in enumerate(report.violations, 1):
            print(f"{i}. Attribute: {v.attribute.upper()}")
            print(f"   Metric:    {v.metric}")
            print(f"   Severity:  {v.severity}")
            print(f"   Issue:     {v.interpretation}")
            print(f"   Values:    {v.group_values}")
            print("   " + "-"*40)
    else:
        print("\n✅ No bias violations detected based on current thresholds.")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        file_path = sys.argv[1]
        target_col = sys.argv[2]
        run_generic_audit(file_path, target_col)
    else:
        print("Usage: python generic_bias_test.py <path_to_csv> <target_column_name>")