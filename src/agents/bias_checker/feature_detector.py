"""
Sensitive Feature Detector - Auto-identifies demographic columns in datasets.

This module provides utilities to automatically detect columns that may
represent protected demographic attributes, making the bias checker
work with any dataset.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

# Known patterns for sensitive attributes (Regex)
SENSITIVE_PATTERNS = {
    "sex": [
        r"^sex$", r"^gender$", r"^male$", r"^female$", r"^m/f$", r"^sex_.*", r"^gender_.*"
    ],
    "age": [
        r"^age$", r"^age_.*", r".*_age$", r"^patient_age$", r"^years$", r"^age_group$"
    ],
    "race": [
        r"^race$", r"^ethnicity$", r"^ethnic.*", r"^race_.*", r"^racial.*"
    ],
    "nationality": [
        r"^nation.*", r"^country.*", r"^origin$", r"^birth_country$", r"^foreign.*"
    ],
    "marital": [
        r"^marital.*", r"^married$", r"^marriage.*", r"^spouse.*"
    ],
    "religion": [
        r"^religion$", r"^religious.*", r"^faith$"
    ],
    "disability": [
        r"^disab.*", r"^handicap.*", r"^impair.*"
    ],
    "insurance": [
        r"^insurance.*", r"^insured$", r"^coverage.*", r"^medicaid$", r"^medicare$"
    ]
}

# Common value patterns that indicate demographic data
VALUE_PATTERNS = {
    "sex": [
        {"male", "female"},
        {"m", "f"},
        {"man", "woman"},
        {0, 1},  # Binary encoding
        {"0", "1"},
    ]
}


class SensitiveFeatureDetector:
    """
    Automatically detects sensitive/demographic features in a dataset.
    
    Usage:
        detector = SensitiveFeatureDetector()
        detected_cols = detector.detect(df)
        # Returns: {"sex": "gender_column", "age": "patient_age"}
        
        # Then prepare them for the checker:
        sensitive_features = prepare_sensitive_features(df, detected_cols)
    """
    
    def __init__(self, custom_patterns: Optional[Dict[str, List[str]]] = None):
        """
        Initialize detector with optional custom patterns.
        """
        self.patterns = SENSITIVE_PATTERNS.copy()
        if custom_patterns:
            for key, patterns in custom_patterns.items():
                if key in self.patterns:
                    self.patterns[key].extend(patterns)
                else:
                    self.patterns[key] = patterns
    
    def detect(
        self, 
        df: pd.DataFrame,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Detect sensitive features in a DataFrame.
        
        Returns a dictionary mapping the attribute type to the actual column name.
        e.g., {"sex": "gender", "age": "patient_age"}
        """
        detected = {}
        
        # Determine which columns to scan
        columns = include if include else df.columns.tolist()
        if exclude:
            columns = [c for c in columns if c not in exclude]
        
        for col in columns:
            col_lower = col.lower().strip()
            
            # 1. Check Column Names against Regex Patterns
            for attr_type, patterns in self.patterns.items():
                for pattern in patterns:
                    if re.match(pattern, col_lower, re.IGNORECASE):
                        # Don't overwrite if we already found one for this type
                        if attr_type not in detected:
                            detected[attr_type] = col
                        break
            
            # 2. Check Values (specifically for Sex/Gender)
            # Sometimes the column name is weird (e.g. "Var_01"), but the data is "M", "F"
            if "sex" not in detected:
                unique_vals = set(df[col].dropna().unique())
                # Normalize values to string & lowercase for comparison
                unique_lower = {str(v).lower() for v in unique_vals}
                
                for sex_pattern in VALUE_PATTERNS["sex"]:
                    if isinstance(sex_pattern, set):
                        if unique_lower == {str(v).lower() for v in sex_pattern}:
                            detected["sex"] = col
                            break
        
        return detected


def prepare_sensitive_features(
    df: pd.DataFrame,
    detected: Dict[str, str],
    create_age_groups: bool = True,
    age_bins: List[int] = [0, 40, 60, 100],
    age_labels: List[str] = ["<40", "40-60", ">60"]
) -> Dict[str, np.ndarray]:
    """
    Extracts and formats the detected features so the BiasChecker can use them.
    
    - Converts Age numbers into Groups (bins).
    - Ensures everything is a string.
    - Returns NumPy arrays.
    """
    result = {}
    
    for attr_type, col_name in detected.items():
        values = df[col_name].copy()
        
        # Special Handling for Age: Binning numbers into groups
        if attr_type == "age" and create_age_groups:
            if pd.api.types.is_numeric_dtype(values):
                values = pd.cut(
                    values,
                    bins=age_bins,
                    labels=age_labels,
                    include_lowest=True
                )
                attr_type = "age_group"  # Rename "age" -> "age_group"
        
        # Ensure values are strings (Fairlearn prefers string labels for groups)
        values = values.astype(str)
        
        result[attr_type] = values.values
    
    return result

# -------------------------------------------------------------------
# Quick Test (Run this file directly to see it work)
if __name__ == "__main__":
    # 1. Create Dummy Data
    df = pd.DataFrame({
        "id": range(5),
        "gender_col": ["M", "F", "F", "M", "M"],    # Should detect as 'sex'
        "dob_years": [25, 65, 32, 45, 80],          # Should detect as 'age'
        "diagnosis": [0, 1, 0, 1, 0]
    })
    
    print("Dataset Columns:", df.columns.tolist())

    # 2. Detect
    detector = SensitiveFeatureDetector()
    detected_cols = detector.detect(df)
    print(f"Detected Columns: {detected_cols}")  
    # Expected: {'sex': 'gender_col', 'age': 'dob_years'}

    # 3. Prepare
    final_features = prepare_sensitive_features(df, detected_cols)
    print(f"Final Processed Keys: {list(final_features.keys())}")
    # Expected: ['sex', 'age_group']
    
    print("Age Group Example:", final_features['age_group'])