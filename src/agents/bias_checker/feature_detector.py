"""
Sensitive Feature Detector - Auto-identifies demographic columns in datasets.

This module provides utilities to automatically detect columns that may
represent protected demographic attributes, making the bias checker
work with any dataset.
"""

import re
import numpy as np
import pandas as pd
import warnings  # For data quality warnings
from typing import Dict, List, Optional, Union

# Known patterns for sensitive attributes (Regex)
# this is to detect columns
# for example: patient_age matches r"^age_.*"
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
# matches column variables/cells/entries to identify demographis contents
# if a column has male/female variable. it is likely sex column
# Note: Uses subset matching, so these patterns detect columns that contain AT LEAST these values
# This allows detection of inclusive datasets with "other", "non-binary", "prefer not to say", etc.
VALUE_PATTERNS = {
    "sex": [
        {"male", "female"},
        {"m", "f"},
        {"man", "woman"},
        {0, 1},  # Binary encoding
        {"0", "1"},
        # Add patterns that explicitly include non-binary options
        {"male", "female", "other"},
        {"m", "f", "o"},
        {"male", "female", "non-binary"},
        {"m", "f", "nb"},
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
        self.patterns = SENSITIVE_PATTERNS.copy() # shallow copy. Efficient. 

        # we ahould give the user the chance to give their custom_patterns
        # if the key alr exists in our declared pattern, just extend the values(patterns)
        #if it does not exist patterns[key] = patterns
        if custom_patterns:
            for key, patterns in custom_patterns.items(): # dictionary unpacking loop. copy-pasted. DEBUG
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
        # UPDATE: A better design decision will be to make the include columns mandatory
        # then, df.columns.tolist(), ignore the copies
        columns = include if include else df.columns.tolist()

        if exclude:
            columns = [c for c in columns if c not in exclude]
        
        for col in columns:
            col_lower = col.lower().strip() #make lowercase, then remove whitespace from both ends.
            
            # 1. Check Column Names against Regex Patterns
            for attr_type, patterns in self.patterns.items(): # attr_type is the key
                for pattern in patterns: # nested loop, now we are checking every pattern for every attr_type

                    # Match pattern from start of string (^ anchor). re.IGNORECASE makes it case-insensitive despite col_lower already being lowercase
                    # re.IGNORECASE might be redundant. UPDATE {not important}
                    if re.match(pattern, col_lower, re.IGNORECASE):
                        # Don't overwrite if we already found one for this type
                        if attr_type not in detected:
                            detected[attr_type] = col
                        break
            
            # 2. Check Values (specifically for Sex/Gender)
            # Sometimes the column name is weird (e.g. "Var_01"), but the data is "M", "F"
            # Note: Uses subset matching to support non-binary genders (other, non-binary, prefer not to say, etc.)
            # test case: make some false gender entries in the datasets 
            # result: will not work if the dataSet has only "male", OR "male", "other" 
            # only works only if both "male" and "female" are present
            if "sex" not in detected:
                unique_vals = set(df[col].dropna().unique())
                # Normalize values to string & lowercase for comparison
                unique_lower = {str(v).lower() for v in unique_vals}

                for sex_pattern in VALUE_PATTERNS["sex"]:
                    if isinstance(sex_pattern, set):
                        # Use subset matching: detects if column contains at least male/female (or m/f, etc.)
                        # This also catches columns with additional values like "other", "non-binary", etc.
                        normalized_pattern = {str(v).lower() for v in sex_pattern}
                        if normalized_pattern.issubset(unique_lower):
                            detected["sex"] = col
                            break
        
        return detected


def prepare_sensitive_features(
    df: pd.DataFrame,
    detected: Dict[str, str],
    create_age_groups: bool = True,  # Whether to bin age into groups (default: True)
    age_bins: List[int] = [0, 40, 60, 100],  # Bin edges: [0-40), [40-60), [60-100]
    age_labels: List[str] = ["<40", "40-60", ">60"]  # Labels for each bin
) -> Dict[str, np.ndarray]:
    """
    Extracts and formats the detected features so the BiasChecker can use them.

    Args:
        df: The original DataFrame
        detected: Dict mapping attr_type to column names (e.g., {"sex": "gender_col", "age": "patient_age"})
        create_age_groups: Whether to convert numeric ages into categorical groups (default: True)
        age_bins: Bin edges for age grouping (default: [0, 40, 60, 100])
        age_labels: Labels for each age bin (default: ["<40", "40-60", ">60"])

    Returns:
        Dict mapping attr_type to numpy arrays of string values
        e.g., {"sex": array(["male", "female", ...]), "age_group": array(["<40", ">60", ...])}

    - Does renaming of columns
    - Converts Age numbers into Groups (bins).
    - Ensures everything is a string.
    - Returns NumPy arrays.
    """
    result = {}  # Will hold {attr_type: numpy_array_of_values}

    for attr_type, col_name in detected.items():  # e.g., attr_type="sex", col_name="gender_col"
        values = df[col_name].copy()  # Extract column as pandas Series (copy to avoid modifying original)

        # Special Handling for Age: Binning numbers into groups
        if attr_type == "age" and create_age_groups:  # Only bin if age is numeric
            if pd.api.types.is_numeric_dtype(values):  # Check if column contains numbers (e.g., 25, 65, 32)
                values = pd.cut(  # Bin continuous ages into discrete groups
                    values,  # e.g., [25, 65, 32, 45, 80]
                    bins=age_bins,  # e.g., [0, 40, 60, 100] creates ranges: 0-40, 40-60, 60-100
                    labels=age_labels,  # e.g., ["<40", "40-60", ">60"] assigns labels to bins
                    include_lowest=True  # Include 0 in the first bin (0-40 instead of 1-40)
                )  # Result: ["<40", ">60", "<40", "40-60", ">60"]
                attr_type = "age_group"  # Rename key from "age" to "age_group" in result dict
            else:
                # Fix after initial test run: diabetic_data had pre-binned ages like "[0-10)", "[10-20)"
                # Skip binning if age is already categorical/string (already binned in the dataset)
                # Just rename to "age_group" and use as-is
                warnings.warn(
                    f"Age column '{col_name}' is already categorical/binned (not numeric). "
                    f"Using existing bins instead of applying custom binning. "
                    f"Example values: {values.unique()[:3]}",
                    UserWarning
                )
                attr_type = "age_group"  # Still rename to age_group for consistency

        # Ensure values are strings (Fairlearn prefers string labels for groups)
        values = values.astype(str)  # Convert everything to string (e.g., 0 → "0", NaN → "nan")

        # Fix after initial test run: diabetic_data had "nan" and "Unknown/Invalid" values
        # These created false bias violations (e.g., "nan" race showing 53% lower prediction rate)
        # Solution: Replace invalid/missing values with actual NaN, then forward-fill or use mode
        # This prevents treating missing data as a separate demographic group
        invalid_values = ['nan', 'None', 'NaN', 'unknown', 'Unknown', 'Unknown/Invalid', '?', '']
        mask = values.isin(invalid_values)

        if mask.any():
            # Fix after initial test run: Warn user about data quality issues
            # can use the other agent i developed, but might be an overkill
            num_invalid = mask.sum()
            pct_invalid = (num_invalid / len(values)) * 100
            warnings.warn(
                f"Data quality issue in '{attr_type}' ({col_name}): "
                f"{num_invalid} ({pct_invalid:.1f}%) invalid/missing values detected. "
                f"Imputing with mode to prevent false bias violations.",
                UserWarning
            )

            # Replace invalid values with the most common valid value (mode)
            valid_values = values[~mask]
            if len(valid_values) > 0:
                mode_value = valid_values.mode()[0] if len(valid_values.mode()) > 0 else valid_values.iloc[0]
                values[mask] = mode_value
                # Note: This imputation ensures we don't exclude data but also don't create spurious "nan" groups

        result[attr_type] = values.values  # Convert pandas Series to numpy array and store
    
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