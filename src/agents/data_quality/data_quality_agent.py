"""
Data Quality Validation Agent

Part of the Nimblemind Agentic AI Framework extension.
This agent validates input datasets for quality issues before 
they enter the ML pipeline.

Author: Bimukti Mozzumdar
Date: January 2025
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

#---------0---------------------------
#ENUMS
class ValidationStatus(Enum):
    """Overall validation result."""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"

class IssueSeverity(Enum):
    """Severity levels for detected issues."""
    CRITICAL = "critical"  # Blocks pipeline
    WARNING = "warning"    # Should review
    INFO = "info"          # FYI
#---------0---------------------------

#DATACLASSES

#a simple dataClass should do for our data
#ignoring boilerPlate for now
@dataclass
class Issue:
    """A single detected quality issue."""
    severity: IssueSeverity #how bad is it?
    issue_type: str #EG: MISSING_VALUES, TYPE_MISMATCH
    column: Optional[str] #which column??
    description: str #readable explanation
    affected_rows: int = 0 #how many rows have this problem. default 0
    affected_row_indices: List[int] = field(default_factory=list)  # NEW: Row positions
    examples: List[Dict[str, Any]] = field(default_factory=list)   # Now: [{row, value}]

@dataclass
class ColumnProfile:
    name: str                # Column name
    dtype: str               # Data type (int64, float64, object, etc.)
    null_count: int          # How many nulls?
    null_percentage: float   # What % is null?
    unique_count: int        # How many unique values?
    unique_percentage: float = 0.0    # What % are unique?
    sample_values: List[Any] = field(default_factory=list)  # Sample values from column
    mean: Optional[float] = None      # Mean (numeric only)
    std: Optional[float] = None       # Standard deviation (numeric only)
    min_val: Optional[Any] = None     # Minimum value
    max_val: Optional[Any] = None     # Maximum value
    top_categories: Optional[Dict[str, int]] = None  # Top categories (categorical only)

@dataclass
class ValidationReport:
    status: ValidationStatus              # Overall: PASS/WARNING/FAIL
    summary: Dict[str, Any]               # Quick stats {rows: 1000, columns: 17, ...}
    critical_issues: List[Issue]          # Blockers
    warnings: List[Issue]                 # Concerns
    info: List[Issue]                     # FYI
    column_profiles: Dict[str, ColumnProfile]  # Per-column analysis
    recommendations: List[str]            # "Consider removing column X"
    ml_methods_used: List[str] = field(default_factory=list)  # ML methods used for anomaly detection

#---------0---------------------------

#AGENT: DataQualityValidationAgent

class DataQualityValidationAgent:
    """
    Validates input datasets for quality issues.
    
    Checks performed:
    1. Schema validation (basic structure)
    2. Completeness (missing values)
    3. Type consistency
    4. Domain validation (clinical ranges)
    5. Statistical outliers
    6. ML-based anomaly detection
    """

    #constructor
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the agent.

        Args:
            config: Optional configuration overrides
        """
        # Default configuration
        self.config = {
            "null_threshold_warning": 0.10,    # 10% nulls = warning
            "null_threshold_critical": 0.50,   # 50% nulls = critical
            "outlier_zscore_threshold": 3.0,   # Z-score > 3 = outlier
            "min_rows": 10,                    # Minimum rows required
            "max_examples": 20,                # Maximum examples to show in reports
            "anomaly_contamination": 0.1,      # Expected proportion of outliers (10%)
            # Fix after initial test run: diabetic_data used '?' for missing values (UCI ML Repository standard)
            # 192,849 missing values went undetected without this (weight column was 96.8% missing!)
            "missing_indicators": ['?', '', ' ', 'NA', 'N/A', 'null', 'NULL', 'None'],  # Values treated as missing
        }

        # I passed config as a parameter
        # if user changes config, self.config should also uodate with the provided config
        if config:
            self.config.update(config)

        # if we find issues(dataclass), we store them in a List.
        self.issues: List[Issue] = []

        # Track ML methods used for anomaly detection
        self.ml_methods_used: List[str] = []

        # Store column profiles for reporting
        self.column_profiles: Dict[str, ColumnProfile] = {}
    
    def validate(self, data: pd.DataFrame) -> ValidationReport:
        """
        Main entry point. Validates a DataFrame.

        Args:
            data: The DataFrame to validate

        Returns:
            ValidationReport with all findings
        """
        # Reset state for new validation
        self.issues = []
        self.ml_methods_used = []
        self.column_profiles = {}

        # Run all validation checks
        self._check_schema(data)                    # 1. Schema validation
        self._check_completeness(data)              # 2. Completeness checks
        self._check_type_consistency(data)          # 3. Type consistency
        self._check_domain_validity(data)           # 4. Domain validation
        self._check_statistical_outliers(data)      # 5. Outlier detection
        self._check_anomalies_ml(data)              # 6. Anomaly detection (ML)
        self._build_column_profiles(data)           # 7. Build column profiles

        # Build and return report
        report = self._build_report(data)

        return report


    #---------0---------------------------
    # validation checks, ig better to first implement all the checks then move back to validate function
    # better to keep the function private as the user will only need to call validate() to see the result. 
    # will use _ infront of function names to declare them private(got it from claude. RECHECK later with crosschecking with actual documents)
    # and most functions should not return anything. all they need to do is to append to self.issues
    # if we return self.issues everytime, this will not be efficient. self.issues is not constant, so there should not be any need for a return. we should just be able to change it within our function
    # we can keep the return for all these functions as none then


    def _check_schema(self, df: pd.DataFrame) -> None:
        """Check basic schema validity."""
        # we should be keep UPDATING after running the data through the script. I am listing some initial cases from my notes

        # Check 1: Do we have any rows? --> critical
        if len(df) == 0:
            self.issues.append(Issue(
                severity=IssueSeverity.CRITICAL, # listing as critical. is there is no row, dataset is empty
                issue_type="EMPTY_DATASET",
                column=None,  # Not a column-specific issue
                description="Dataset has no rows",
                affected_rows=0
            ))
            return  # No point checking further, returning early
        
        # Check 2: Do we have columns? --> critical, empty
        if len(df.columns) == 0:
            self.issues.append(Issue(
                severity=IssueSeverity.CRITICAL,
                issue_type="NO_COLUMNS",
                column=None,
                description="Dataset has no columns",
                affected_rows=len(df)
            ))
            return
        
        # Check 3: Duplicate column names?
        duplicate_cols = df.columns[df.columns.duplicated()].tolist() # list of duplicate columns
        
        if duplicate_cols:
            self.issues.append(Issue(
                severity=IssueSeverity.CRITICAL,
                issue_type="DUPLICATE_COLUMNS",
                column=None,
                description=f"Duplicate column names found: {duplicate_cols}",
                affected_rows=len(df),
                examples=[{"column": col} for col in duplicate_cols]
            ))


        # Check 4: Unnamed columns (from bad CSV parsing)
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)] #copied from geeks for geeks
        if unnamed_cols:
            self.issues.append(Issue(
                severity=IssueSeverity.WARNING,
                issue_type="UNNAMED_COLUMNS",
                column=None,
                description=f"Found {len(unnamed_cols)} unnamed columns (possible CSV parsing issue)",
                affected_rows=len(df),
                examples=[{"column": col} for col in unnamed_cols]
            ))

        # Check 5: Minimum rows for meaningful analysis
        min_rows = self.config["min_rows"] #declared it to be 10 in our config
        
        if len(df) < min_rows:
            self.issues.append(Issue(
                severity=IssueSeverity.WARNING,
                issue_type="INSUFFICIENT_ROWS",
                column=None,
                description=f"Dataset has only {len(df)} rows (minimum recommended: {min_rows})",
                affected_rows=len(df)
            ))


        # Check 6: Duplicate rows
        duplicate_mask = df.duplicated()
        duplicate_count = duplicate_mask.sum()
        if duplicate_count > 0:
            duplicate_indices = df.index[duplicate_mask].tolist()
            self.issues.append(Issue(
                severity=IssueSeverity.WARNING,
                issue_type="DUPLICATE_ROWS",
                column=None,
                description=f"Found {duplicate_count} duplicate rows",
                affected_rows=duplicate_count,
                affected_row_indices=duplicate_indices[:20],
                examples=[{"row": idx} for idx in duplicate_indices[:20]]
            ))

        # keys need to be unique. so we should check if "id" columns have any non-unique value
        # in the paper, they mention one of their agents works on uniform "Header"
        # but for the test case, let's use some placeHolder potential column names. 

        # Check 7: Duplicate IDs (critical for healthcare data)
        id_columns = ["patient_id", "id", "ID", "PatientID", "patient_ID", "record_id"]
        
        for id_col in id_columns:
            if id_col in df.columns:
                dup_mask = df[id_col].duplicated()
                dup_count = dup_mask.sum()
                
                if dup_count > 0:
                    dup_indices = df.index[dup_mask].tolist()
                    dup_values = df.loc[dup_mask, id_col].tolist()
                    
                    examples = [
                        {"row": idx, "value": val}
                        for idx, val in zip(dup_indices[:20], dup_values[:20]) #for loop showing first 20 occurance
                    ]
                    
                    self.issues.append(Issue(
                        severity=IssueSeverity.CRITICAL,  # Critical, not warning!
                        issue_type="DUPLICATE_IDS",
                        column=id_col,
                        description=f"Column '{id_col}' has {dup_count} duplicate values (should be unique)",
                        affected_rows=dup_count,
                        affected_row_indices=dup_indices[:20],
                        examples=examples
                    ))
                
                break  # Only check the first ID column found, no point in checking others. 

          
    
    def _check_completeness(self, df: pd.DataFrame) -> None:
        """Check for missing values."""

        warning_threshold = self.config["null_threshold_warning"]
        critical_threshold = self.config["null_threshold_critical"]

        for col in df.columns:
            null_count = df[col].isnull().sum() #counting all null cells
            null_pct = null_count / len(df) # deviding by number of rows

            # NOTE: This only DETECTS and REPORTS empty columns
            # Empty columns are also DROPPED in _check_anomalies_ml() to prevent ML crashes
            if null_pct == 1.0:     # empty column
                self.issues.append(Issue(
                    severity=IssueSeverity.CRITICAL,
                    issue_type="EMPTY_COLUMN",
                    column=col,
                    description=f"Column '{col}' is completely empty (100% null)",
                    affected_rows=null_count # number of null rows. can be len(df) too for this case
                ))
            
            elif null_pct >= critical_threshold: # 0.5 is the default, unless user puts specific threshold
                self.issues.append(Issue(
                    severity=IssueSeverity.CRITICAL,
                    issue_type="HIGH_NULL_RATE",
                    column=col,
                    description=f"Column '{col}' has {null_pct:.1%} missing values",
                    affected_rows=null_count
                ))
            
            elif null_pct >= warning_threshold:
                self.issues.append(Issue(
                    severity=IssueSeverity.WARNING,
                    issue_type="MODERATE_NULL_RATE",
                    column=col,
                    description=f"Column '{col}' has {null_pct:.1%} missing values",
                    affected_rows=null_count
                ))
            
            elif null_count > 0: # just letting the user know
                self.issues.append(Issue(
                    severity=IssueSeverity.INFO,
                    issue_type="LOW_NULL_RATE",
                    column=col,
                    description=f"Column '{col}' has {null_pct:.1%} missing values ({null_count} rows)",
                    affected_rows=null_count
                ))

    
    # use case: columns should have integers, but rather have strings. Problem: "Seventy" instead of 70
    # should return the position(row number), so that it's easy to edit after the program runs (if there is any inconsistency)
    def _check_type_consistency(self, df: pd.DataFrame) -> None:
        """Check for type inconsistencies within columns."""
        
        for col in df.columns:
            # Only check object (string) columns
            if df[col].dtype != 'object': # if it's not string, continue
                continue
            
            # Skip likely categorical columns (few unique values)
            # for example, yes, no, not certain (3 unique values)
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05: # subject to change. UPDATE
                continue
            
            # Try converting to numeric
            numeric_converted = pd.to_numeric(df[col], errors='coerce') 
            
            # Count failed conversions
            # '72' will be converted to 72.0, but svemty two will be converted to null
            # so we count null
            original_nulls = df[col].isnull().sum()
            converted_nulls = numeric_converted.isnull().sum()
            failed_conversions = converted_nulls - original_nulls
            
            if failed_conversions > 0:
                # Get row indices where conversion failed
                mask = numeric_converted.isnull() & df[col].notna()
                bad_indices = df.index[mask].tolist()
                
                # Get examples with row position AND value
                examples = [
                    {"row": idx, "value": df.loc[idx, col]}
                    for idx in bad_indices[:10]  # Limit to first 10
                ]
                
                self.issues.append(Issue(
                    severity=IssueSeverity.WARNING,
                    issue_type="TYPE_INCONSISTENCY",
                    column=col,
                    description=f"Column '{col}' appears numeric but has {failed_conversions} non-numeric values",
                    affected_rows=failed_conversions,
                    affected_row_indices=bad_indices,
                    examples=examples
                ))

    
    def _check_domain_validity(self, df: pd.DataFrame) -> None:
        """Check values against clinical domain rules."""
        
        clinical_rules = {
            # Demographics
            "age": {"min": 0, "max": 120},
            
            # Vitals
            # used claude AI to get possible min and max 
            # subject to CHANGE
            "blood_pressure_systolic": {"min": 50, "max": 250},
            "blood_pressure_diastolic": {"min": 30, "max": 150},
            "heart_rate": {"min": 20, "max": 250},
            "temperature": {"min": 90, "max": 110},
            "temperature_c": {"min": 32, "max": 43},
            "oxygen_saturation": {"min": 0, "max": 100},
            "respiratory_rate": {"min": 5, "max": 60},
            
            # Body measurements
            "bmi": {"min": 10, "max": 70},
            "weight_kg": {"min": 20, "max": 300},
            "height_cm": {"min": 50, "max": 250},
            
            # Scores
            "ECOG": {"min": 0, "max": 4},
            "anxiety_score": {"min": 0, "max": 10},
            "pain_score": {"min": 0, "max": 10},
        }
        
        for col, rules in clinical_rules.items():
            # Skip if column doesn't exist
            if col not in df.columns:
                continue
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            min_val = rules["min"]
            max_val = rules["max"]
            
            # Check below minimum
            below_min_mask = df[col] < min_val
            below_min_count = below_min_mask.sum()
            
            if below_min_count > 0:
                bad_indices = df.index[below_min_mask].tolist() #list of bad indices
                examples = [
                    {"row": idx, "value": df.loc[idx, col]}
                    for idx in bad_indices[:20] # it's a for loop, where we push first 20 bad_indices to examples
                ]
                
                self.issues.append(Issue(
                    severity=IssueSeverity.CRITICAL,
                    issue_type="VALUE_BELOW_RANGE",
                    column=col,
                    description=f"Column '{col}' has {below_min_count} values below minimum ({min_val})",
                    affected_rows=below_min_count,
                    affected_row_indices=bad_indices[:20],
                    examples=examples
                ))
            
            # Check above maximum
            above_max_mask = df[col] > max_val
            above_max_count = above_max_mask.sum()
            
            if above_max_count > 0:
                bad_indices = df.index[above_max_mask].tolist()
                examples = [
                    {"row": idx, "value": df.loc[idx, col]}
                    for idx in bad_indices[:20]
                ]
                
                self.issues.append(Issue(
                    severity=IssueSeverity.CRITICAL,
                    issue_type="VALUE_ABOVE_RANGE",
                    column=col,
                    description=f"Column '{col}' has {above_max_count} values above maximum ({max_val})",
                    affected_rows=above_max_count,
                    affected_row_indices=bad_indices[:20],
                    examples=examples
                ))
    

    # alternative if ML is not intended to be used. 
    # this mainly checks local outliers
    # we will barely use this function as _check_anomaliesMl checks both global and local outliers
    def _check_statistical_outliers(self, df: pd.DataFrame) -> None:
        """Detect statistical outliers using z-scores."""
        
        zscore_threshold = self.config["outlier_zscore_threshold"]
        
        for col in df.columns:
            # Only check numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Skip columns with too few values
            valid_data = df[col].dropna()
            if len(valid_data) < 10:
                continue
            
            # Calculate mean and std
            mean = valid_data.mean()
            std = valid_data.std()
            
            # Avoid division by zero
            if std == 0:
                continue
            
            # Calculate z-scores
            z_scores = (df[col] - mean) / std
            
            # Find outliers
            outlier_mask = z_scores.abs() > zscore_threshold #stores a df column with boolean type, a for loop can be used instead
            outlier_count = outlier_mask.sum() #DEBUG: Use for loop if it does not work for dataset. 
            
            if outlier_count > 0:
                outlier_indices = df.index[outlier_mask].tolist()
                
                examples = [
                    {
                        "row": idx,
                        "value": df.loc[idx, col],
                        "z_score": round(z_scores.loc[idx], 2)
                    }
                    for idx in outlier_indices[:20]
                ]
                
                self.issues.append(Issue(
                    severity=IssueSeverity.WARNING,
                    issue_type="STATISTICAL_OUTLIER",
                    column=col,
                    description=f"Column '{col}' has {outlier_count} statistical outliers (|z-score| > {zscore_threshold})",
                    affected_rows=outlier_count,
                    affected_row_indices=outlier_indices[:20],
                    examples=examples
                ))
    

    # used IsolationForest and LOF
    # IsolationForest --> for contextual global outliers 
    # LOF ensemble --> for local outliers 
    def _check_anomalies_ml(self, df: pd.DataFrame) -> None:
            """Detect anomalies using Isolation Forest + LOF ensemble."""
            
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.neighbors import LocalOutlierFactor
            except ImportError:
                return # DEBUG: try..except was added later as I was getting errors. Decided to keep it. 
            
            max_ex = self.config["max_examples"]
            contamination = self.config["anomaly_contamination"]
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2 or len(df) < 50:
                return
            
            # Prepare data
            numeric_data = df[numeric_cols].copy() # UPGRADE: check if python has move semantics. If it is actually copying all numerical columns, will make the program slow.

            # FIX: NHANES - Drop completely empty columns before imputation (BMIHEAD is 100% NaN)
            # NOTE: _check_completeness() already DETECTS empty columns, but doesn't DROP them
            # We must DROP here because median(all NaN) = NaN, which crashes Isolation Forest/LOF
            empty_cols = numeric_data.columns[numeric_data.isnull().all()].tolist()
            if empty_cols:
                numeric_data = numeric_data.drop(columns=empty_cols)
                numeric_cols = [c for c in numeric_cols if c not in empty_cols]

            # FIX: NHANES - Use 0 as fallback if median is NaN (handles sparse columns)
            for col in numeric_cols:
                median_val = numeric_data[col].median()
                if pd.isna(median_val):
                    median_val = 0  # Fallback for columns with all NaN values
                numeric_data[col] = numeric_data[col].fillna(median_val)
            
            # Method 1: Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination, # expected number of outliers (0.1 = 10%)
                random_state=42, # for reproducibility
                n_jobs=-1 # use all CPU cores
            )
            iso_predictions = iso_forest.fit_predict(numeric_data) # 1 normal, -1 anamoly
            iso_scores = iso_forest.decision_function(numeric_data) # lower = more anamoly
            self.ml_methods_used.append("Isolation Forest")
            
            # Method 2: Local Outlier Factor
            # using both catches more types of anamoly
            lof = LocalOutlierFactor(
                n_neighbors=20, # 20 nearest neighbors, norm (IBM blog)
                contamination=contamination,
                novelty=False #Training mode is false
            )
            lof_predictions = lof.fit_predict(numeric_data)
            lof_scores = lof.negative_outlier_factor_ # more negative = more anamoly
            self.ml_methods_used.append("Local Outlier Factor (LOF)")
            
            # Ensemble analysis
            both_agree_mask = (iso_predictions == -1) & (lof_predictions == -1)
            either_flags_mask = (iso_predictions == -1) | (lof_predictions == -1)
            
            # High confidence anomalies (both agree)
            high_conf_count = both_agree_mask.sum()
            if high_conf_count > 0:
                anomaly_indices = df.index[both_agree_mask].tolist()
                
                examples = []
                for idx in anomaly_indices[:max_ex]:
                    loc = df.index.get_loc(idx)
                    example = {
                        "row": idx,
                        "iso_score": round(float(iso_scores[loc]), 3),
                        "lof_score": round(float(lof_scores[loc]), 3),
                        "values": {col: df.loc[idx, col] for col in numeric_cols[:5]}
                    }
                    examples.append(example)
                
                self.issues.append(Issue(
                    severity=IssueSeverity.WARNING,
                    issue_type="MULTIVARIATE_ANOMALY",
                    column=None,
                    description=f"Found {high_conf_count} high-confidence anomalies (both IF and LOF agree)",
                    affected_rows=high_conf_count,
                    affected_row_indices=anomaly_indices[:max_ex],
                    examples=examples
                ))
            
            # Medium confidence (one method flags)
            only_one_flags_mask = either_flags_mask & ~both_agree_mask
            medium_conf_count = only_one_flags_mask.sum()
            
            if 0 < medium_conf_count <= 50:
                anomaly_indices = df.index[only_one_flags_mask].tolist()
                
                examples = []
                for idx in anomaly_indices[:10]:
                    loc = df.index.get_loc(idx)
                    flagged_by = []
                    if iso_predictions[loc] == -1:
                        flagged_by.append("Isolation Forest")
                    if lof_predictions[loc] == -1:
                        flagged_by.append("LOF")
                    
                    example = {
                        "row": idx,
                        "flagged_by": flagged_by,
                        "values": {col: df.loc[idx, col] for col in numeric_cols[:5]}
                    }
                    examples.append(example)
                
                self.issues.append(Issue(
                    severity=IssueSeverity.INFO,
                    issue_type="POTENTIAL_ANOMALY",
                    column=None,
                    description=f"Found {medium_conf_count} potential anomalies (one algorithm flagged)",
                    affected_rows=medium_conf_count,
                    affected_row_indices=anomaly_indices[:max_ex],
                    examples=examples
                ))
    
    # ============================================================
    # REPORT BUILDING
    # ============================================================

    def _build_column_profiles(self, df: pd.DataFrame) -> None:
        """Build statistical profiles for each column."""

        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = null_count / len(df)
            unique_count = df[col].nunique()
            unique_pct = unique_count / len(df)

            profile = ColumnProfile(
                name=col,
                dtype=str(df[col].dtype),
                null_count=int(null_count),
                null_percentage=null_pct,
                unique_count=int(unique_count),
                unique_percentage=unique_pct,
                sample_values=df[col].dropna().head(5).tolist()
            )

            # Add numeric stats if applicable
            if pd.api.types.is_numeric_dtype(df[col]):
                profile.mean = float(df[col].mean())
                profile.std = float(df[col].std())
                profile.min_val = float(df[col].min())
                profile.max_val = float(df[col].max())

            # Add categorical stats if applicable
            elif df[col].dtype == 'object':
                value_counts = df[col].value_counts().head(5).to_dict()
                profile.top_categories = value_counts

            self.column_profiles[col] = profile

    def _build_report(self, df: pd.DataFrame) -> ValidationReport:
        """Compile all findings into a final report."""
        
        critical = [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]
        warnings = [i for i in self.issues if i.severity == IssueSeverity.WARNING]
        info = [i for i in self.issues if i.severity == IssueSeverity.INFO]
        
        if critical:
            status = ValidationStatus.FAIL
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASS
        
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "total_missing_values": int(df.isnull().sum().sum()),
            "overall_completeness": f"{(1 - df.isnull().sum().sum() / df.size):.1%}",
            "duplicate_rows": int(df.duplicated().sum()),
            "critical_issues": len(critical),
            "warnings": len(warnings),
            "info": len(info),
        }
        
        recommendations = self._generate_recommendations(critical, warnings)
        
        return ValidationReport(
            status=status,
            summary=summary,
            critical_issues=critical,
            warnings=warnings,
            info=info,
            column_profiles=self.column_profiles,
            recommendations=recommendations,
            ml_methods_used=self.ml_methods_used
        )
    

    # this can be done by using LLM integration, but seemed like an over-kill as there are not much to interpret. 
    # rn, I am hard-coding some recommendation strings using claude 
    def _generate_recommendations(self, critical: List[Issue], warnings: List[Issue]) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        issue_types = {i.issue_type for i in critical + warnings}
        
        if "DUPLICATE_IDS" in issue_types:
            recommendations.append(
                "🚨 CRITICAL: Duplicate patient IDs detected. Deduplicate before proceeding."
            )
        
        if "VALUE_BELOW_RANGE" in issue_types or "VALUE_ABOVE_RANGE" in issue_types:
            recommendations.append(
                "🚨 CRITICAL: Impossible clinical values detected. Review data entry or unit conversions."
            )
        
        if "HIGH_NULL_RATE" in issue_types or "EMPTY_COLUMN" in issue_types:
            recommendations.append(
                "🚨 CRITICAL: Columns with >50% missing. Consider removing or investigating data collection."
            )
        
        if "TYPE_INCONSISTENCY" in issue_types:
            recommendations.append(
                "⚠️ WARNING: Mixed types found. Clean or convert before model training."
            )
        
        if "DUPLICATE_ROWS" in issue_types:
            recommendations.append(
                "⚠️ WARNING: Duplicate rows found. Verify if legitimate or errors."
            )
        
        if "STATISTICAL_OUTLIER" in issue_types:
            recommendations.append(
                "⚠️ WARNING: Statistical outliers detected. Review for validity."
            )
        
        if "MULTIVARIATE_ANOMALY" in issue_types:
            recommendations.append(
                "⚠️ WARNING: ML detected unusual row patterns. Manual review recommended."
            )
        
        if not recommendations:
            recommendations.append("✅ No major issues. Data appears suitable for ML pipeline.")
        
        return recommendations


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def validate_dataset(data: pd.DataFrame, config: Optional[Dict] = None) -> ValidationReport:
    """Convenience function to validate a DataFrame."""
    agent = DataQualityValidationAgent(config=config)
    return agent.validate(data)


def validate_file(filepath: str, config: Optional[Dict] = None) -> ValidationReport:
    """Convenience function to validate a CSV/Excel file."""

    # Fix after initial test run: diabetic_data had '?' as missing indicator (UCI ML Repository standard)
    # Without na_values, 192,849 missing values (51.5% of total) went undetected
    # Get missing indicators from config or use defaults
    agent = DataQualityValidationAgent(config=config)
    missing_indicators = agent.config.get("missing_indicators", ['?', '', ' ', 'NA', 'N/A', 'null', 'NULL'])

    if filepath.endswith('.csv'):
        # Read CSV with custom missing value indicators
        df = pd.read_csv(filepath, na_values=missing_indicators, keep_default_na=True)
    elif filepath.endswith(('.xlsx', '.xls')):
        # Excel files also need na_values parameter
        df = pd.read_excel(filepath, na_values=missing_indicators, keep_default_na=True)
    elif filepath.endswith('.parquet'):
        # Parquet doesn't support na_values, but stores nulls properly
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

    return validate_dataset(df, config)


# ============================================================
# REPORT FORMATTING
# ============================================================

def format_report_text(report: ValidationReport) -> str:
    """Format report as human-readable text."""
    
    lines = []
    lines.append("=" * 60)
    lines.append("DATA QUALITY VALIDATION REPORT")
    lines.append("=" * 60)
    
    status_emoji = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}
    lines.append(f"\nStatus: {status_emoji.get(report.status.value, '')} {report.status.value}")
    
    lines.append(f"\n--- Summary ---")
    for key, value in report.summary.items():
        lines.append(f"  {key}: {value}")
    
    if report.ml_methods_used:
        lines.append(f"\n--- ML Methods Used ---")
        for method in report.ml_methods_used:
            lines.append(f"  • {method}")
    
    if report.critical_issues:
        lines.append(f"\n--- Critical Issues ({len(report.critical_issues)}) ---")
        for issue in report.critical_issues:
            lines.append(f"\n  🚨 [{issue.issue_type}]")
            lines.append(f"     {issue.description}")
            if issue.column:
                lines.append(f"     Column: {issue.column}")
            lines.append(f"     Affected: {issue.affected_rows} rows")
            if issue.examples:
                lines.append(f"     Examples: {issue.examples[:3]}")
    
    if report.warnings:
        lines.append(f"\n--- Warnings ({len(report.warnings)}) ---")
        for issue in report.warnings:
            lines.append(f"\n  ⚠️ [{issue.issue_type}]")
            lines.append(f"     {issue.description}")
            if issue.column:
                lines.append(f"     Column: {issue.column}")
    
    if report.info:
        lines.append(f"\n--- Info ({len(report.info)}) ---")
        for issue in report.info[:5]:  # Limit info items shown
            lines.append(f"  ℹ️ [{issue.issue_type}] {issue.description}")
        if len(report.info) > 5:
            lines.append(f"  ... and {len(report.info) - 5} more")
    
    lines.append(f"\n--- Recommendations ---")
    for rec in report.recommendations:
        lines.append(f"  {rec}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def format_report_json(report: ValidationReport) -> Dict:
    """Format report as JSON-serializable dict."""
    
    def issue_to_dict(issue: Issue) -> Dict:
        return {
            "severity": issue.severity.value,
            "issue_type": issue.issue_type,
            "column": issue.column,
            "description": issue.description,
            "affected_rows": issue.affected_rows,
            "affected_row_indices": issue.affected_row_indices,
            "examples": issue.examples
        }
    
    def profile_to_dict(profile: ColumnProfile) -> Dict:
        return {
            "name": profile.name,
            "dtype": profile.dtype,
            "null_count": profile.null_count,
            "null_percentage": profile.null_percentage,
            "unique_count": profile.unique_count,
            "unique_percentage": profile.unique_percentage,
            "sample_values": profile.sample_values,
            "mean": profile.mean,
            "std": profile.std,
            "min_val": profile.min_val,
            "max_val": profile.max_val,
            "top_categories": profile.top_categories,
        }
    
    return {
        "status": report.status.value,
        "summary": report.summary,
        "critical_issues": [issue_to_dict(i) for i in report.critical_issues],
        "warnings": [issue_to_dict(i) for i in report.warnings],
        "info": [issue_to_dict(i) for i in report.info],
        "column_profiles": {n: profile_to_dict(p) for n, p in report.column_profiles.items()},
        "recommendations": report.recommendations,
        "ml_methods_used": report.ml_methods_used
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    print("Data Quality Validation Agent")
    print("=" * 40)

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"\nValidating: {filepath}")

        # Read the file based on extension
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, na_values=['?', '', ' ', 'NA', 'N/A', 'null', 'NULL'], keep_default_na=True)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath, na_values=['?', '', ' ', 'NA', 'N/A', 'null', 'NULL'], keep_default_na=True)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")

        report = validate_dataset(df)
    else:
        print("\nNo file provided. Running demo...")

        sample_data = pd.DataFrame({
            "patient_id": [1, 2, 3, 4, 5, 5],
            "age": [25, 150, 45, -5, 60, 60],
            "heart_rate": [72, 80, "high", 95, 70, 70],
            "blood_pressure_systolic": [120, 130, 500, 110, None, None],
            "oxygen_saturation": [98, 97, 150, 95, 99, 99],
        })

        print(f"\nSample data:\n{sample_data}")
        df = sample_data
        report = validate_dataset(df)

    print(format_report_text(report))

    # Offer interactive remediation
    if report.status in [ValidationStatus.FAIL, ValidationStatus.WARNING]:
        print("\n" + "="*60)
        print("🔧 AUTOMATED REMEDIATION AVAILABLE")
        print("="*60)
        print("Would you like to interactively fix issues? (y/n): ", end="")

        try:
            response = input().strip().lower()
            if response == 'y':
                from .remediation_wizard import run_remediation_wizard

                df_cleaned = run_remediation_wizard(df, report)

                # Save cleaned dataset
                if len(sys.argv) > 1:
                    output_path = filepath.replace('.csv', '_cleaned.csv').replace('.xlsx', '_cleaned.csv')
                else:
                    output_path = "data_cleaned.csv"

                df_cleaned.to_csv(output_path, index=False)
                print(f"\n✅ Cleaned dataset saved to: {output_path}")

                # Re-validate cleaned data
                print("\n" + "="*60)
                print("RE-VALIDATING CLEANED DATA")
                print("="*60)
                report_cleaned = validate_dataset(df_cleaned)
                print(format_report_text(report_cleaned))
        except KeyboardInterrupt:
            print("\n\n⚠️  Remediation cancelled by user.")
        except Exception as e:
            print(f"\n⚠️  Error during remediation: {e}")





