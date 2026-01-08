# Nimblemind Agentic AI Framework - Case Study

An extension to the Nimblemind Agentic AI Framework for healthcare ML pipelines, implementing two critical agents for data quality validation and bias detection.

**Author:** Bimukti Mozzumdar
**Date:** January 2025

---

## Overview

This project implements two intelligent agents designed to enhance the safety and reliability of machine learning pipelines in healthcare applications:

1. **Data Quality Validation Agent** - Validates datasets for quality issues before ML pipeline processing
2. **Ethics and Bias Checker Agent** - Detects and reports fairness violations in ML model predictions

Both agents are designed to work with healthcare data and follow defensive security practices.

---

## Features

### Data Quality Validation Agent

The Data Quality Agent performs comprehensive validation checks on input datasets:

#### Validation Checks
- **Schema Validation**: Empty datasets, missing columns, duplicate column names, unnamed columns
- **Completeness**: Missing value detection with configurable thresholds
- **Type Consistency**: Identifies mixed types within columns (e.g., "seventy" vs 70)
- **Domain Validation**: Clinical range validation (age, vitals, BMI, etc.)
- **Statistical Outliers**: Z-score based outlier detection
- **ML-based Anomaly Detection**: Ensemble approach using:
  - Isolation Forest (global/contextual anomalies)
  - Local Outlier Factor (local anomalies)
- **Duplicate Detection**: Identifies duplicate rows and duplicate IDs in key columns

#### Key Features
- Configurable severity thresholds (CRITICAL, WARNING, INFO)
- Detailed column profiling with statistics
- Row-level issue tracking with affected indices
- Actionable recommendations
- Multiple output formats (text, JSON)

### Ethics and Bias Checker Agent

The Bias Checker Agent detects fairness violations in ML model predictions:

#### Fairness Metrics
- **Demographic Parity**: Equal positive prediction rates across groups
- **Equal Opportunity**: Equal true positive rates for different groups
- **Equalized Odds**: Equal TPR and FPR across groups
- **Predictive Parity**: Equal precision across groups

#### Key Features
- Automatic sensitive feature detection (age, sex, race, etc.)
- Configurable thresholds for bias severity (CRITICAL, WARNING)
- Group-level analysis with detailed metrics
- Violations report with interpretations
- Export to JSON and Markdown formats

---

## Project Structure

```
nm-case-study/
├── src/
│   ├── agents/
│   │   ├── data_quality/
│   │   │   ├── data_quality_agent.py   # Data Quality Validation Agent
│   │   │   ├── remediation_wizard.py   # Interactive Remediation Wizard
│   │   │   └── __init__.py
│   │   ├── bias_checker/
│   │   │   ├── bias_checker_agent.py   # Bias Checker Agent
│   │   │   ├── feature_detector.py     # Sensitive feature detection
│   │   │   └── report.py               # Report generation
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── test_data_quality_agent.py      # Unit tests for data quality agent
│   ├── test_bias_checker_agent.py      # Generic bias audit runner
│   └── __init__.py
├── data/                               # Sample datasets (not tracked)
├── outputs/                            # Generated reports
├── notebooks/                          # Jupyter notebooks for demos
├── docs/                               # Documentation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd nm-case-study
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- fairlearn >= 0.10.0
- scipy >= 1.10.0

---

## Usage

### Data Quality Validation Agent

#### Basic Usage

```python
from src.agents.data_quality import validate_dataset, format_report_text
import pandas as pd

# Load your data
df = pd.read_csv("data/patient_data.csv")

# Run validation
report = validate_dataset(df)

# Print formatted report
print(format_report_text(report))
```

#### With Custom Configuration

```python
from src.agents.data_quality import DataQualityValidationAgent

# Custom thresholds
config = {
    "null_threshold_warning": 0.15,    # 15% nulls = warning
    "null_threshold_critical": 0.40,   # 40% nulls = critical
    "outlier_zscore_threshold": 2.5,   # Z-score > 2.5 = outlier
    "min_rows": 20,
    "max_examples": 10,
    "anomaly_contamination": 0.05      # Expect 5% outliers
}

agent = DataQualityValidationAgent(config=config)
report = agent.validate(df)
```

#### Interactive Remediation Wizard (NEW)

After validation, you can interactively fix detected issues:

```python
from src.agents.data_quality import validate_dataset, run_remediation_wizard

# Run validation
df = pd.read_csv("data/patient_data.csv")
report = validate_dataset(df)

# Launch interactive wizard to fix issues
if report.status in ["FAIL", "WARNING"]:
    df_cleaned = run_remediation_wizard(df, report)
    df_cleaned.to_csv("data/patient_data_cleaned.csv", index=False)
```

#### Validate from File

```python
from src.agents.data_quality import validate_file

# Supports CSV, Excel, Parquet
report = validate_file("data/patient_data.csv")
```

#### Command Line Usage

```bash
# Run validation on a file (with interactive remediation)
python3 src/agents/data_quality/data_quality_agent.py data/sample.csv

# Run demo with sample data
python3 src/agents/data_quality/data_quality_agent.py
```

**Interactive Wizard Flow:**
1. Validation report displays all issues
2. Prompt: "Would you like to interactively fix issues? (y/n)"
3. If yes: Issues grouped by category (Empty Columns, Missing Values, etc.)
4. For each category: Choose to handle, skip, or skip all remaining
5. For each issue: Select fix strategy from multiple options
6. Review summary of all fixes
7. Apply fixes and save cleaned dataset
8. Re-validate to confirm improvements

#### Understanding the Report

```python
# Check overall status
print(report.status)  # PASS, WARNING, or FAIL

# Access issues by severity
print(f"Critical Issues: {len(report.critical_issues)}")
print(f"Warnings: {len(report.warnings)}")
print(f"Info: {len(report.info)}")

# Get column profiles
for col_name, profile in report.column_profiles.items():
    print(f"{col_name}: {profile.null_percentage:.1%} nulls")

# Get recommendations
for rec in report.recommendations:
    print(rec)
```

#### Remediation Wizard Features

**Issue Categories:**
- **Empty Columns**: Drop or keep columns with 100% missing values
- **Missing Values**: Fill with median/mean/mode or drop rows
- **Type Inconsistencies**: Convert to numeric with different NaN handling strategies
- **Domain Violations**: Cap values, replace with median, or drop invalid rows
- **Duplicate Data**: Handle duplicate rows and IDs
- **Outliers & Anomalies**: Cap, replace, keep, or export for review

**Example Interaction:**
```
Category: MISSING VALUES (5 issue(s))
Do you want to handle Missing Values issues? (y/n/skip_all): y

[1] Column 'weight' has 49.1% missing values
   Column: 'weight'
   Options:
     1. Fill with median
     2. Fill with mean
     3. Drop rows with missing values
     4. Skip
   Your choice (1/2/3/4): 1

✅ Will fill 'weight' missing values with median
```

### Ethics and Bias Checker Agent

#### Basic Usage

```python
from src.agents.bias_checker.bias_checker_agent import BiasChecker
from src.agents.bias_checker.feature_detector import SensitiveFeatureDetector, prepare_sensitive_features

# Your test data and predictions
y_true = [0, 1, 0, 1, ...]  # Actual labels
y_pred = [0, 1, 1, 1, ...]  # Model predictions
test_data = pd.DataFrame(...)  # Original test data with demographics

# Detect sensitive features
detector = SensitiveFeatureDetector()
sensitive_cols = detector.detect(test_data)

# Prepare features for bias checking
sensitive_features = prepare_sensitive_features(test_data, sensitive_cols)

# Run bias check
checker = BiasChecker()
report = checker.check(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

# Print results
print(f"Status: {report.overall_status}")
for violation in report.violations:
    print(f"{violation.attribute}: {violation.interpretation}")
```

#### Using the Generic Audit Runner

The test file includes a complete audit runner that handles data loading, preprocessing, model training, and bias checking:

```bash
python3 tests/test_bias_checker_agent.py data/your_dataset.csv target_column_name
```

Example:
```bash
python3 tests/test_bias_checker_agent.py data/patient_outcomes.csv readmitted
```

This will:
1. Load and preprocess your data
2. Train a proxy Random Forest model
3. Detect sensitive features automatically
4. Run bias analysis
5. Generate a detailed report

#### Export Reports

```python
from src.agents.bias_checker.report import export_json, export_markdown

# Export to JSON
export_json(report, "outputs/bias_report.json")

# Export to Markdown
export_markdown(report, "outputs/bias_report.md")
```

---

## Testing

### Run Data Quality Agent Tests

```bash
# With pytest
pytest tests/test_data_quality_agent.py -v

# Direct execution
python3 tests/test_data_quality_agent.py
```

### Run Bias Checker Tests

The bias checker test is designed as a generic audit runner. See usage above.

---

## Configuration

### Data Quality Agent Configuration

```python
config = {
    "null_threshold_warning": 0.10,     # % nulls to trigger warning (default: 10%)
    "null_threshold_critical": 0.50,    # % nulls to trigger critical (default: 50%)
    "outlier_zscore_threshold": 3.0,    # Z-score threshold for outliers
    "min_rows": 10,                     # Minimum rows for meaningful analysis
    "max_examples": 20,                 # Max examples to show in reports
    "anomaly_contamination": 0.1        # Expected % of outliers for ML detection
}
```

### Bias Checker Configuration

Thresholds are defined in [bias_checker_agent.py:44-62](src/agents/bias_checker/bias_checker_agent.py#L44-L62):

```python
# Modify thresholds in the BiasChecker class
THRESHOLDS = {
    "demographic_parity": {"critical": 0.20, "warning": 0.10},
    "equal_opportunity": {"critical": 0.20, "warning": 0.10},
    "equalized_odds": {"critical": 0.20, "warning": 0.10},
    "predictive_parity": {"critical": 0.15, "warning": 0.08}
}
```

---

## Clinical Domain Validation Rules

The Data Quality Agent includes pre-configured clinical ranges for healthcare data:

| Feature | Min | Max |
|---------|-----|-----|
| Age | 0 | 120 |
| Blood Pressure (Systolic) | 50 | 250 |
| Blood Pressure (Diastolic) | 30 | 150 |
| Heart Rate | 20 | 250 |
| Temperature (F) | 90 | 110 |
| Temperature (C) | 32 | 43 |
| Oxygen Saturation | 0 | 100 |
| Respiratory Rate | 5 | 60 |
| BMI | 10 | 70 |
| Weight (kg) | 20 | 300 |
| Height (cm) | 50 | 250 |
| ECOG Score | 0 | 4 |
| Pain Score | 0 | 10 |
| Anxiety Score | 0 | 10 |

These ranges can be modified in [data_quality_agent.py:375-399](src/agents/data_quality_agent.py#L375-L399).

---

## Examples

### Example 1: Validating Patient Data

```python
import pandas as pd
from src.agents.data_quality_agent import validate_dataset, format_report_text

# Sample patient data
df = pd.DataFrame({
    "patient_id": [1, 2, 3, 4, 5, 5],  # Duplicate ID
    "age": [25, 150, 45, -5, 60, 60],  # Invalid ages
    "heart_rate": [72, 80, "high", 95, 70, 70],  # Type inconsistency
    "blood_pressure_systolic": [120, 130, 500, 110, None, None],
})

report = validate_dataset(df)
print(format_report_text(report))
```

### Example 2: Checking Model Bias

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.agents.bias_checker.bias_checker_agent import BiasChecker
from src.agents.bias_checker.feature_detector import prepare_sensitive_features

# Train a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Extract sensitive features
sensitive_features = prepare_sensitive_features(
    test_data,
    ["age", "sex", "race"]
)

# Check for bias
checker = BiasChecker()
report = checker.check(y_test, y_pred, sensitive_features)

if report.violations:
    print(f"Found {len(report.violations)} bias violations!")
```

---

## Development Notes

### Design Decisions

1. **Ensemble Anomaly Detection**: Uses both Isolation Forest and Local Outlier Factor to catch different types of anomalies (global vs local)
2. **Row-level Tracking**: All issues include affected row indices for easy debugging
3. **Defensive Coding**: Extensive error handling, type checking, and validation
4. **Healthcare Focus**: Clinical domain rules and sensitive feature detection tailored for healthcare data
5. **Minimal Dependencies**: Core functionality uses only pandas, numpy, and scikit-learn

### Future Improvements

- [ ] Add more ML-based anomaly detection methods (DBSCAN, One-Class SVM)
- [ ] Implement automated data cleaning suggestions
- [ ] Add temporal validation for time-series data
- [ ] Extend bias metrics (calibration curves, individual fairness)
- [ ] Add visualization dashboards for reports
- [ ] Support for real-time streaming data validation

---

## Contributing

This is a case study project. For questions or suggestions, please contact the author.

---

## Security & Ethics

**IMPORTANT**: This framework is designed for defensive security purposes only:

- ✅ **Allowed**: Data quality analysis, bias detection, vulnerability identification, security documentation
- ❌ **Not Allowed**: Malicious code generation, credential harvesting, bulk data scraping for exploitation

All agents follow ethical AI principles and are designed to improve fairness and safety in healthcare ML systems.

---

## License

See LICENSE file for details.

---

## References

- Fairlearn: https://fairlearn.org/
- Scikit-learn Isolation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- Healthcare Data Quality Standards: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371315/

---

## Contact

**Bimukti Mozzumdar**
For questions or collaboration opportunities, please open an issue in the repository.

---

*Last Updated: January 2025*
