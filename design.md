# System Design Document
## Nimblemind Agentic AI Framework Extension

**Author:** Bimukti Mozzumdar | **Date:** January 8th 2025

---
## Agents
1. Data Quality Validation Agent
2. Ethics and Bias Checker Agent

## Datasets
1. **[diabetes_012_health_indicators_BRFSS2015.csv](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)**
   - Quality: Clean survey data (BRFSS 2015)
   - Dimensions: 253,680 rows × 22 columns
   - Total data points: 5,580,960
   - Issues: 9.4% duplicate rows, no missing values

2. **[diabetic_data.csv](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)**
   - Quality: Messy clinical data with extensive missing values
   - Dimensions: 101,766 rows × 50 columns
   - Total data points: 5,088,300
   - Issues: 96.9% missing (weight), 94.7% missing (max_glu_serum), non-standard `?` missing indicators

3. **[nhanes_for_bias_test.csv](https://wwwn.cdc.gov/nchs/nhanes/)**
   - Quality: Extremely messy (100% null columns, coded missing as -1, impossible BP values)
   - Dimensions: 9,813 rows × 35 columns
   - Total data points: 343,455
   - Note: Small but structurally challenging - helped test edge cases for both agents

4. **[LLCP2022_reduced.csv](https://www.cdc.gov/brfss/annual_data/annual_2022.html)**
   - Quality: Large-scale survey with moderate missingness
   - Dimensions: 400,000 rows × 30 columns
   - Total data points: 12,000,000
   - Note: Reduced from original (445k × 328 = 146M points) via random sampling to meet GitHub's 100MB limit 

---

## Agent 1: Data Quality Validation Agent

### Purpose
Autonomous validation of healthcare datasets before ML pipeline entry. Detects structural issues, missing values, type inconsistencies, domain violations, and anomalies using ensemble statistical + ML methods. Includes interactive remediation wizard for semi-automated fixing.

### Architecture

**Location:** `src/agents/data_quality/`

**Components:**
1. `data_quality_agent.py` - Core validation engine 
2. `remediation_wizard.py` - Interactive issue fixing 

<!-- **Core Data Structures:**
```python
@dataclass
class Issue:
    severity: IssueSeverity          # CRITICAL/WARNING/INFO
    issue_type: str                  # HIGH_NULL_RATE, TYPE_MISMATCH, etc.
    column: Optional[str]
    description: str
    affected_rows: int
    affected_row_indices: List[int]  # Exact row positions
    examples: List[Dict]             # Sample values

@dataclass
class ValidationReport:
    status: ValidationStatus         # PASS/WARNING/FAIL
    summary: Dict[str, Any]
    critical_issues: List[Issue]
    warnings: List[Issue]
    info: List[Issue]
    column_profiles: Dict            # Per-column stats
    recommendations: List[str]
    ml_methods_used: List[str]
    execution_time_seconds: float
``` -->

**Configuration:**
```python
{
    "null_threshold_warning": 0.10,        # 10% nulls → WARNING
    "null_threshold_critical": 0.50,       # 50% nulls → CRITICAL
    "outlier_zscore_threshold": 3.0,
    "anomaly_contamination": 0.1,
    "max_data_points_for_ml": 5_600_000,   # Auto-disable ML above threshold
    "missing_indicators": ['?', '', 'NA']   # Non-standard indicators
}
```

### Validation Pipeline

**7 Sequential Checks:**

1. **Schema Validation**
   - Empty dataset, no columns, duplicate columns, unnamed columns
   - Insufficient rows (min=10)
   - Duplicate rows and duplicate IDs

2. **Completeness Check**
   - Tiered: 100% null → CRITICAL, ≥50% → CRITICAL, ≥10% → WARNING, >0% → INFO
   - Non-standard missing indicator detection (`?`, `NA`, `null`)

3. **Type Consistency**
   - Mixed type detection via `pd.to_numeric(errors='coerce')`
   - Returns exact row indices + problematic values to help with cleaning the dataset

4. **Domain Validation**
   - Pre-configured clinical ranges: age [0,120], BP_systolic [50,250], BMI [10,70], heart_rate [20,250], etc.
   - Flags below-min and above-max separately

5. **Statistical Outliers**
   - Z-score based: `|z| > 3.0` → WARNING
   - Univariate analysis

6. **ML Anomaly Detection** (Conditional)
   - **Trigger:** Only if `rows × cols ≤ 5.6M`
   - **Ensemble:** Isolation Forest (global) + LOF (local)
   - **Consensus:** Both agree → HIGH confidence WARNING
   - **Preprocessing:** Drops 100% null columns before ML (prevents median=NaN crash)
   - **Performance:** Auto-disables for large datasets (74x speedup). LOF is O(N^2). This algorithm significantly increases our runtime. 

7. **Column Profiling**
   - Statistics: dtype, nulls, unique values, mean/std/min/max
   - Sample values and top categories

### Interactive Remediation Wizard

**Purpose:** User-guided automated fixing of detected issues.

**Issue Categories:** Empty Columns, Missing Values, Type Inconsistencies, Domain Violations, Duplicates, Outliers

**Fix Strategies:** Fill with median/mean/mode, drop rows, convert types, cap values, manual review

**Workflow:** Group issues → prompt per category → apply fixes → auto re-validate

### Inputs/Outputs

**Input: Command Line Usage** 

```bash
# Run validation on a file (with interactive remediation)
python3 src/agents/data_quality/data_quality_agent.py data/sample/your_file.csv
```

**Output**
```bash
Affected: 6549 rows

  🚨 [HIGH_NULL_RATE]
     Column 'OHX19SE' has 66.7% missing values
     Column: OHX19SE
     Affected: 6549 rows

continues....
```


### Key Design Decisions

1. **Ensemble ML:** IF + LOF consensus reduces false positives 
2. **Performance optimization:** Auto-disable ML for datasets >5.6M points (LOF O(n²))
3. **Defensive preprocessing:** Handles `?` as missing, drops empty columns before ML
4. **Row-level tracking:** All issues include exact indices for debugging
5. **Healthcare domain:** 13 pre-configured clinical features
6. **Interactive remediation:** Grouped workflow with multiple fix strategies

---

## Agent 2: Ethics and Bias Checker Agent

### Purpose
Autonomous fairness auditing of ML predictions. Detects disparate impact across demographic groups using industry-standard metrics (Fairlearn). Auto-detects sensitive attributes and provides healthcare-specific interpretations aligned with FDA AI/ML guidelines and EEOC 4/5 rule.

### Architecture

**Location:** `src/agents/bias_checker/`

**Components:**
1. `bias_checker_agent.py` - Core metric computation 
2. `feature_detector.py` - Auto-detection of demographic columns 
3. `report.py` - JSON/Markdown export generators 

<!-- **Core Data Structures:**
```python
@dataclass
class BiasViolation:
    metric: str                      # demographic_parity_ratio, etc.
    attribute: str                   # race, sex, age_group
    value: float                     # 0.67 = 33% disparity
    threshold: float                 # 0.8 (EEOC 4/5 rule)
    severity: str                    # HIGH/MEDIUM/LOW
    interpretation: str              # Human-readable impact
    group_values: Dict[str, float]   # Per-group metrics

@dataclass
class BiasReport:
    timestamp: str
    dataset_info: Dict[str, Any]
    overall_status: str              # PASS / BIAS_DETECTED
    metrics: Dict[str, Any]
    violations: List[BiasViolation]
    recommendations: List[str]
    execution_time_seconds: float
``` -->

**Configuration:**
```python
{
    "demographic_parity_ratio_min": 0.8,      # EEOC 4/5 rule
    "demographic_parity_ratio_max": 1.25,
    "demographic_parity_diff_threshold": 0.1,
    "equal_opportunity_threshold": 0.1,       # Critical for healthcare
    "equalized_odds_threshold": 0.1,
    "severity_thresholds": {"HIGH": 0.2, "MEDIUM": 0.1, "LOW": 0.05}
}
```

### Fairness Metrics

**4 Complementary Metrics:**

1. **Demographic Parity Ratio:** `min(selection_rate) / max(selection_rate)` | Threshold: [0.8, 1.25]
2. **Demographic Parity Difference:** `max - min` | Threshold: ≤0.1
3. **Equal Opportunity Difference:** TPR difference | Threshold: ≤0.1 (critical for disease detection)
4. **Equalized Odds Difference:** Max(TPR_diff, FPR_diff) | Threshold: ≤0.1

### Sensitive Feature Detection

**Auto-Detection Strategy:**

1. **Column Name Matching:** Regex patterns for common demographics
   - Age: `r"^age$|^ridageyr$|^_age_g$"` (case-insensitive)
   - Sex: `r"^sex$|^gender$|^sexvar$"`
   - Race: `r"^race$|^ethnicity$|^_race1$"`

2. **Value Pattern Matching:** Detects demographics from data values
   - Sex: `{"male", "female"}` or `{1, 2}` with n=2
   - Uses subset matching, so these patterns detect columns that contain AT LEAST these values. eg: {"male","female", "non-binary"} --> detected. 
   - Not efficient, but was primarily hard coded to allow me test mutliple datasets for bias checker agent

3. **Age Binning:** Numeric age → bins [0, 40, 60, 100] → ["<40", "40-60", ">60"]
   - Skips if already categorical (pre-binned detection)

**Critical Preprocessing:**
- Missing/invalid values (`'nan'`, `'Unknown'`, `'?'`) → impute with mode


### Analysis Pipeline

**Per-Attribute Workflow:**

1. Compute group metrics (via Fairlearn `MetricFrame`): selection rate, TPR, FPR, precision
2. Filter small groups (exclude n<30, warn user)
3. Calculate all 4 fairness metrics per attribute
4. Detect violations (compare to thresholds)
5. automatically checks combinations of attributes. It no longer just checks if the model is fair to "Women"; it checks if it is fair to "Black Women" or "Older Men." This prevents "fairness gerrymandering."
6. The interrelational fairness test was **implemented very last moment**, and not optimized properly. It produces many redundant groupings (e.g. pregnant_&_male).

### Proxy Model for Testing

To test the bias checker agent, we need model predictions. The test script (`test_bias_checker_agent.py`) uses a simple Random Forest classifier (10 trees, depth=10) as a stand-in model. It trains on 70% of the data and predicts on the remaining 30%. These predictions are then checked for bias. This lets us test the agent quickly on any dataset without needing a real production model.

### Inputs/Outputs

**Input:CLI**
```bash
python3 tests/test_bias_checker_agent.py data/sample/your_dataset.csv target_column_name
```

Example:
```bash
python3 tests/test_bias_checker_agent.py data/sample/diabetic_data.csv readmitted
```



**Output:**
```bash
============================================================
🤖  GENERIC BIAS AUDIT RUNNER
    File:   data/sample/diabetic_data.csv
    Target: readmitted
============================================================

✅ Loaded 101766 rows.
⚙️  Preprocessing data...
```

**Healthcare Interpretations:**
- Demographic Parity: "Other patients flagged 28% less often" (not "ratio=0.72")
- Equal Opportunity: "Model misses 14% more disease cases in Other group" (not "TPR_diff=0.14")

### Key Design Decisions

1. **Multi-metric approach:** No single metric captures all fairness notions (complementary coverage)
2. **Auto-detection:** Works without manual demographic specification (hardcoded patterns + value matching)
3. **Small group filtering:** Prevents spurious violations from n<30 (statistical validity)
4. **Missing data imputation:** Avoids false bias from 'nan' groups
5. **Regulatory alignment:** EEOC 4/5 rule (0.8 threshold), FDA AI/ML guidance
6. **Healthcare context:** Clinical interpretations {LLM was used for generating interpretations and then it was hardcoded in our agent}
---

## Performance Summary

### Data Quality Agent

| Dataset | Rows × Cols | Time | Status |
|---------|-------------|------|--------|
| diabetic_data.csv | 101,766 × 50 | 2.77s | FAIL (3 critical) |
| diabetes_012_(..).csv | 253,680 × 22 | 31.71s | WARNING |
| LLCP2022_reduced.csv | 400,000 × 30 | 1.16s | WARNING (ML disabled) |

**Optimization:** Auto-disables ML when data points >5.6M (LOF O(n²) → 74x speedup)

### Bias Checker Agent

| Dataset | Test Samples | Time | Violations |
|---------|--------------|------|------------|
| diabetic_data.csv | 30,530 | 3.40s | 8 (Race, Age) |
| diabetes_012_(..).csv | 76,104 | 4.44s | 0 |
| LLCP2022_reduced.csv | 120,000 | 11.53s | 1 (Race) |

**Scales linearly:** Handles 120k samples efficiently.

---

## Limitations & Future Work

### Data Quality Agent
- Z-score inappropriate for binary data (known limitation)
- LOF struggles with high-duplicate datasets
- Domain rules hardcoded (13 clinical features only)

### Bias Checker Agent
- Feature detector uses hardcoded patterns (needs integration with nimblemind's existing learning techniques to be production ready).
- Assumes binary classification (0/1 labels)
- Pre-binned age detection heuristic

---

## Integration with Nimblemind.ai Framework

Based on the Nimblemind agentic AI pipeline (7 agents total: Feature Identifier, Data Anonymization, Feature Extraction, Model-Data Matcher, Preprocessing Recommender, Preprocessing Implementor, Model Inference), our agents can be integrated as follows:

**Data Quality Agent:**
- **Position:** Insert after Data Anonymization Agent, before Feature Extraction Agent
- **Purpose:** Validates anonymized data quality before feature extraction and model selection

**Bias Checker Agent:**
- **Position:** Insert after Model Inference Agent
- **Purpose:** Audits model predictions for fairness violations before final output to users

This integration adds automated quality assurance and fairness validation to the end-to-end clinical AI workflow.


