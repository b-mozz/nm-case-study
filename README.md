## 1. Install Dependencies

```bash
pip install -r requirements.txt
```


## 1. Data Quality Validation Agent

Run validation on a file (includes interactive remediation):

```bash
python3 src/agents/data_quality/data_quality_agent.py data/sample/your_file.csv
```

## 2. Ethics and Bias Checker Agent

Run the bias checker using the generic audit runner:
Bash

```bash
python3 tests/test_bias_checker_agent.py data/sample/your_dataset.csv target_column_name
```

Example:
```bash
python3 tests/test_bias_checker_agent.py data/sample/diabetic_data.csv readmitted
```

