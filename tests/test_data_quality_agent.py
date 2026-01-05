"""
Minimal tests for Data Quality Validation Agent
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.data_quality_agent import (
    DataQualityValidationAgent,
    ValidationStatus,
    validate_dataset
)


def test_basic_validation():
    """Test basic validation with a simple dataset."""
    # Create sample data
    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 4, 5],
        "age": [25, 45, 60, 35, 50],
        "heart_rate": [72, 80, 85, 95, 70],
    })

    # Run validation
    agent = DataQualityValidationAgent()
    report = agent.validate(df)

    # Basic assertions
    assert report is not None
    assert report.status in [ValidationStatus.PASS, ValidationStatus.WARNING, ValidationStatus.FAIL]
    assert report.summary["total_rows"] == 5
    assert report.summary["total_columns"] == 3


def test_validation_with_issues():
    """Test validation detects common issues."""
    # Create data with known issues
    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 4, 5, 5],  # Duplicate ID
        "age": [25, 150, 45, -5, 60, 60],  # Invalid ages
        "heart_rate": [72, 80, "high", 95, 70, 70],  # Type inconsistency
    })

    # Run validation
    report = validate_dataset(df)

    # Should have critical issues
    assert len(report.critical_issues) > 0
    assert report.status == ValidationStatus.FAIL

    # Check specific issue types
    issue_types = [issue.issue_type for issue in report.critical_issues]
    assert "DUPLICATE_IDS" in issue_types


def test_empty_dataset():
    """Test validation handles empty dataset."""
    df = pd.DataFrame()

    agent = DataQualityValidationAgent()
    report = agent.validate(df)

    assert report.status == ValidationStatus.FAIL
    assert len(report.critical_issues) > 0


if __name__ == "__main__":
    print("Running tests...")
    test_basic_validation()
    print(" test_basic_validation passed")

    test_validation_with_issues()
    print(" test_validation_with_issues passed")

    test_empty_dataset()
    print(" test_empty_dataset passed")

    print("\nAll tests passed!")
