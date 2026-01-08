"""
BiasChecker - Core class for the Ethics and Bias Checker Agent.

This module provides the main interface for checking model predictions
for potential bias across protected demographic attributes.

Uses Fairlearn for metric calculations.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Union #return convenience
from dataclasses import dataclass, field # simple dataclass
from datetime import datetime

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    count
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass
class BiasViolation:
    """Represents a single bias violation detected."""
    metric: str  # demographic_parity_ratio, demographic_parity_difference, etc.
    attribute: str # category: age, race, etc
    value: float # 0.25 = 25% gap between groups
    threshold: float # 0.1 = 10% difference
    severity: str  # HIGH, MEDIUM, LOW
    interpretation: str # the gap is massive, low --> barely crossed the line etc
    group_values: Dict[str, float] = field(default_factory=dict) # a dictionary. key: subGroup of attribute (if attribute is gender then male, female are the keys)

@dataclass
class BiasReport:
    """Complete bias analysis report."""
    timestamp: str # what time audit took place
    dataset_info: Dict[str, Any] # everything about the dataSet. for eg: "rows" = 1000;
    overall_status: str  # PASS, BIAS_DETECTED
    metrics: Dict[str, Any] # numbers calculated by FairLearn
    violations: List[BiasViolation] # list of all the BiasViolation Object (right on top of this dataclass)
    recommendations: List[str] # advice for the developers/whoever is working with the data
    execution_time_seconds: float = 0.0  # Total execution time in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        
        def safe_round(val):
            """Round floats, handle dicts and other types."""
            if isinstance(val, float):
                return round(val, 4)
            elif isinstance(val, dict):
                return {k: safe_round(v) for k, v in val.items()}
            else:
                return val
        
        return {
            "timestamp": self.timestamp,
            "dataset_info": self.dataset_info,
            "overall_status": self.overall_status,
            "metrics": self.metrics,
            "violations": [
                {
                    "metric": v.metric,
                    "attribute": v.attribute,
                    "value": round(v.value, 4),
                    "threshold": v.threshold,
                    "severity": v.severity,
                    "interpretation": v.interpretation,
                    "group_values": {k: safe_round(val) for k, val in v.group_values.items()}
                }
                for v in self.violations
            ],
            "recommendations": self.recommendations,
            "execution_time_seconds": self.execution_time_seconds
        }


class BiasChecker:
    """
    Ethics and Bias Checker Agent for healthcare ML pipelines.
    
    Analyzes model predictions for potential bias across protected
    demographic attributes using industry-standard fairness metrics.
    
    Usage:
        checker = BiasChecker()
        report = checker.check(
            y_true=actual_labels,
            y_pred=predictions,
            sensitive_features={"sex": sex_column, "age_group": age_column}
        )
        
        if report.overall_status == "BIAS_DETECTED":
            print(report.violations)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BiasChecker with configuration.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = {
            # Demographic parity thresholds
            "demographic_parity_ratio_min": 0.8,
            "demographic_parity_ratio_max": 1.25,
            "demographic_parity_diff_threshold": 0.1,
            
            # Equalized odds / equal opportunity thresholds
            "equalized_odds_threshold": 0.1,
            "equal_opportunity_threshold": 0.1,
            
            # Severity classification
            "severity_thresholds": {
                "HIGH": 0.2,
                "MEDIUM": 0.1,
                "LOW": 0.05
            },
            
            # Healthcare context
            "positive_outcome_label": "high-risk",
            "negative_outcome_label": "low-risk",
            "clinical_context": "disease_prediction"
        }
        
        if config:
            self.config.update(config)
    
    def check(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        sensitive_features: Dict[str, Union[np.ndarray, pd.Series, List]]
    ) -> BiasReport:
        """
        Perform bias analysis on model predictions.

        Args:
            y_true: Actual outcome labels (0/1)
            y_pred: Model predictions (0/1)
            sensitive_features: Dict mapping attribute names to their values
                               e.g., {"sex": [...], "age_group": [...]}

        Returns:
            BiasReport with metrics, violations, and recommendations
        """
        # Start timing
        start_time = time.time()

        # Convert to numpy arrays, user might provide list, pandas series or NumPy arrays
        # maintains uniformity so that our imported functions dont crash
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Validate inputs
        self._validate_inputs(y_true, y_pred, sensitive_features)

        # Collect all metrics and violations
        all_metrics = {}
        all_violations = []

        # Analyze each sensitive attribute
        # for example we want to check Age AND Gender AND Race, every category together
        for attr_name, attr_values in sensitive_features.items():
            attr_values = np.array(attr_values)

            metrics, violations = self._analyze_attribute(
                y_true, y_pred, attr_values, attr_name
            )

            all_metrics[attr_name] = metrics
            all_violations.extend(violations)

        # Determine overall status
        overall_status = "BIAS_DETECTED" if all_violations else "PASS"

        # Generate recommendations
        recommendations = self._generate_recommendations(all_violations)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build report
        report = BiasReport(
            timestamp=datetime.now().isoformat(),
            dataset_info={
                "total_samples": len(y_true),
                "positive_count": int(y_true.sum()),
                "negative_count": int(len(y_true) - y_true.sum()),
                "positive_rate": round(float(y_true.mean()), 4),
                "prediction_positive_rate": round(float(y_pred.mean()), 4),
                "groups_analyzed": list(sensitive_features.keys())
            },
            overall_status=overall_status,
            metrics=all_metrics,
            violations=all_violations,
            recommendations=recommendations,
            execution_time_seconds=execution_time
        )

        return report
    
    def _validate_inputs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: Dict[str, Any]
    ) -> None:
        """Validate input data."""

        # not one to one
        # error
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
        
        # not one to one
        for attr_name, attr_values in sensitive_features.items():
            if len(attr_values) != len(y_true):
                raise ValueError(
                    f"Length mismatch: {attr_name} ({len(attr_values)}) vs y_true ({len(y_true)})"
                )
        
        # Check for binary classification
        unique_true = set(np.unique(y_true))
        unique_pred = set(np.unique(y_pred))
        
        if not unique_true.issubset({0, 1}):
            raise ValueError(f"y_true must be binary (0/1), got: {unique_true}")
        if not unique_pred.issubset({0, 1}):
            raise ValueError(f"y_pred must be binary (0/1), got: {unique_pred}")
    
    def _analyze_attribute(
        self,
        y_true: np.ndarray, # in the parent function we convert all file types to NpArray
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray,
        attr_name: str
    ) -> tuple: # constant, we are returning two variables. python returns them as tuple types
        """
        Analyze bias for a single sensitive attribute.
        
        Returns:
            Tuple of (metrics_dict, list_of_violations)
        """
        violations = []
        
        # Create MetricFrame for group-level analysis
        # Reminder: MetricFrame is imported from FareLearn
        # the code for this Object is mostly copy and pasted 
        metric_frame = MetricFrame(
            metrics={
                "count": count,
                "selection_rate": selection_rate,
                "accuracy": accuracy_score,
                "precision": lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
                "recall": lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_feature
        )
        
        # Get group-level metrics
        # metric_frame.by_group --> pandas
        # we convert it to a dict using to_dict()
        group_metrics = metric_frame.by_group.to_dict()
        #then, we list only the sensitive features
        groups = list(metric_frame.by_group.index)

        # Fix after initial test run: diabetic_data had groups with very few samples (e.g., "Unknown/Invalid" gender)
        # These tiny groups cause spurious violations (e.g., 100% difference with 0% prediction rate)
        # Solution: Filter out groups with sample size below minimum threshold
        MIN_GROUP_SIZE = 30  # Statistical minimum for reliable metrics (similar to Central Limit Theorem)
        small_groups = []
        for group in groups:
            group_size = int(group_metrics["count"][group])
            if group_size < MIN_GROUP_SIZE:
                small_groups.append((group, group_size))

        if small_groups:
            import warnings
            warnings.warn(
                f"Excluding {len(small_groups)} group(s) from '{attr_name}' analysis due to small sample size (< {MIN_GROUP_SIZE}): "
                f"{[(g, n) for g, n in small_groups]}. "
                f"Metrics on tiny groups are unreliable and create false violations.",
                UserWarning
            )
            # Filter out small groups
            groups = [g for g in groups if int(group_metrics["count"][g]) >= MIN_GROUP_SIZE]

        if len(groups) < 2:
            # Not enough groups to compare - skip this attribute
            return {}, []  # Return empty metrics and no violations
        
        # Format group metrics nicely
        # before: {"accuracy": {"male": 0.8, "female": 0.9}, "count": ...}
        # after: {"male": {"accuracy": 0.8, "count": ...}, "female": ...} ----> organized by groups
        formatted_group_metrics = {}
        for group in groups:
            formatted_group_metrics[str(group)] = {
                "count": int(group_metrics["count"][group]), # returns NpArray, we convert it to int. if not it will crash while outputting as JSON file
                "selection_rate": round(float(group_metrics["selection_rate"][group]), 4),
                "accuracy": round(float(group_metrics["accuracy"][group]), 4),
                "precision": round(float(group_metrics["precision"][group]), 4),
                "recall": round(float(group_metrics["recall"][group]), 4),
                "tpr": round(float(group_metrics["tpr"][group]) if not np.isnan(group_metrics["tpr"][group]) else 0, 4),
                "fpr": round(float(group_metrics["fpr"][group]) if not np.isnan(group_metrics["fpr"][group]) else 0, 4),
            }
        
        # Calculate fairness metrics
        
        # 1. Demographic Parity Ratio
        dp_ratio = demographic_parity_ratio(
            y_true, y_pred, sensitive_features=sensitive_feature
        )
        
        # 2. Demographic Parity Difference
        dp_diff = demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_feature
        )
        
        # 3. Equalized Odds Difference
        try:
            eo_diff = equalized_odds_difference(
                y_true, y_pred, sensitive_features=sensitive_feature
            )
        except Exception:
            eo_diff = None  # May fail if no positive samples in a group
        
        # 4. Equal Opportunity Difference (TPR difference)
        tpr_values = [formatted_group_metrics[str(g)]["tpr"] for g in groups]
        eop_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0
        
        # Check for violations
        
        # Demographic Parity Ratio
        if dp_ratio < self.config["demographic_parity_ratio_min"]:
            violations.append(self._create_violation(
                metric="demographic_parity_ratio",
                attribute=attr_name,
                value=dp_ratio,
                threshold=self.config["demographic_parity_ratio_min"],
                group_values={str(g): formatted_group_metrics[str(g)]["selection_rate"] for g in groups},
                direction="below"
            ))
        elif dp_ratio > self.config["demographic_parity_ratio_max"]:
            violations.append(self._create_violation(
                metric="demographic_parity_ratio",
                attribute=attr_name,
                value=dp_ratio,
                threshold=self.config["demographic_parity_ratio_max"],
                group_values={str(g): formatted_group_metrics[str(g)]["selection_rate"] for g in groups},
                direction="above"
            ))
        
        # Demographic Parity Difference
        if abs(dp_diff) > self.config["demographic_parity_diff_threshold"]:
            violations.append(self._create_violation(
                metric="demographic_parity_difference",
                attribute=attr_name,
                value=dp_diff,
                threshold=self.config["demographic_parity_diff_threshold"],
                group_values={str(g): formatted_group_metrics[str(g)]["selection_rate"] for g in groups},
                direction="exceeds"
            ))
        
        # Equalized Odds
        if eo_diff is not None and eo_diff > self.config["equalized_odds_threshold"]:
            violations.append(self._create_violation(
                metric="equalized_odds_difference",
                attribute=attr_name,
                value=eo_diff,
                threshold=self.config["equalized_odds_threshold"],
                group_values={str(g): {"tpr": formatted_group_metrics[str(g)]["tpr"], 
                                       "fpr": formatted_group_metrics[str(g)]["fpr"]} for g in groups},
                direction="exceeds"
            ))
        
        # Equal Opportunity
        if eop_diff > self.config["equal_opportunity_threshold"]:
            violations.append(self._create_violation(
                metric="equal_opportunity_difference",
                attribute=attr_name,
                value=eop_diff,
                threshold=self.config["equal_opportunity_threshold"],
                group_values={str(g): formatted_group_metrics[str(g)]["tpr"] for g in groups},
                direction="exceeds"
            ))
        
        # Compile metrics summary
        metrics = {
            "demographic_parity_ratio": round(float(dp_ratio), 4),
            "demographic_parity_difference": round(float(dp_diff), 4),
            "equalized_odds_difference": round(float(eo_diff), 4) if eo_diff is not None else None,
            "equal_opportunity_difference": round(float(eop_diff), 4),
            "group_metrics": formatted_group_metrics
        }
        
        return metrics, violations
    
    def _create_violation(
        self,
        metric: str,
        attribute: str,
        value: float,
        threshold: float,
        group_values: Dict,
        direction: str
    ) -> BiasViolation:
        """Create a BiasViolation with healthcare-specific interpretation."""
        
        # Determine severity
        diff = abs(value - threshold) if direction != "exceeds" else abs(value)
        if diff > self.config["severity_thresholds"]["HIGH"]:
            severity = "HIGH"
        elif diff > self.config["severity_thresholds"]["MEDIUM"]:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            metric, attribute, value, threshold, group_values, direction
        )
        
        return BiasViolation(
            metric=metric,
            attribute=attribute,
            value=value,
            threshold=threshold,
            severity=severity,
            interpretation=interpretation,
            group_values=group_values
        )
    
    def _generate_interpretation(
        self,
        metric: str,
        attribute: str,
        value: float,
        threshold: float,
        group_values: Dict,
        direction: str
    ) -> str:
        """Generate healthcare-specific interpretation of a violation."""
        
        pos_label = self.config["positive_outcome_label"]
        
        if metric == "demographic_parity_ratio":
            # Find which group has lower selection rate
            if isinstance(list(group_values.values())[0], dict):
                rates = {k: v.get("selection_rate", v) for k, v in group_values.items()}
            else:
                rates = group_values
            
            min_group = min(rates, key=rates.get)
            max_group = max(rates, key=rates.get)
            pct_diff = (1 - value) * 100
            
            return (
                f"{min_group.title()} patients flagged as {pos_label} "
                f"{pct_diff:.0f}% less often than {max_group} patients"
            )
        
        elif metric == "demographic_parity_difference":
            if isinstance(list(group_values.values())[0], dict):
                rates = {k: v.get("selection_rate", v) for k, v in group_values.items()}
            else:
                rates = group_values
            
            min_group = min(rates, key=rates.get)
            diff_pct = abs(value) * 100
            
            return (
                f"Prediction rate differs by {diff_pct:.1f}% across {attribute} groups; "
                f"{min_group} patients may be under-identified"
            )
        
        elif metric == "equalized_odds_difference":
            return (
                f"Model accuracy differs by {value*100:.1f}% across {attribute} groups; "
                f"TPR and/or FPR are unequal"
            )
        
        elif metric == "equal_opportunity_difference":
            # Find group with lower TPR
            if isinstance(list(group_values.values())[0], dict):
                tprs = {k: v.get("tpr", v) for k, v in group_values.items()}
            else:
                tprs = group_values
            
            min_group = min(tprs, key=tprs.get)
            diff_pct = value * 100
            
            return (
                f"Model misses {diff_pct:.0f}% more actual {pos_label} cases in "
                f"{min_group} patients (lower sensitivity)"
            )
        
        return f"{metric} violation: {value:.4f} (threshold: {threshold})"
    
    def _generate_recommendations(self, violations: List[BiasViolation]) -> List[str]:
        """Generate actionable recommendations based on violations."""
        
        if not violations:
            return ["No bias violations detected. Continue monitoring with new data."]
        
        recommendations = []
        
        # Get unique attributes and metrics with violations
        attrs_with_violations = set(v.attribute for v in violations)
        metrics_with_violations = set(v.metric for v in violations)
        high_severity = any(v.severity == "HIGH" for v in violations)
        
        # General recommendations
        if high_severity:
            recommendations.append(
                "⚠️ HIGH SEVERITY: Flag all predictions for manual clinical review "
                "until bias is addressed"
            )
        
        # Attribute-specific recommendations
        for attr in attrs_with_violations:
            attr_violations = [v for v in violations if v.attribute == attr]
            
            recommendations.append(
                f"Review training data for {attr}-based representation imbalance"
            )
            
            # Find disadvantaged group
            for v in attr_violations:
                if isinstance(list(v.group_values.values())[0], dict):
                    continue  # Complex structure, skip
                min_group = min(v.group_values, key=v.group_values.get)
                recommendations.append(
                    f"Consider threshold adjustment for {min_group} patient predictions"
                )
                break
        
        # Metric-specific recommendations
        if "equal_opportunity_difference" in metrics_with_violations:
            recommendations.append(
                "Sensitivity differs across groups — investigate features correlated "
                "with protected attributes"
            )
        
        if "equalized_odds_difference" in metrics_with_violations:
            recommendations.append(
                "Consider fairness-aware training methods (e.g., reweighting, "
                "adversarial debiasing)"
            )
        
        # Always recommend
        recommendations.append(
            "Document bias findings for regulatory compliance (FDA AI/ML guidelines)"
        )
        
        return recommendations


# Quick test
if __name__ == "__main__":
    # Simple test case
    np.random.seed(42)
    n = 200
    
    # Simulated data with intentional bias
    y_true = np.random.binomial(1, 0.5, n)
    sex = np.array(["male"] * 120 + ["female"] * 80)
    
    # Biased predictions: lower positive rate for females
    y_pred = y_true.copy()
    female_mask = sex == "female"
    # Flip some female positives to negative (introduce bias)
    flip_mask = female_mask & (y_true == 1) & (np.random.random(n) < 0.3)
    y_pred[flip_mask] = 0
    
    checker = BiasChecker()
    report = checker.check(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features={"sex": sex}
    )
    
    print(f"Status: {report.overall_status}")
    print(f"Violations: {len(report.violations)}")
    for v in report.violations:
        print(f"  - [{v.severity}] {v.metric}: {v.interpretation}")

