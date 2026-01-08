"""
Interactive Remediation Wizard for Data Quality Agent

Provides user-friendly prompts to fix detected data quality issues.

Author: Bimukti Mozzumdar
Date: January 2025
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class Fix:
    """Represents a single remediation action."""
    issue_type: str
    column: Optional[str]
    action: str
    parameters: Dict[str, Any]
    description: str


class RemediationWizard:
    """
    Interactive wizard to fix data quality issues.

    Usage:
        wizard = RemediationWizard(df, validation_report)
        df_cleaned = wizard.run()
    """

    def __init__(self, df: pd.DataFrame, report):
        """
        Initialize wizard.

        Args:
            df: Original DataFrame
            report: ValidationReport from DataQualityValidationAgent
        """
        self.df = df.copy()  # Work on a copy
        self.report = report
        self.fixes: List[Fix] = []
        self.issue_count = 0

    def run(self) -> pd.DataFrame:
        """
        Run interactive wizard and apply fixes.

        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*60)
        print("🔧 INTERACTIVE REMEDIATION WIZARD")
        print("="*60)
        print(f"\nFound {len(self.report.critical_issues)} critical issues and {len(self.report.warnings)} warnings.")
        print("Let's fix them interactively.\n")

        # Group issues by category
        all_issues = self.report.critical_issues + self.report.warnings
        issue_groups = self._group_issues(all_issues)

        # Show summary and let user choose which categories to handle
        print("--- ISSUE SUMMARY ---\n")
        for i, (category, issues) in enumerate(issue_groups.items(), 1):
            print(f"{i}. {category}: {len(issues)} issue(s)")
        print()

        # Process each category
        for category, issues in issue_groups.items():
            print("\n" + "="*60)
            print(f"Category: {category.upper()} ({len(issues)} issue(s))")
            print("="*60)
            print(f"Do you want to handle {category} issues? (y/n/skip_all): ", end="")

            response = input().strip().lower()

            if response == 'skip_all':
                print("⏭️  Skipping all remaining categories.\n")
                break
            elif response == 'n':
                print(f"⏭️  Skipping {category} issues.\n")
                continue
            elif response == 'y':
                print()
                for issue in issues:
                    self.issue_count += 1
                    self._handle_issue(issue, is_critical=(issue in self.report.critical_issues))

        # Apply all fixes
        if self.fixes:
            print("\n" + "="*60)
            print(f"📝 Summary: {len(self.fixes)} fixes to apply")
            print("="*60)
            for i, fix in enumerate(self.fixes, 1):
                print(f"{i}. {fix.description}")

            print(f"\nApply all fixes? (y/n): ", end="")
            if input().strip().lower() == 'y':
                df_cleaned = self._apply_fixes()
                print("✅ All fixes applied successfully!")
                return df_cleaned
            else:
                print("❌ Fixes cancelled. Returning original data.")
                return self.df
        else:
            print("\n✅ No fixes to apply (all skipped).")
            return self.df

    def _group_issues(self, issues) -> dict:
        """Group issues by category for batch handling."""
        groups = {
            "Empty Columns": [],
            "Missing Values": [],
            "Type Inconsistencies": [],
            "Domain Violations": [],
            "Duplicate Data": [],
            "Outliers & Anomalies": [],
            "Other": []
        }

        for issue in issues:
            if issue.issue_type == "EMPTY_COLUMN":
                groups["Empty Columns"].append(issue)
            elif issue.issue_type in ["HIGH_NULL_RATE", "MODERATE_NULL_RATE", "LOW_NULL_RATE"]:
                groups["Missing Values"].append(issue)
            elif issue.issue_type == "TYPE_INCONSISTENCY":
                groups["Type Inconsistencies"].append(issue)
            elif issue.issue_type in ["VALUE_BELOW_RANGE", "VALUE_ABOVE_RANGE"]:
                groups["Domain Violations"].append(issue)
            elif issue.issue_type in ["DUPLICATE_ROWS", "DUPLICATE_IDS", "UNNAMED_COLUMNS"]:
                groups["Duplicate Data"].append(issue)
            elif issue.issue_type in ["STATISTICAL_OUTLIER", "MULTIVARIATE_ANOMALY"]:
                groups["Outliers & Anomalies"].append(issue)
            else:
                groups["Other"].append(issue)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _handle_issue(self, issue, is_critical: bool):
        """Route issue to appropriate handler."""
        if issue.issue_type == "EMPTY_COLUMN":
            self._prompt_empty_column(issue)
        elif issue.issue_type in ["HIGH_NULL_RATE", "MODERATE_NULL_RATE"]:
            self._prompt_missing_values(issue)
        elif issue.issue_type == "TYPE_INCONSISTENCY":
            self._prompt_type_inconsistency(issue)
        elif issue.issue_type in ["VALUE_BELOW_RANGE", "VALUE_ABOVE_RANGE"]:
            self._prompt_domain_violation(issue)
        elif issue.issue_type == "DUPLICATE_ROWS":
            self._prompt_duplicate_rows(issue)
        elif issue.issue_type == "DUPLICATE_IDS":
            self._prompt_duplicate_ids(issue)
        elif issue.issue_type == "UNNAMED_COLUMNS":
            self._prompt_unnamed_columns(issue)
        elif issue.issue_type == "STATISTICAL_OUTLIER":
            self._prompt_outliers(issue)
        elif issue.issue_type == "MULTIVARIATE_ANOMALY":
            self._prompt_anomalies(issue)
        else:
            # Generic handler
            print(f"[{self.issue_count}] {issue.description}")
            print("   (No automated fix available - skip)\n")

    def _prompt_empty_column(self, issue):
        """Handle empty columns."""
        print(f"[{self.issue_count}] {issue.description}")
        print(f"   Column: '{issue.column}'")
        print("   Options:")
        print("     1. Drop column entirely")
        print("     2. Keep as-is")
        print("     3. Skip")

        choice = self._get_choice([1, 2, 3])

        if choice == 1:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=issue.column,
                action="drop_column",
                parameters={},
                description=f"Drop empty column '{issue.column}'"
            ))
        print()

    def _prompt_missing_values(self, issue):
        """Handle moderate missing values."""
        print(f"[{self.issue_count}] {issue.description}")
        print(f"   Column: '{issue.column}'")

        # Check if numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(self.df[issue.column])

        print("   Options:")
        if is_numeric:
            print("     1. Fill with median")
            print("     2. Fill with mean")
            print("     3. Drop rows with missing values")
            print("     4. Skip")
            choice = self._get_choice([1, 2, 3, 4])

            if choice == 1:
                self.fixes.append(Fix(
                    issue_type=issue.issue_type,
                    column=issue.column,
                    action="fill_median",
                    parameters={},
                    description=f"Fill '{issue.column}' missing values with median"
                ))
            elif choice == 2:
                self.fixes.append(Fix(
                    issue_type=issue.issue_type,
                    column=issue.column,
                    action="fill_mean",
                    parameters={},
                    description=f"Fill '{issue.column}' missing values with mean"
                ))
            elif choice == 3:
                self.fixes.append(Fix(
                    issue_type=issue.issue_type,
                    column=issue.column,
                    action="drop_rows",
                    parameters={},
                    description=f"Drop rows where '{issue.column}' is missing"
                ))
        else:
            print("     1. Fill with mode (most common value)")
            print("     2. Drop rows with missing values")
            print("     3. Skip")
            choice = self._get_choice([1, 2, 3])

            if choice == 1:
                self.fixes.append(Fix(
                    issue_type=issue.issue_type,
                    column=issue.column,
                    action="fill_mode",
                    parameters={},
                    description=f"Fill '{issue.column}' missing values with mode"
                ))
            elif choice == 2:
                self.fixes.append(Fix(
                    issue_type=issue.issue_type,
                    column=issue.column,
                    action="drop_rows",
                    parameters={},
                    description=f"Drop rows where '{issue.column}' is missing"
                ))
        print()

    def _prompt_type_inconsistency(self, issue):
        """Handle type inconsistencies."""
        print(f"[{self.issue_count}] {issue.description}")
        print(f"   Column: '{issue.column}'")
        if issue.examples:
            print(f"   Examples: {issue.examples[:3]}")

        print("   Options:")
        print("     1. Convert to numeric, fill invalid with median")
        print("     2. Convert to numeric, drop invalid rows")
        print("     3. Convert to numeric, leave as NaN")
        print("     4. Skip")

        choice = self._get_choice([1, 2, 3, 4])

        if choice == 1:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=issue.column,
                action="convert_numeric_fill_median",
                parameters={},
                description=f"Convert '{issue.column}' to numeric, fill invalid with median"
            ))
        elif choice == 2:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=issue.column,
                action="convert_numeric_drop",
                parameters={},
                description=f"Convert '{issue.column}' to numeric, drop invalid rows"
            ))
        elif choice == 3:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=issue.column,
                action="convert_numeric_nan",
                parameters={},
                description=f"Convert '{issue.column}' to numeric, leave invalid as NaN"
            ))
        print()

    def _prompt_domain_violation(self, issue):
        """Handle domain violations (out of range)."""
        print(f"[{self.issue_count}] {issue.description}")
        print(f"   Column: '{issue.column}'")
        if issue.examples:
            print(f"   Examples: {issue.examples[:3]}")

        print("   Options:")
        print("     1. Cap values at valid range")
        print("     2. Replace with median")
        print("     3. Drop rows")
        print("     4. Skip")

        choice = self._get_choice([1, 2, 3, 4])

        # Extract min/max from issue description or use defaults
        if "below minimum" in issue.description.lower():
            direction = "below"
        else:
            direction = "above"

        if choice == 1:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=issue.column,
                action="cap_values",
                parameters={"direction": direction},
                description=f"Cap '{issue.column}' values at valid range"
            ))
        elif choice == 2:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=issue.column,
                action="replace_median",
                parameters={"indices": issue.affected_row_indices},
                description=f"Replace invalid '{issue.column}' values with median"
            ))
        elif choice == 3:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=issue.column,
                action="drop_rows",
                parameters={"indices": issue.affected_row_indices},
                description=f"Drop rows with invalid '{issue.column}' values"
            ))
        print()

    def _prompt_duplicate_rows(self, issue):
        """Handle duplicate rows."""
        print(f"[{self.issue_count}] {issue.description}")

        print("   Options:")
        print("     1. Keep first occurrence, drop duplicates")
        print("     2. Keep last occurrence, drop duplicates")
        print("     3. Keep all (no action)")

        choice = self._get_choice([1, 2, 3])

        if choice == 1:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=None,
                action="drop_duplicates",
                parameters={"keep": "first"},
                description="Drop duplicate rows (keep first occurrence)"
            ))
        elif choice == 2:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=None,
                action="drop_duplicates",
                parameters={"keep": "last"},
                description="Drop duplicate rows (keep last occurrence)"
            ))
        print()

    def _prompt_duplicate_ids(self, issue):
        """Handle duplicate IDs (CRITICAL)."""
        print(f"[{self.issue_count}] ⚠️  CRITICAL: {issue.description}")
        print(f"   Column: '{issue.column}'")
        if issue.examples:
            print(f"   Examples: {issue.examples[:3]}")

        print("\n   ⚠️  WARNING: This is critical. Manual review strongly recommended.")
        print("   Options:")
        print("     1. Keep first occurrence only (⚠️ data loss)")
        print("     2. Stop - manual review required (recommended)")

        choice = self._get_choice([1, 2])

        if choice == 1:
            print("   ⚠️  Are you sure? This will delete data. Type 'yes' to confirm: ", end="")
            confirm = input().strip().lower()
            if confirm == 'yes':
                self.fixes.append(Fix(
                    issue_type=issue.issue_type,
                    column=issue.column,
                    action="drop_duplicate_ids",
                    parameters={"id_column": issue.column},
                    description=f"Drop duplicate IDs in '{issue.column}' (keep first)"
                ))
            else:
                print("   Cancelled. Skipping this fix.")
        elif choice == 2:
            print("   ⚠️  Stopping wizard. Please review duplicate IDs manually.")
            raise KeyboardInterrupt("Manual review required for duplicate IDs")
        print()

    def _prompt_unnamed_columns(self, issue):
        """Handle unnamed columns."""
        print(f"[{self.issue_count}] {issue.description}")
        if issue.examples:
            unnamed_cols = [ex.get('column') for ex in issue.examples if 'column' in ex]
            print(f"   Columns: {unnamed_cols}")

        print("   Options:")
        print("     1. Drop all unnamed columns")
        print("     2. Skip")

        choice = self._get_choice([1, 2])

        if choice == 1:
            unnamed_cols = [col for col in self.df.columns if 'Unnamed' in str(col)]
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=None,
                action="drop_unnamed_columns",
                parameters={"columns": unnamed_cols},
                description=f"Drop unnamed columns: {unnamed_cols}"
            ))
        print()

    def _prompt_outliers(self, issue):
        """Handle statistical outliers."""
        print(f"[{self.issue_count}] {issue.description}")
        print(f"   Column: '{issue.column}'")
        if issue.examples:
            print(f"   Examples: {issue.examples[:3]}")

        print("   Options:")
        print("     1. Cap outliers at 3 standard deviations")
        print("     2. Replace with median")
        print("     3. Keep all (legitimate extreme values)")
        print("     4. Drop outlier rows")

        choice = self._get_choice([1, 2, 3, 4])

        if choice == 1:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=issue.column,
                action="cap_outliers",
                parameters={"threshold": 3.0},
                description=f"Cap '{issue.column}' outliers at ±3 std dev"
            ))
        elif choice == 2:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=issue.column,
                action="replace_outliers_median",
                parameters={"indices": issue.affected_row_indices},
                description=f"Replace '{issue.column}' outliers with median"
            ))
        elif choice == 4:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=issue.column,
                action="drop_rows",
                parameters={"indices": issue.affected_row_indices},
                description=f"Drop rows with '{issue.column}' outliers"
            ))
        print()

    def _prompt_anomalies(self, issue):
        """Handle ML-detected anomalies."""
        print(f"[{self.issue_count}] {issue.description}")

        print("   Options:")
        print("     1. Export anomalies to CSV for review")
        print("     2. Drop all anomaly rows (⚠️ risky)")
        print("     3. Keep all (no action)")

        choice = self._get_choice([1, 2, 3])

        if choice == 1:
            self.fixes.append(Fix(
                issue_type=issue.issue_type,
                column=None,
                action="export_anomalies",
                parameters={"indices": issue.affected_row_indices},
                description="Export anomalies to anomalies.csv"
            ))
        elif choice == 2:
            print("   ⚠️  Are you sure? This will delete data. Type 'yes' to confirm: ", end="")
            confirm = input().strip().lower()
            if confirm == 'yes':
                self.fixes.append(Fix(
                    issue_type=issue.issue_type,
                    column=None,
                    action="drop_rows",
                    parameters={"indices": issue.affected_row_indices},
                    description=f"Drop {len(issue.affected_row_indices)} anomaly rows"
                ))
            else:
                print("   Cancelled. Skipping this fix.")
        print()

    def _get_choice(self, valid_choices: List[int]) -> int:
        """Get valid choice from user."""
        while True:
            try:
                print(f"   Your choice ({'/'.join(map(str, valid_choices))}): ", end="")
                choice = int(input().strip())
                if choice in valid_choices:
                    return choice
                else:
                    print(f"   Invalid choice. Please enter one of: {valid_choices}")
            except ValueError:
                print("   Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\n⚠️  Wizard interrupted by user.")
                raise

    def _apply_fixes(self) -> pd.DataFrame:
        """Apply all queued fixes to DataFrame."""
        df = self.df.copy()

        for fix in self.fixes:
            try:
                if fix.action == "drop_column":
                    df = df.drop(columns=[fix.column])

                elif fix.action == "fill_median":
                    median_val = df[fix.column].median()
                    df[fix.column] = df[fix.column].fillna(median_val)

                elif fix.action == "fill_mean":
                    mean_val = df[fix.column].mean()
                    df[fix.column] = df[fix.column].fillna(mean_val)

                elif fix.action == "fill_mode":
                    mode_val = df[fix.column].mode()[0] if len(df[fix.column].mode()) > 0 else None
                    if mode_val is not None:
                        df[fix.column] = df[fix.column].fillna(mode_val)

                elif fix.action == "drop_rows":
                    if "indices" in fix.parameters:
                        df = df.drop(index=fix.parameters["indices"], errors='ignore')
                    else:
                        df = df.dropna(subset=[fix.column])

                elif fix.action == "convert_numeric_fill_median":
                    df[fix.column] = pd.to_numeric(df[fix.column], errors='coerce')
                    median_val = df[fix.column].median()
                    df[fix.column] = df[fix.column].fillna(median_val)

                elif fix.action == "convert_numeric_drop":
                    numeric_col = pd.to_numeric(df[fix.column], errors='coerce')
                    df = df[numeric_col.notna()]

                elif fix.action == "convert_numeric_nan":
                    df[fix.column] = pd.to_numeric(df[fix.column], errors='coerce')

                elif fix.action == "cap_values":
                    # Get clinical ranges from issue description (simplified)
                    # For now, cap at current min/max of valid values
                    median_val = df[fix.column].median()
                    mean_val = df[fix.column].mean()
                    std_val = df[fix.column].std()

                    # Cap at ±3 std dev from mean
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    df[fix.column] = df[fix.column].clip(lower=lower_bound, upper=upper_bound)

                elif fix.action == "replace_median":
                    median_val = df[fix.column].median()
                    indices = fix.parameters.get("indices", [])
                    df.loc[indices, fix.column] = median_val

                elif fix.action == "drop_duplicates":
                    df = df.drop_duplicates(keep=fix.parameters["keep"])

                elif fix.action == "drop_duplicate_ids":
                    df = df.drop_duplicates(subset=[fix.parameters["id_column"]], keep='first')

                elif fix.action == "drop_unnamed_columns":
                    df = df.drop(columns=fix.parameters["columns"], errors='ignore')

                elif fix.action == "cap_outliers":
                    mean_val = df[fix.column].mean()
                    std_val = df[fix.column].std()
                    threshold = fix.parameters["threshold"]

                    lower_bound = mean_val - threshold * std_val
                    upper_bound = mean_val + threshold * std_val
                    df[fix.column] = df[fix.column].clip(lower=lower_bound, upper=upper_bound)

                elif fix.action == "replace_outliers_median":
                    median_val = df[fix.column].median()
                    indices = fix.parameters.get("indices", [])
                    df.loc[indices, fix.column] = median_val

                elif fix.action == "export_anomalies":
                    indices = fix.parameters.get("indices", [])
                    anomalies_df = df.loc[indices]
                    anomalies_df.to_csv("anomalies.csv", index=True)
                    print(f"   ✅ Exported {len(indices)} anomalies to anomalies.csv")

            except Exception as e:
                print(f"   ⚠️  Error applying fix '{fix.description}': {e}")
                continue

        return df


# Convenience function
def run_remediation_wizard(df: pd.DataFrame, report) -> pd.DataFrame:
    """
    Run interactive remediation wizard.

    Args:
        df: DataFrame to clean
        report: ValidationReport from DataQualityValidationAgent

    Returns:
        Cleaned DataFrame
    """
    wizard = RemediationWizard(df, report)
    return wizard.run()
