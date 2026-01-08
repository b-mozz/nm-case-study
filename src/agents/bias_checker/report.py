"""
Report generator for the Ethics and Bias Checker Agent.

Exports bias analysis reports to JSON and Markdown formats.
"""

import json
from typing import Optional
from datetime import datetime
from pathlib import Path

# helper to make our new combo names look clean in the report
# turns "sex_&_race" into "Sex + Race"
def _format_attr_name(name: str) -> str:
    if "_&_" in name:
        # handling our new intersectional keys
        return name.replace("_&_", " + ").replace("_", " ").title()
    return name.replace("_", " ").title()

def export_json(report, output_path: str) -> str:
    """
    Export bias report to JSON file.
    
    Args:
        report: BiasReport object
        output_path: Path to save JSON file
    
    Returns:
        Path to created file
    """
    with open(output_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    
    return output_path


def export_markdown(report, output_path: str) -> str:
    """
    Export bias report to Markdown file.
    
    Args:
        report: BiasReport object
        output_path: Path to save Markdown file
    
    Returns:
        Path to created file
    """
    lines = []
    
    # Header
    lines.append("# Bias Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append(f"**Status:** {'🔴 ' + report.overall_status if report.overall_status == 'BIAS_DETECTED' else '🟢 ' + report.overall_status}")
    lines.append("")
    
    # Dataset Info
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(f"- **Total samples:** {report.dataset_info['total_samples']}")
    lines.append(f"- **Positive cases:** {report.dataset_info['positive_count']} ({report.dataset_info['positive_rate']*100:.1f}%)")
    lines.append(f"- **Negative cases:** {report.dataset_info['negative_count']}")
    lines.append(f"- **Model positive rate:** {report.dataset_info['prediction_positive_rate']*100:.1f}%")
    
    # fix: filter out the messy combo names from the summary list so it stays clean
    # we only want to show the "Base" attributes here
    all_groups = report.dataset_info['groups_analyzed']
    base_groups = [g for g in all_groups if "_&_" not in g]
    combo_count = len(all_groups) - len(base_groups)
    
    lines.append(f"- **Base Attributes:** {', '.join(base_groups)}")
    # brag that we did intersectional analysis
    if combo_count > 0:
        lines.append(f"- **Intersectional Combinations:** {combo_count} additional groups analyzed")
    lines.append("")
    
    # Violations
    if report.violations:
        lines.append("## ⚠️ Bias Violations Detected")
        lines.append("")
        
        for i, v in enumerate(report.violations, 1):
            severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[v.severity]
            # use our helper to make the title look pro
            attr_display = _format_attr_name(v.attribute)
            lines.append(f"### {i}. {v.metric.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"- **Attribute:** {attr_display}")
            lines.append(f"- **Severity:** {severity_emoji} {v.severity}")
            lines.append(f"- **Value:** {v.value:.4f}")
            lines.append(f"- **Threshold:** {v.threshold}")
            lines.append(f"- **Interpretation:** {v.interpretation}")
            lines.append("")
            
            # Group breakdown
            if v.group_values:
                lines.append("**Group breakdown:**")
                lines.append("")
                for group, val in v.group_values.items():
                    if isinstance(val, dict):
                        val_str = ", ".join(f"{k}={v:.3f}" for k, v in val.items())
                        lines.append(f"- {group}: {val_str}")
                    else:
                        lines.append(f"- {group}: {val:.4f}")
                lines.append("")
    else:
        lines.append("## ✅ No Bias Violations Detected")
        lines.append("")
        lines.append("All fairness metrics are within acceptable thresholds.")
        lines.append("")
    
    # Detailed Metrics - SPLIT into two sections for readability
    # first, filter the keys
    single_keys = [k for k in report.metrics.keys() if "_&_" not in k]
    intersectional_keys = [k for k in report.metrics.keys() if "_&_" in k]

    # Helper function to generate the table rows (DRY principle!)
    def _append_metrics_for_keys(keys_list):
        for attr_name in keys_list:
            metrics = report.metrics[attr_name]
            # use the helper again
            lines.append(f"### {_format_attr_name(attr_name)}")
            lines.append("")
            
            # Fairness metrics table
            lines.append("| Metric | Value | Status |")
            lines.append("|--------|-------|--------|")
            
            dp_ratio = metrics.get("demographic_parity_ratio")
            dp_diff = metrics.get("demographic_parity_difference")
            eo_diff = metrics.get("equalized_odds_difference")
            eop_diff = metrics.get("equal_opportunity_difference")
            
            if dp_ratio is not None:
                status = "✅" if 0.8 <= dp_ratio <= 1.25 else "❌"
                lines.append(f"| Demographic Parity Ratio | {dp_ratio:.4f} | {status} |")
            
            if dp_diff is not None:
                status = "✅" if abs(dp_diff) <= 0.1 else "❌"
                lines.append(f"| Demographic Parity Diff | {dp_diff:.4f} | {status} |")
            
            if eo_diff is not None:
                status = "✅" if eo_diff <= 0.1 else "❌"
                lines.append(f"| Equalized Odds Diff | {eo_diff:.4f} | {status} |")
            
            if eop_diff is not None:
                status = "✅" if eop_diff <= 0.1 else "❌"
                lines.append(f"| Equal Opportunity Diff | {eop_diff:.4f} | {status} |")
            
            lines.append("")

    # Section 1: Base Attributes
    lines.append("## Detailed Metrics: Single Attributes")
    lines.append("")
    _append_metrics_for_keys(single_keys)

    # Section 2: Intersectional (only if we have them)
    if intersectional_keys:
        lines.append("## Detailed Metrics: Intersectional Analysis")
        lines.append("_Analysis of combined demographic groups (e.g. Race + Gender)_")
        lines.append("")
        _append_metrics_for_keys(intersectional_keys)
    
    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    for rec in report.recommendations:
        lines.append(f"- {rec}")
    lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by Ethics and Bias Checker Agent*")
    lines.append(f"*Using Fairlearn metrics library*")
    
    # Write file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return output_path


def print_summary(report) -> None:
    """
    Print a colored summary of the bias report to console.
    """
    # ANSI colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    print()
    print(f"{BOLD}{'='*60}{END}")
    print(f"{BOLD}  BIAS ANALYSIS REPORT{END}")
    print(f"{'='*60}")
    print()
    
    # Status
    if report.overall_status == "BIAS_DETECTED":
        print(f"  Status: {RED}{BOLD}⚠ BIAS DETECTED{END}")
    else:
        print(f"  Status: {GREEN}{BOLD}✓ PASS{END}")
    
    print()
    
    # Dataset info
    print(f"  {CYAN}Dataset:{END}")
    print(f"    Samples: {report.dataset_info['total_samples']}")
    print(f"    Positive rate: {report.dataset_info['positive_rate']*100:.1f}%")
    
    # nicer display for console too
    all_groups = report.dataset_info['groups_analyzed']
    base_groups = [g for g in all_groups if "_&_" not in g]
    print(f"    Attributes: {', '.join(base_groups)}")
    if len(all_groups) > len(base_groups):
        print(f"    Intersectional: Yes (+{len(all_groups) - len(base_groups)} combinations)")
    print()
    
    # Violations
    if report.violations:
        print(f"  {CYAN}Violations ({len(report.violations)}):{END}")
        for v in report.violations:
            if v.severity == "HIGH":
                color = RED
            elif v.severity == "MEDIUM":
                color = YELLOW
            else:
                color = GREEN
            
            # format the attribute name for console
            attr_nice = _format_attr_name(v.attribute)
            print(f"    {color}[{v.severity}]{END} {v.metric}")
            print(f"          {BOLD}{attr_nice}{END}: {v.interpretation}")
        print()
    
    # Key metrics
    print(f"  {CYAN}Key Metrics:{END}")
    for attr, metrics in report.metrics.items():
        # skip intersectional details in the console summary to keep it short
        # unless it has a violation? nah, keep it simple.
        if "_&_" in attr: 
            continue 
            
        print(f"    {_format_attr_name(attr)}:")
        dp_ratio = metrics.get("demographic_parity_ratio")
        if dp_ratio is not None:
            status = f"{GREEN}✓{END}" if 0.8 <= dp_ratio <= 1.25 else f"{RED}✗{END}"
            print(f"      Demographic Parity Ratio: {dp_ratio:.3f} {status}")
        
        eop_diff = metrics.get("equal_opportunity_difference")
        if eop_diff is not None:
            status = f"{GREEN}✓{END}" if eop_diff <= 0.1 else f"{RED}✗{END}"
            print(f"      Equal Opportunity Diff:   {eop_diff:.3f} {status}")
    
    # simple footer to let them know
    has_intersectional = any("_&_" in k for k in report.metrics.keys())
    if has_intersectional:
         print(f"    {YELLOW}(Intersectional metrics available in full report){END}")

    print()
    
    # Top recommendation
    if report.recommendations:
        print(f"  {CYAN}Top Recommendation:{END}")
        print(f"    {report.recommendations[0]}")
    
    print()
    print(f"{'='*60}")
    print()