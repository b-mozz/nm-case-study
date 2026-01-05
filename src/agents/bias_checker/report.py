"""
Report generator for the Ethics and Bias Checker Agent.

Exports bias analysis reports to JSON and Markdown formats.
"""

import json
from typing import Optional
from datetime import datetime
from pathlib import Path


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
    lines.append(f"- **Attributes analyzed:** {', '.join(report.dataset_info['groups_analyzed'])}")
    lines.append("")
    
    # Violations
    if report.violations:
        lines.append("## ⚠️ Bias Violations Detected")
        lines.append("")
        
        for i, v in enumerate(report.violations, 1):
            severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[v.severity]
            lines.append(f"### {i}. {v.metric.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"- **Attribute:** {v.attribute}")
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
    
    # Metrics by Attribute
    lines.append("## Detailed Metrics by Attribute")
    lines.append("")
    
    for attr_name, metrics in report.metrics.items():
        lines.append(f"### {attr_name.replace('_', ' ').title()}")
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
        
        # Group metrics table
        if "group_metrics" in metrics:
            lines.append("**Performance by group:**")
            lines.append("")
            lines.append("| Group | Count | Selection Rate | Accuracy | TPR | FPR |")
            lines.append("|-------|-------|----------------|----------|-----|-----|")
            
            for group, gm in metrics["group_metrics"].items():
                lines.append(
                    f"| {group} | {gm['count']} | {gm['selection_rate']:.3f} | "
                    f"{gm['accuracy']:.3f} | {gm['tpr']:.3f} | {gm['fpr']:.3f} |"
                )
            
            lines.append("")
    
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
    print(f"    Attributes: {', '.join(report.dataset_info['groups_analyzed'])}")
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
            
            print(f"    {color}[{v.severity}]{END} {v.metric}")
            print(f"          {v.interpretation}")
        print()
    
    # Key metrics
    print(f"  {CYAN}Key Metrics:{END}")
    for attr, metrics in report.metrics.items():
        print(f"    {attr}:")
        dp_ratio = metrics.get("demographic_parity_ratio")
        if dp_ratio is not None:
            status = f"{GREEN}✓{END}" if 0.8 <= dp_ratio <= 1.25 else f"{RED}✗{END}"
            print(f"      Demographic Parity Ratio: {dp_ratio:.3f} {status}")
        
        eop_diff = metrics.get("equal_opportunity_difference")
        if eop_diff is not None:
            status = f"{GREEN}✓{END}" if eop_diff <= 0.1 else f"{RED}✗{END}"
            print(f"      Equal Opportunity Diff:   {eop_diff:.3f} {status}")
    print()
    
    # Top recommendation
    if report.recommendations:
        print(f"  {CYAN}Top Recommendation:{END}")
        print(f"    {report.recommendations[0]}")
    
    print()
    print(f"{'='*60}")
    print()