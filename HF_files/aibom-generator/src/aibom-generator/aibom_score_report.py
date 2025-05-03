import json
from typing import Dict

def humanize(text: str) -> str:
    return text.replace('_', ' ').title()

def render_score_html(score_report: Dict[str, any]) -> str:
    max_scores = score_report.get("max_scores", {
        "required_fields": 20,
        "metadata": 20,
        "component_basic": 20,
        "component_model_card": 30,
        "external_references": 10
    })
    
    total_max = 100
    
    html = f"""
    <html>
    <head>
        <title>AIBOM Score Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 60%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background-color: #f9f9f9; }}
            ul {{ list-style: none; padding-left: 0; }}
            li::before {{ content: "\\2713 "; color: green; margin-right: 6px; }}
            li.missing::before {{ content: "\\2717 "; color: red; }}
            details {{ margin-top: 20px; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h2>AIBOM Completeness Score: <strong>{score_report['total_score']}/{total_max}</strong></h2>
        <h3>Section Scores</h3>
        <table>
            <tr><th>Section</th><th>Score</th></tr>
    """
    for section, score in score_report.get("section_scores", {}).items():
        max_score = max_scores.get(section, 0)
        html += f"<tr><td>{humanize(section)}</td><td>{score}/{max_score}</td></tr>"

    html += "</table>"

    if "field_checklist" in score_report:
        html += "<h3>Field Checklist</h3><ul>"
        for field, mark in score_report["field_checklist"].items():
            css_class = "missing" if mark == "✘" else ""
            html += f"<li class=\"{css_class}\">{field}</li>"
        html += "</ul>"

    html += f"""
        <details>
            <summary>Raw Score Report</summary>
            <pre>{json.dumps(score_report, indent=2)}</pre>
        </details>
    </body>
    </html>
    """
    return html

def save_score_report_html(score_report: Dict[str, any], output_path: str):
    html_content = render_score_html(score_report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Score report saved to {output_path}")
