import json
from typing import Dict, Optional, Any
from jinja2 import Template

def render_improved_score_template(model_id: str, aibom: Dict[str, Any], completeness_score: Dict[str, Any], enhancement_report: Optional[Dict[str, Any]] = None) -> str:
    """
    Render the improved scoring HTML template with AIBOM data and enhancement information.
    
    Args:
        model_id: The Hugging Face model ID
        aibom: The generated AIBOM data
        completeness_score: The completeness score report
        enhancement_report: Optional enhancement report with AI improvement information
        
    Returns:
        Rendered HTML content
    """
    with open('/home/ubuntu/improved_scoring_template.html', 'r') as f:
        template_str = f.read()
    
    template = Template(template_str)
    
    # Convert scores to percentages for progress bars
    if completeness_score:
        completeness_score['total_score'] = round(completeness_score.get('total_score', 0))
    
    if enhancement_report and enhancement_report.get('original_score'):
        enhancement_report['original_score']['total_score'] = round(enhancement_report['original_score'].get('total_score', 0))
    
    if enhancement_report and enhancement_report.get('final_score'):
        enhancement_report['final_score']['total_score'] = round(enhancement_report['final_score'].get('total_score', 0))
    
    return template.render(
        model_id=model_id,
        aibom=aibom,
        completeness_score=completeness_score,
        enhancement_report=enhancement_report
    )

def save_improved_score_html(model_id: str, aibom: Dict[str, Any], completeness_score: Dict[str, Any], 
                           output_path: str, enhancement_report: Optional[Dict[str, Any]] = None):
    """
    Save the improved scoring HTML to a file.
    
    Args:
        model_id: The Hugging Face model ID
        aibom: The generated AIBOM data
        completeness_score: The completeness score report
        output_path: Path to save the HTML file
        enhancement_report: Optional enhancement report with AI improvement information
    """
    html_content = render_improved_score_template(model_id, aibom, completeness_score, enhancement_report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Improved scoring HTML saved to {output_path}")
