"""
CLI interface for the AIBOM Generator.
"""

import argparse
import json
import os
import sys
from typing import Optional

from aibom_generator.generator import AIBOMGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate AI Bills of Materials (AIBOMs) in CycloneDX format for Hugging Face models."
    )
    
    parser.add_argument(
        "model_id",
        help="Hugging Face model ID (e.g., 'google/bert-base-uncased')"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: <model_id>.aibom.json)",
        default=None
    )
    
    parser.add_argument(
        "--token",
        help="Hugging Face API token for accessing private models",
        default=os.environ.get("HF_TOKEN")
    )
    
    parser.add_argument(
        "--inference-url",
        help="URL of the inference model service for metadata extraction",
        default=os.environ.get("AIBOM_INFERENCE_URL")
    )
    
    parser.add_argument(
        "--no-inference",
        help="Disable inference model for metadata extraction",
        action="store_true"
    )
    
    parser.add_argument(
        "--cache-dir",
        help="Directory to cache API responses and model cards",
        default=os.environ.get("AIBOM_CACHE_DIR", ".aibom_cache")
    )
    
    parser.add_argument(
        "--completeness-threshold",
        help="Minimum completeness score (0-100) required for the AIBOM",
        type=int,
        default=0
    )
    
    parser.add_argument(
        "--format",
        help="Output format (json or yaml)",
        choices=["json", "yaml"],
        default="json"
    )
    
    parser.add_argument(
        "--pretty",
        help="Pretty-print the output",
        action="store_true"
    )

    parser.add_argument(
        "--template-attestation",
        help="Path to JSON file containing external security attestation for chat template",
        default=None
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Determine output file if not specified
    if not args.output:
        model_name = args.model_id.replace("/", "_")
        args.output = f"{model_name}.aibom.json"
    
    # Create the generator
    generator = AIBOMGenerator(
        hf_token=args.token,
        inference_model_url=args.inference_url,
        use_inference=not args.no_inference,
        cache_dir=args.cache_dir
    )
    
    try:
        template_attestation = None
        if args.template_attestation:
            try:
                with open(args.template_attestation, 'r') as f:
                    template_attestation = json.load(f)
                print(f"Loaded template attestation from: {args.template_attestation}")
            except Exception as e:
                print(f"Warning: Could not load template attestation file: {e}", file=sys.stderr)

        # Generate the AIBOM
        aibom = generator.generate_aibom(
            model_id=args.model_id,
            output_file=None,
            template_attestation=template_attestation
        )
        
        # Calculate completeness score (placeholder for now)
        completeness_score = calculate_completeness_score(aibom)
        
        # Check if it meets the threshold
        if completeness_score < args.completeness_threshold:
            print(f"Warning: AIBOM completeness score ({completeness_score}) is below threshold ({args.completeness_threshold})")
        
        # Save the output
        save_output(aibom, args.output, args.format, args.pretty)
        
        print(f"AIBOM generated successfully: {args.output}")
        print(f"Completeness score: {completeness_score}/100")
        
        return 0
    
    except Exception as e:
        print(f"Error generating AIBOM: {e}", file=sys.stderr)
        return 1


def calculate_completeness_score(aibom):
    """
    Calculate a completeness score for the AIBOM.
    
    This is a placeholder implementation that will be replaced with a more
    sophisticated scoring algorithm based on the field mapping framework.
    """
    # TODO: Implement proper completeness scoring
    score = 0
    
    # Check required fields
    if all(field in aibom for field in ["bomFormat", "specVersion", "serialNumber", "version"]):
        score += 20
    
    # Check metadata
    if "metadata" in aibom:
        metadata = aibom["metadata"]
        if "timestamp" in metadata:
            score += 5
        if "tools" in metadata and metadata["tools"]:
            score += 5
        if "authors" in metadata and metadata["authors"]:
            score += 5
        if "component" in metadata:
            score += 5
    
    # Check components
    if "components" in aibom and aibom["components"]:
        component = aibom["components"][0]
        if "type" in component and component["type"] == "machine-learning-model":
            score += 10
        if "name" in component:
            score += 5
        if "bom-ref" in component:
            score += 5
        if "licenses" in component:
            score += 5
        if "externalReferences" in component:
            score += 5
        if "modelCard" in component:
            model_card = component["modelCard"]
            if "modelParameters" in model_card:
                score += 10
            if "quantitativeAnalysis" in model_card:
                score += 10
            if "considerations" in model_card:
                score += 10
    
    return score


def save_output(aibom, output_file, format_type, pretty):
    """Save the AIBOM to the specified output file."""
    if format_type == "json":
        with open(output_file, "w") as f:
            if pretty:
                json.dump(aibom, f, indent=2)
            else:
                json.dump(aibom, f)
    else:  # yaml
        try:
            import yaml
            with open(output_file, "w") as f:
                yaml.dump(aibom, f, default_flow_style=False)
        except ImportError:
            print("Warning: PyYAML not installed. Falling back to JSON format.")
            with open(output_file, "w") as f:
                json.dump(aibom, f, indent=2 if pretty else None)


if __name__ == "__main__":
    sys.exit(main())
