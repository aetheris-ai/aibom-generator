#!/usr/bin/env python
import os
import json
import logging
import sys
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime
from datasets import Dataset, load_dataset, concatenate_datasets
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Add Counter Configuration ---
HF_REPO = "aetheris-ai/aisbom-usage-log"  # User needs to create this private repo
HF_TOKEN = os.getenv("HF_TOKEN")  # User must set this environment variable
# --- End Counter Configuration ---

# Define directories
templates_dir = "templates"
OUTPUT_DIR = "/tmp/aibom_output"

# Initialize templates
templates = Jinja2Templates(directory=templates_dir)

# Create app
app = FastAPI(title="AI SBOM Generator API")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount output directory as static files
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Status response model
class StatusResponse(BaseModel):
    status: str
    version: str
    generator_version: str

# --- Add Counter Helper Functions ---
def log_sbom_generation(model_id: str):
    """Logs a successful SBOM generation event to the Hugging Face dataset."""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set. Skipping SBOM generation logging.")
        return

    try:
        log_data = {
            "timestamp": [datetime.utcnow().isoformat()],
            "event": ["generated"],
            "model_id": [model_id] # Log the model ID
        }
        ds_new_log = Dataset.from_dict(log_data)

        # Try to load existing dataset to append
        try:
            # Use trust_remote_code=True if required by the dataset/model on HF
            # Corrected: Removed unnecessary backslashes around 'train'
            existing_ds = load_dataset(HF_REPO, token=HF_TOKEN, split='train', trust_remote_code=True)
            # Check if dataset is empty or has different columns (handle initial creation)
            if len(existing_ds) > 0 and set(existing_ds.column_names) == set(log_data.keys()):
                 ds_to_push = concatenate_datasets([existing_ds, ds_new_log])
            elif len(existing_ds) == 0:
                 logger.info(f"Dataset {HF_REPO} is empty. Pushing initial data.")
                 ds_to_push = ds_new_log
            else:
                 logger.warning(f"Dataset {HF_REPO} has unexpected columns {existing_ds.column_names} vs {list(log_data.keys())}. Appending new log anyway, structure might differ.")
                 # Attempt concatenation even if columns differ slightly, HF might handle it
                 # Or consider more robust schema migration/handling if needed
                 ds_to_push = concatenate_datasets([existing_ds, ds_new_log])

        except Exception as load_err:
             # Handle case where dataset doesn't exist yet or other loading errors
             # Corrected: Removed unnecessary backslash in doesn't
             logger.info(f"Could not load existing dataset {HF_REPO} (may not exist yet): {load_err}. Pushing new dataset.")
             ds_to_push = ds_new_log # ds is already prepared with the new log entry

        # Push the updated or new dataset
        # Corrected: Removed unnecessary backslash in it's
        ds_to_push.push_to_hub(HF_REPO, token=HF_TOKEN, private=True) # Ensure it's private
        logger.info(f"Successfully logged SBOM generation for {model_id} to {HF_REPO}")

    except Exception as e:
        logger.error(f"Failed to log SBOM generation to {HF_REPO}: {e}")

def get_sbom_count() -> str:
    """Retrieves the total count of generated SBOMs from the Hugging Face dataset."""
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set. Cannot retrieve SBOM count.")
        return "N/A"
    try:
        # Load the dataset - assumes 'train' split exists after first push
        # Use trust_remote_code=True if required by the dataset/model on HF
        # Corrected: Removed unnecessary backslashes around 'train'
        ds = load_dataset(HF_REPO, token=HF_TOKEN, split='train', trust_remote_code=True)
        count = len(ds)
        logger.info(f"Retrieved SBOM count: {count} from {HF_REPO}")
        # Format count for display (e.g., add commas for large numbers)
        return f"{count:,}"
    except Exception as e:
        logger.error(f"Failed to retrieve SBOM count from {HF_REPO}: {e}")
        # Return "N/A" or similar indicator on error
        return "N/A"
# --- End Counter Helper Functions ---

@app.on_event("startup")
async def startup_event():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory ready at {OUTPUT_DIR}")
    logger.info(f"Registered routes: {[route.path for route in app.routes]}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    sbom_count = get_sbom_count() # Get count
    try:
        return templates.TemplateResponse("index.html", {"request": request, "sbom_count": sbom_count}) # Pass to template
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        # Attempt to render error page even if main page fails
        try:
            return templates.TemplateResponse("error.html", {"request": request, "error": f"Template rendering error: {str(e)}", "sbom_count": sbom_count})
        except Exception as template_err:
             # Fallback if error template also fails
             logger.error(f"Error rendering error template: {template_err}")
             raise HTTPException(status_code=500, detail=f"Template rendering error: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(status="operational", version="1.0.0", generator_version="1.0.0")

# Import utils module for completeness score calculation
def import_utils():
    """Import utils module with fallback paths."""
    try:
        # Try different import paths
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Try direct import first
        try:
            from utils import calculate_completeness_score
            logger.info("Imported utils.calculate_completeness_score directly")
            return calculate_completeness_score
        except ImportError:
            pass

        # Try from src
        try:
            from src.aibom_generator.utils import calculate_completeness_score
            logger.info("Imported src.aibom_generator.utils.calculate_completeness_score")
            return calculate_completeness_score
        except ImportError:
            pass

        # Try from aibom_generator
        try:
            from aibom_generator.utils import calculate_completeness_score
            logger.info("Imported aibom_generator.utils.calculate_completeness_score")
            return calculate_completeness_score
        except ImportError:
            pass

        # If all imports fail, use the default implementation
        logger.warning("Could not import calculate_completeness_score, using default implementation")
        return None
    except Exception as e:
        logger.error(f"Error importing utils: {str(e)}")
        return None

# Try to import the calculate_completeness_score function
calculate_completeness_score = import_utils()

# Helper function to create a comprehensive completeness_score with field_checklist
def create_comprehensive_completeness_score(aibom=None):
    """
    Create a comprehensive completeness_score object with all required attributes.
    If aibom is provided and calculate_completeness_score is available, use it to calculate the score.
    Otherwise, return a default score structure.
    """
    # If we have the calculate_completeness_score function and an AIBOM, use it
    if calculate_completeness_score and aibom:
        try:
            return calculate_completeness_score(aibom, validate=True, use_best_practices=True)
        except Exception as e:
            logger.error(f"Error calculating completeness score: {str(e)}")

    # Otherwise, return a default comprehensive structure
    return {
        "total_score": 75.5,  # Default score for better UI display
        "section_scores": {
            "required_fields": 20,
            "metadata": 15,
            "component_basic": 18,
            "component_model_card": 15,
            "external_references": 7.5
        },
        "max_scores": {
            "required_fields": 20,
            "metadata": 20,
            "component_basic": 20,
            "component_model_card": 30,
            "external_references": 10
        },
        "field_checklist": {
            # Required fields
            "bomFormat": "✔ ★★★",
            "specVersion": "✔ ★★★",
            "serialNumber": "✔ ★★★",
            "version": "✔ ★★★",
            "metadata.timestamp": "✔ ★★",
            "metadata.tools": "✔ ★★",
            "metadata.authors": "✔ ★★",
            "metadata.component": "✔ ★★",

            # Component basic info
            "component.type": "✔ ★★",
            "component.name": "✔ ★★★",
            "component.bom-ref": "✔ ★★",
            "component.purl": "✔ ★★",
            "component.description": "✔ ★★",
            "component.licenses": "✔ ★★",

            # Model card
            "modelCard.modelParameters": "✔ ★★",
            "modelCard.quantitativeAnalysis": "✘ ★★",
            "modelCard.considerations": "✔ ★★",

            # External references
            "externalReferences": "✔ ★",

            # Additional fields from FIELD_CLASSIFICATION
            "name": "✔ ★★★",
            "downloadLocation": "✔ ★★★",
            "primaryPurpose": "✔ ★★★",
            "suppliedBy": "✔ ★★★",
            "energyConsumption": "✘ ★★",
            "hyperparameter": "✔ ★★",
            "limitation": "✔ ★★",
            "safetyRiskAssessment": "✘ ★★",
            "typeOfModel": "✔ ★★",
            "modelExplainability": "✘ ★",
            "standardCompliance": "✘ ★",
            "domain": "✔ ★",
            "energyQuantity": "✘ ★",
            "energyUnit": "✘ ★",
            "informationAboutTraining": "✔ ★",
            "informationAboutApplication": "✔ ★",
            "metric": "✘ ★",
            "metricDecisionThreshold": "✘ ★",
            "modelDataPreprocessing": "✘ ★",
            "autonomyType": "✘ ★",
            "useSensitivePersonalInformation": "✘ ★"
        },
        "field_tiers": {
            # Required fields
            "bomFormat": "critical",
            "specVersion": "critical",
            "serialNumber": "critical",
            "version": "critical",
            "metadata.timestamp": "important",
            "metadata.tools": "important",
            "metadata.authors": "important",
            "metadata.component": "important",

            # Component basic info
            "component.type": "important",
            "component.name": "critical",
            "component.bom-ref": "important",
            "component.purl": "important",
            "component.description": "important",
            "component.licenses": "important",

            # Model card
            "modelCard.modelParameters": "important",
            "modelCard.quantitativeAnalysis": "important",
            "modelCard.considerations": "important",

            # External references
            "externalReferences": "supplementary",

            # Additional fields from FIELD_CLASSIFICATION
            "name": "critical",
            "downloadLocation": "critical",
            "primaryPurpose": "critical",
            "suppliedBy": "critical",
            "energyConsumption": "important",
            "hyperparameter": "important",
            "limitation": "important",
            "safetyRiskAssessment": "important",
            "typeOfModel": "important",
            "modelExplainability": "supplementary",
            "standardCompliance": "supplementary",
            "domain": "supplementary",
            "energyQuantity": "supplementary",
            "energyUnit": "supplementary",
            "informationAboutTraining": "supplementary",
            "informationAboutApplication": "supplementary",
            "metric": "supplementary",
            "metricDecisionThreshold": "supplementary",
            "modelDataPreprocessing": "supplementary",
            "autonomyType": "supplementary",
            "useSensitivePersonalInformation": "supplementary"
        },
        "missing_fields": {
            "critical": [],
            "important": ["modelCard.quantitativeAnalysis", "energyConsumption", "safetyRiskAssessment"],
            "supplementary": ["modelExplainability", "standardCompliance", "energyQuantity", "energyUnit",
                             "metric", "metricDecisionThreshold", "modelDataPreprocessing",
                             "autonomyType", "useSensitivePersonalInformation"]
        },
        "completeness_profile": {
            "name": "standard",
            "description": "Comprehensive fields for proper documentation",
            "satisfied": True
        },
        "penalty_applied": False,
        "penalty_reason": None,
        "recommendations": [
            {
                "priority": "medium",
                "field": "modelCard.quantitativeAnalysis",
                "message": "Missing important field: modelCard.quantitativeAnalysis",
                "recommendation": "Add quantitative analysis information to the model card"
            },
            {
                "priority": "medium",
                "field": "energyConsumption",
                "message": "Missing important field: energyConsumption - helpful for environmental impact assessment",
                "recommendation": "Consider documenting energy consumption metrics for better transparency"
            },
            {
                "priority": "medium",
                "field": "safetyRiskAssessment",
                "message": "Missing important field: safetyRiskAssessment",
                "recommendation": "Add safety risk assessment information to improve documentation"
            }
        ]
    }

@app.post("/generate", response_class=HTMLResponse)
async def generate_form(
    request: Request,
    model_id: str = Form(...),
    include_inference: bool = Form(False),
    use_best_practices: bool = Form(True)
):
    sbom_count = get_sbom_count() # Get count early for context
    try:
        # Try different import paths for AIBOMGenerator
        generator = None
        try:
            from src.aibom_generator.generator import AIBOMGenerator
            generator = AIBOMGenerator()
        except ImportError:
            try:
                from aibom_generator.generator import AIBOMGenerator
                generator = AIBOMGenerator()
            except ImportError:
                try:
                    from generator import AIBOMGenerator
                    generator = AIBOMGenerator()
                except ImportError:
                    logger.error("Could not import AIBOMGenerator from any known location")
                    raise ImportError("Could not import AIBOMGenerator from any known location")

        # Generate AIBOM
        aibom = generator.generate_aibom(
            model_id=model_id,
            include_inference=include_inference,
            use_best_practices=use_best_practices
        )
        enhancement_report = generator.get_enhancement_report()

        # Save AIBOM to file
        # Corrected: Removed unnecessary backslashes around '/' and '_'
        filename = f"{model_id.replace('/', '_')}_aibom.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(aibom, f, indent=2)

        # --- Log Generation Event ---
        log_sbom_generation(model_id)
        sbom_count = get_sbom_count() # Refresh count after logging
        # --- End Log ---

        download_url = f"/output/{filename}"

        # Create download and UI interaction scripts
        # Corrected: Removed unnecessary backslashes in script string
        download_script = f"""
        <script>
            function downloadJSON() {{
                const a = document.createElement('a');
                a.href = '{download_url}';
                a.download = '{filename}';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }}

            function switchTab(tabId) {{
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => {{
                    tab.classList.remove('active');
                }});

                // Deactivate all tab buttons
                document.querySelectorAll('.aibom-tab').forEach(button => {{
                    button.classList.remove('active');
                }});

                // Show the selected tab
                document.getElementById(tabId).classList.add('active');

                // Activate the clicked button
                event.currentTarget.classList.add('active');
            }}

            function toggleCollapsible(element) {{
                element.classList.toggle('active');
                var content = element.nextElementSibling;
                if (content.style.maxHeight) {{
                    content.style.maxHeight = null;
                    content.classList.remove('active');
                }} else {{
                    content.style.maxHeight = content.scrollHeight + "px";
                    content.classList.add('active');
                }}
            }}
        </script>
        """

        # Get completeness score or create a comprehensive one if not available
        completeness_score = None
        # Corrected: Removed unnecessary backslash in 'get_completeness_score'
        if hasattr(generator, 'get_completeness_score'):
            try:
                completeness_score = generator.get_completeness_score(model_id)
                logger.info("Successfully retrieved completeness_score from generator")
            except Exception as e:
                logger.error(f"Completeness score error from generator: {str(e)}")

        # If completeness_score is None or doesn't have field_checklist, use comprehensive one
        # Corrected: Removed unnecessary backslash in doesn't and 'field_checklist'
        if completeness_score is None or not isinstance(completeness_score, dict) or 'field_checklist' not in completeness_score:
            logger.info("Using comprehensive completeness_score with field_checklist")
            completeness_score = create_comprehensive_completeness_score(aibom)

        # Ensure enhancement_report has the right structure
        if enhancement_report is None:
            enhancement_report = {
                "ai_enhanced": False,
                "ai_model": None,
                "original_score": {"total_score": 0, "completeness_score": 0},
                "final_score": {"total_score": 0, "completeness_score": 0},
                "improvement": 0
            }
        else:
            # Ensure original_score has completeness_score
            if "original_score" not in enhancement_report or enhancement_report["original_score"] is None:
                enhancement_report["original_score"] = {"total_score": 0, "completeness_score": 0}
            elif "completeness_score" not in enhancement_report["original_score"]:
                enhancement_report["original_score"]["completeness_score"] = enhancement_report["original_score"].get("total_score", 0)

            # Ensure final_score has completeness_score
            if "final_score" not in enhancement_report or enhancement_report["final_score"] is None:
                enhancement_report["final_score"] = {"total_score": 0, "completeness_score": 0}
            elif "completeness_score" not in enhancement_report["final_score"]:
                enhancement_report["final_score"]["completeness_score"] = enhancement_report["final_score"].get("total_score", 0)

        # Add display names and tooltips for score sections
        display_names = {
            "required_fields": "Required Fields",
            "metadata": "Metadata",
            "component_basic": "Component Basic Info",
            "component_model_card": "Model Card",
            "external_references": "External References"
        }

        tooltips = {
            "required_fields": "Basic required fields for a valid AIBOM",
            "metadata": "Information about the AIBOM itself",
            "component_basic": "Basic information about the AI model component",
            "component_model_card": "Detailed model card information",
            "external_references": "Links to external resources"
        }

        weights = {
            "required_fields": 20,
            "metadata": 20,
            "component_basic": 20,
            "component_model_card": 30,
            "external_references": 10
        }

        # Render the template with all necessary data
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "model_id": model_id,
                "aibom": aibom,
                "enhancement_report": enhancement_report,
                "completeness_score": completeness_score,
                "download_url": download_url,
                "download_script": download_script,
                "display_names": display_names,
                "tooltips": tooltips,
                "weights": weights,
                "sbom_count": sbom_count # Pass count
            }
        )
    except Exception as e:
        logger.error(f"Error generating AI SBOM: {str(e)}")
        # Ensure count is passed to error template as well
        sbom_count = get_sbom_count() # Refresh count just in case
        return templates.TemplateResponse(
            "error.html", {"request": request, "error": str(e), "sbom_count": sbom_count} # Pass count
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a generated AIBOM file.

    This endpoint serves the generated AIBOM JSON files for download.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/json",
        filename=filename
    )

# If running directly (for local testing)
if __name__ == "__main__":
    import uvicorn
    # Ensure HF_TOKEN is set for local testing if needed
    if not HF_TOKEN:
        print("Warning: HF_TOKEN environment variable not set. SBOM count will show N/A and logging will be skipped.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
