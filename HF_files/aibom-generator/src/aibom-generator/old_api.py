import logging
import os
from typing import Dict, List, Optional, Any, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel

from aibom_generator.generator import AIBOMGenerator
    allow_headers=["*"],
)


# Create generator instance
generator = AIBOMGenerator(
    hf_token=os.environ.get("HF_TOKEN"),
    version: str


# Define API endpoints
@app.get("/", response_model=StatusResponse)
async def root():
    """Get API status."""
    return {
        "status": "ok",
        "version": "1.0.0",
    }

@app.post("/generate", response_model=GenerateResponse)

async def generate_aibom(request: GenerateRequest):
    """Generate an AI SBOM for a Hugging Face model."""
    try:
        # Generate the AIBOM
        aibom = generator.generate_aibom(
            model_id=request.model_id,
            include_inference=request.include_inference,
        )
        
        # Calculate completeness score
        completeness_score = calculate_completeness_score(aibom)
        
        # Check if it meets the threshold
        if completeness_score < request.completeness_threshold:
            raise HTTPException(
                status_code=400,
                detail=f"AI SBOM completeness score ({completeness_score}) is below threshold ({request.completeness_threshold})",
            )
        
        return {
            "aibom": aibom,
            "completeness_score": completeness_score,
        )


@app.post("/generate/async")
async def generate_aibom_async(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    """Generate an AI SBOM asynchronously for a Hugging Face model."""
    # Add to background tasks
    background_tasks.add_task(
        _generate_aibom_background,
        request.model_id,
        request.include_inference,
        request.completeness_threshold,
    )
    
    return {
        "status": "accepted",
        "message": f"AI SBOM generation for {request.model_id} started in the background",
    }


async def _generate_aibom_background(
    model_id: str,
    include_inference: Optional[bool] = None,
    completeness_threshold: Optional[int] = 0,
):
    """Generate an AI SBOM in the background."""
    try:
        # Generate the AIBOM
        aibom = generator.generate_aibom(
            model_id=model_id,
            include_inference=include_inference,
        )
        
        # Calculate completeness score
        completeness_score = calculate_completeness_score(aibom)
        
        # TODO: Store the result or notify the user
        logger.info(f"Background AI SBOM generation completed for {model_id}")
        logger.info(f"Completeness score: {completeness_score}")
    except Exception as e:
        logger.error(f"Error in background AI SBOM generation for {model_id}: {e}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))