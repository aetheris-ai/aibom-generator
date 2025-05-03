import json
import uuid
import datetime
from typing import Dict, Optional, Any, List


from huggingface_hub import HfApi, ModelCard
from urllib.parse import urlparse
from .utils import calculate_completeness_score


class AIBOMGenerator:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        inference_model_url: Optional[str] = None,
        use_inference: bool = True,
        cache_dir: Optional[str] = None,
        use_best_practices: bool = True,  # Added parameter for industry-neutral scoring
    ):
        self.hf_api = HfApi(token=hf_token)
        self.inference_model_url = inference_model_url
        self.use_inference = use_inference
        self.cache_dir = cache_dir
        self.enhancement_report = None  # Store enhancement report as instance variable
        self.use_best_practices = use_best_practices  # Store best practices flag

    def generate_aibom(
        self,
        model_id: str,
        output_file: Optional[str] = None,
        include_inference: Optional[bool] = None,
        use_best_practices: Optional[bool] = None,  # Added parameter for industry-neutral scoring
    ) -> Dict[str, Any]:
        try:
            model_id = self._normalise_model_id(model_id)
            use_inference = include_inference if include_inference is not None else self.use_inference
            # Use method parameter if provided, otherwise use instance variable
            use_best_practices = use_best_practices if use_best_practices is not None else self.use_best_practices
            
            model_info = self._fetch_model_info(model_id)
            model_card = self._fetch_model_card(model_id)
            
            # Store original metadata before any AI enhancement
            original_metadata = self._extract_structured_metadata(model_id, model_info, model_card)
            
            # Create initial AIBOM with original metadata
            original_aibom = self._create_aibom_structure(model_id, original_metadata)
            
            # Calculate initial score with industry-neutral approach if enabled
            original_score = calculate_completeness_score(original_aibom, validate=True, use_best_practices=use_best_practices)
            
            # Final metadata starts with original metadata
            final_metadata = original_metadata.copy() if original_metadata else {}
            
            # Apply AI enhancement if requested
            ai_enhanced = False
            ai_model_name = None
            
            if use_inference and self.inference_model_url:
                try:
                    # Extract additional metadata using AI
                    enhanced_metadata = self._extract_unstructured_metadata(model_card, model_id)
                    
                    # If we got enhanced metadata, merge it with original
                    if enhanced_metadata:
                        ai_enhanced = True
                        ai_model_name = "BERT-base-uncased"  # Will be replaced with actual model name
                        
                        # Merge enhanced metadata with original (enhanced takes precedence)
                        for key, value in enhanced_metadata.items():
                            if value is not None and (key not in final_metadata or not final_metadata[key]):
                                final_metadata[key] = value
                except Exception as e:
                    print(f"Error during AI enhancement: {e}")
                    # Continue with original metadata if enhancement fails
            
            # Create final AIBOM with potentially enhanced metadata
            aibom = self._create_aibom_structure(model_id, final_metadata)
            
            # Calculate final score with industry-neutral approach if enabled
            final_score = calculate_completeness_score(aibom, validate=True, use_best_practices=use_best_practices)
            
            # Ensure metadata.properties exists
            if "metadata" in aibom and "properties" not in aibom["metadata"]:
                aibom["metadata"]["properties"] = []

            # Note: Quality score information is no longer added to the AIBOM metadata
            # This was removed as requested by the user

            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(aibom, f, indent=2)

            # Create enhancement report for UI display and store as instance variable
            self.enhancement_report = {
                "ai_enhanced": ai_enhanced,
                "ai_model": ai_model_name if ai_enhanced else None,
                "original_score": original_score,
                "final_score": final_score,
                "improvement": round(final_score["total_score"] - original_score["total_score"], 2) if ai_enhanced else 0
            }

            # Return only the AIBOM to maintain compatibility with existing code
            return aibom
        except Exception as e:
            print(f"Error generating AIBOM: {e}")
            # Return a minimal valid AIBOM structure in case of error
            return self._create_minimal_aibom(model_id)

    def _create_minimal_aibom(self, model_id: str) -> Dict[str, Any]:
        """Create a minimal valid AIBOM structure in case of errors"""
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": f"urn:uuid:{str(uuid.uuid4())}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "tools": {
                    "components": [{
                        "bom-ref": "pkg:generic/@cybeats/aetheris-aibom-generator@0.1.0",
                        "type": "application",
                        "name": "aetheris-aibom-generator",
                        "version": "0.1.0",
                        "manufacturer": {
                            "name": "Aetheris AI"
                        }
                    }]
                },
                "component": {
                    "bom-ref": f"pkg:generic/{model_id.replace('/', '%2F')}@1.0",
                    "type": "application",
                    "name": model_id.split("/")[-1],
                    "description": f"AI model {model_id}",
                    "version": "1.0",
                    "purl": f"pkg:generic/{model_id.replace('/', '%2F')}@1.0",
                    "copyright": "NOASSERTION"
                }
            },
            "components": [{
                "bom-ref": f"pkg:huggingface/{model_id.replace('/', '/')}@1.0",
                "type": "machine-learning-model",
                "name": model_id.split("/")[-1],
                "version": "1.0",
                "purl": f"pkg:huggingface/{model_id.replace('/', '/')}@1.0"
            }],
            "dependencies": [{
                "ref": f"pkg:generic/{model_id.replace('/', '%2F')}@1.0",
                "dependsOn": [f"pkg:huggingface/{model_id.replace('/', '/')}@1.0"]
            }]
        }

    def get_enhancement_report(self):
        """Return the enhancement report from the last generate_aibom call"""
        return self.enhancement_report

    def _fetch_model_info(self, model_id: str) -> Dict[str, Any]:
        try:
            return self.hf_api.model_info(model_id)
        except Exception as e:
            print(f"Error fetching model info for {model_id}: {e}")
            return {}

    # ---- new helper ---------------------------------------------------------
    @staticmethod
    def _normalise_model_id(raw_id: str) -> str:
        """
        Accept either  'owner/model'  or a full URL like
        'https://huggingface.co/owner/model'.  Return 'owner/model'.
        """
        if raw_id.startswith(("http://", "https://")):
            path = urlparse(raw_id).path.lstrip("/")
            # path can contain extra segments (e.g. /commit/...), keep first two
            parts = path.split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])
            return path
        return raw_id
    # -------------------------------------------------------------------------

    def _fetch_model_card(self, model_id: str) -> Optional[ModelCard]:
        try:
            return ModelCard.load(model_id)
        except Exception as e:
            print(f"Error fetching model card for {model_id}: {e}")
            return None

    def _create_aibom_structure(
        self,
        model_id: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Extract owner and model name from model_id
        parts = model_id.split("/")
        group = parts[0] if len(parts) > 1 else ""
        name = parts[1] if len(parts) > 1 else parts[0]
        
        # Get version from metadata or use default
        version = metadata.get("commit", "1.0")
        
        aibom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": f"urn:uuid:{str(uuid.uuid4())}",
            "version": 1,
            "metadata": self._create_metadata_section(model_id, metadata),
            "components": [self._create_component_section(model_id, metadata)],
            "dependencies": [
                {
                    "ref": f"pkg:generic/{model_id.replace('/', '%2F')}@{version}",
                    "dependsOn": [f"pkg:huggingface/{model_id.replace('/', '/')}@{version}"]
                }
            ]
        }        
        
        # Add downloadLocation if available
        if metadata and "commit_url" in metadata:
            # Add external reference for downloadLocation
            if "externalReferences" not in aibom:
                aibom["externalReferences"] = []
            
            aibom["externalReferences"].append({
                "type": "distribution",
                "url": f"https://huggingface.co/{model_id}"
            })

        return aibom

    def _extract_structured_metadata(
        self,
        model_id: str,
        model_info: Dict[str, Any],
        model_card: Optional[ModelCard],
    ) -> Dict[str, Any]:
        metadata = {}

        if model_info:
            try:
                metadata.update({
                    "name": model_info.modelId.split("/")[-1] if hasattr(model_info, "modelId") else model_id.split("/")[-1],
                    "author": model_info.author if hasattr(model_info, "author") else None,
                    "tags": model_info.tags if hasattr(model_info, "tags") else [],
                    "pipeline_tag": model_info.pipeline_tag if hasattr(model_info, "pipeline_tag") else None,
                    "downloads": model_info.downloads if hasattr(model_info, "downloads") else 0,
                    "last_modified": model_info.lastModified if hasattr(model_info, "lastModified") else None,
                    "commit": model_info.sha[:7] if hasattr(model_info, "sha") and model_info.sha else None,
                    "commit_url": f"https://huggingface.co/{model_id}/commit/{model_info.sha}" if hasattr(model_info, "sha") and model_info.sha else None,
                })
            except Exception as e:
                print(f"Error extracting model info metadata: {e}")

        if model_card and hasattr(model_card, "data") and model_card.data:
            try:
                card_data = model_card.data.to_dict() if hasattr(model_card.data, "to_dict") else {}
                metadata.update({
                    "language": card_data.get("language"),
                    "license": card_data.get("license"),
                    "library_name": card_data.get("library_name"),
                    "base_model": card_data.get("base_model"),
                    "datasets": card_data.get("datasets"),
                    "model_name": card_data.get("model_name"),
                    "tags": card_data.get("tags", metadata.get("tags", [])),
                    "description": card_data.get("model_summary", None)
                })
                if hasattr(model_card.data, "eval_results") and model_card.data.eval_results:
                    metadata["eval_results"] = model_card.data.eval_results
            except Exception as e:
                print(f"Error extracting model card metadata: {e}")

        metadata["ai:type"] = "Transformer"
        metadata["ai:task"] = metadata.get("pipeline_tag", "Text Generation")
        metadata["ai:framework"] = "PyTorch" if "transformers" in metadata.get("library_name", "") else "Unknown"
        
        # Add fields for industry-neutral scoring (silently aligned with SPDX)
        metadata["primaryPurpose"] = metadata.get("ai:task", "Text Generation")
        metadata["suppliedBy"] = metadata.get("author", "Unknown")
        
        # Add typeOfModel field
        metadata["typeOfModel"] = metadata.get("ai:type", "Transformer")

        return {k: v for k, v in metadata.items() if v is not None}

    def _extract_unstructured_metadata(self, model_card: Optional[ModelCard], model_id: str) -> Dict[str, Any]:
        """
        Extract additional metadata from model card using BERT model.
        This is a placeholder implementation that would be replaced with actual BERT inference.
        
        In a real implementation, this would:
        1. Extract text from model card
        2. Use BERT to identify key information
        3. Structure the extracted information
        
        For now, we'll simulate this with some basic extraction logic.
        """
        enhanced_metadata = {}
        
        # In a real implementation, we would use a BERT model here
        # Since we can't install the required libraries due to space constraints,
        # we'll simulate the enhancement with a placeholder implementation
        
        if model_card and hasattr(model_card, "text") and model_card.text:
            try:
                card_text = model_card.text
                
                # Simulate BERT extraction with basic text analysis
                # In reality, this would be done with NLP models
                
                # Extract description if missing
                if card_text and "description" not in enhanced_metadata:
                    # Take first paragraph that's longer than 20 chars as description
                    paragraphs = [p.strip() for p in card_text.split('\n\n')]
                    for p in paragraphs:
                        if len(p) > 20 and not p.startswith('#'):
                            enhanced_metadata["description"] = p
                            break
                
                # Extract limitations if present
                if "limitations" not in enhanced_metadata:
                    if "## Limitations" in card_text:
                        limitations_section = card_text.split("## Limitations")[1].split("##")[0].strip()
                        if limitations_section:
                            enhanced_metadata["limitations"] = limitations_section
                
                # Extract ethical considerations if present
                if "ethical_considerations" not in enhanced_metadata:
                    for heading in ["## Ethical Considerations", "## Ethics", "## Bias"]:
                        if heading in card_text:
                            section = card_text.split(heading)[1].split("##")[0].strip()
                            if section:
                                enhanced_metadata["ethical_considerations"] = section
                                break
                
                # Extract risks if present
                if "risks" not in enhanced_metadata:
                    if "## Risks" in card_text:
                        risks_section = card_text.split("## Risks")[1].split("##")[0].strip()
                        if risks_section:
                            enhanced_metadata["risks"] = risks_section
                
                # Extract datasets if present
                if "datasets" not in enhanced_metadata:
                    datasets = []
                    if "## Dataset" in card_text or "## Datasets" in card_text:
                        dataset_section = ""
                        if "## Dataset" in card_text:
                            dataset_section = card_text.split("## Dataset")[1].split("##")[0].strip()
                        elif "## Datasets" in card_text:
                            dataset_section = card_text.split("## Datasets")[1].split("##")[0].strip()
                        
                        if dataset_section:
                            # Simple parsing to extract dataset names
                            lines = dataset_section.split("\n")
                            for line in lines:
                                if line.strip() and not line.startswith("#"):
                                    datasets.append({
                                        "type": "dataset",
                                        "name": line.strip().split()[0] if line.strip().split() else "Unknown",
                                        "description": line.strip()
                                    })
                    
                    if datasets:
                        enhanced_metadata["datasets"] = datasets
            except Exception as e:
                print(f"Error extracting unstructured metadata: {e}")
        
        return enhanced_metadata

    def _create_metadata_section(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        
        # Get version from metadata or use default
        version = metadata.get("commit", "1.0")
        
        # Create tools section with components array
        tools = {
            "components": [{
                "bom-ref": "pkg:generic/@cybeats/aetheris-aibom-generator@0.1.0",
                "type": "application",
                "name": "aetheris-aibom-generator",
                "version": "1.0",
                "manufacturer": {
                    "name": "Aetheris AI"
                }
            }]
        }

        # Create authors array
        authors = []
        if "author" in metadata and metadata["author"]:
            authors.append({
                "name": metadata["author"]
            })

        # Create component section for metadata
        component = {
            "bom-ref": f"pkg:generic/{model_id.replace('/', '%2F')}@{version}",
            "type": "application",
            "name": metadata.get("name", model_id.split("/")[-1]),
            "description": metadata.get("description", f"AI model {model_id}"),
            "version": version,
            "purl": f"pkg:generic/{model_id.replace('/', '%2F')}@{version}"
        }
        
        # Add authors to component if available
        if authors:
            component["authors"] = authors
            
        # Add publisher and supplier if author is available
        if "author" in metadata and metadata["author"]:
            component["publisher"] = metadata["author"]
            component["supplier"] = {
                "name": metadata["author"]
            }
            component["manufacturer"] = {
                "name": metadata["author"]
            }
            
        # Add copyright
        component["copyright"] = "NOASSERTION"

        # Create properties array for additional metadata
        properties = []
        for key, value in metadata.items():
            if key not in ["name", "author", "license", "description", "commit"] and value is not None:
                if isinstance(value, (list, dict)):
                    if not isinstance(value, str):
                        value = json.dumps(value)
                properties.append({"name": key, "value": str(value)})

        # Assemble metadata section
        metadata_section = {
            "timestamp": timestamp,
            "tools": tools,
            "component": component
        }

        if properties:
            metadata_section["properties"] = properties

        return metadata_section

    def _create_component_section(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        # Extract owner and model name from model_id
        parts = model_id.split("/")
        group = parts[0] if len(parts) > 1 else ""
        name = parts[1] if len(parts) > 1 else parts[0]
        
        # Get version from metadata or use default
        version = metadata.get("commit", "1.0")
        
        # Create PURL with version information if commit is available
        purl = f"pkg:huggingface/{model_id.replace('/', '/')}"
        if "commit" in metadata:
            purl = f"{purl}@{metadata['commit']}"
        else:
            purl = f"{purl}@{version}"
            
        component = {
            "bom-ref": f"pkg:huggingface/{model_id.replace('/', '/')}@{version}",
            "type": "machine-learning-model",
            "group": group,
            "name": name,
            "version": version,
            "purl": purl
        }

        # Add licenses if available
        if "license" in metadata:
            component["licenses"] = [{
                "license": {
                    "id": metadata["license"],
                    "url": self._get_license_url(metadata["license"])
                }
            }]

        # Add description if available
        if "description" in metadata:
            component["description"] = metadata["description"]

        # Add external references
        external_refs = [{
            "type": "website",
            "url": f"https://huggingface.co/{model_id}"
        }]
        if "commit_url" in metadata:
            external_refs.append({
                "type": "vcs",
                "url": metadata["commit_url"]
            })
        component["externalReferences"] = external_refs

        # Add authors, publisher, supplier, manufacturer
        if "author" in metadata and metadata["author"]:
            component["authors"] = [{"name": metadata["author"]}]
            component["publisher"] = metadata["author"]
            component["supplier"] = {
                "name": metadata["author"],
                "url": [f"https://huggingface.co/{metadata['author']}"]
            }
            component["manufacturer"] = {
                "name": metadata["author"],
                "url": [f"https://huggingface.co/{metadata['author']}"]
            }
            
        # Add copyright
        component["copyright"] = "NOASSERTION"

        # Add model card section
        component["modelCard"] = self._create_model_card_section(metadata)

        return component

    def _create_model_card_section(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        model_card_section = {}
        
        # Add quantitative analysis section
        if "eval_results" in metadata:
            model_card_section["quantitativeAnalysis"] = {
                "performanceMetrics": metadata["eval_results"],
                "graphics": {}  # Empty graphics object as in the example
            }
        else:
            model_card_section["quantitativeAnalysis"] = {"graphics": {}}
        
        # Add properties section
        properties = []
        for key, value in metadata.items():
            if key in ["author", "library_name", "license", "downloads", "likes", "tags", "created_at", "last_modified"]:
                properties.append({"name": key, "value": str(value)})
        
        if properties:
            model_card_section["properties"] = properties
        
        # Create model parameters section
        model_parameters = {}
        
        # Add outputs array
        model_parameters["outputs"] = [{"format": "generated-text"}]
        
        # Add task
        model_parameters["task"] = metadata.get("pipeline_tag", "text-generation")
        
        # Add architecture information
        model_parameters["architectureFamily"] = "llama" if "llama" in metadata.get("name", "").lower() else "transformer"
        model_parameters["modelArchitecture"] = f"{metadata.get('name', 'Unknown')}ForCausalLM"
        
        # Add datasets array with proper structure
        if "datasets" in metadata:
            datasets = []
            if isinstance(metadata["datasets"], list):
                for dataset in metadata["datasets"]:
                    if isinstance(dataset, str):
                        datasets.append({
                            "type": "dataset",
                            "name": dataset,
                            "description": f"Dataset used for training {metadata.get('name', 'the model')}"
                        })
                    elif isinstance(dataset, dict) and "name" in dataset:
                        # Ensure dataset has the required structure
                        dataset_entry = {
                            "type": dataset.get("type", "dataset"),
                            "name": dataset["name"],
                            "description": dataset.get("description", f"Dataset: {dataset['name']}")
                        }
                        datasets.append(dataset_entry)
            elif isinstance(metadata["datasets"], str):
                datasets.append({
                    "type": "dataset",
                    "name": metadata["datasets"],
                    "description": f"Dataset used for training {metadata.get('name', 'the model')}"
                })
                
            if datasets:
                model_parameters["datasets"] = datasets
        
        # Add inputs array
        model_parameters["inputs"] = [{"format": "text"}]
        
        # Add model parameters to model card section
        model_card_section["modelParameters"] = model_parameters
        
        # Add considerations section
        considerations = {}
        for k in ["limitations", "ethical_considerations", "bias", "risks"]:
            if k in metadata:
                considerations[k] = metadata[k]
        if considerations:
            model_card_section["considerations"] = considerations

        return model_card_section
        
    def _get_license_url(self, license_id: str) -> str:
        """Get the URL for a license based on its SPDX ID."""
        license_urls = {
            "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
            "MIT": "https://opensource.org/licenses/MIT",
            "BSD-3-Clause": "https://opensource.org/licenses/BSD-3-Clause",
            "GPL-3.0": "https://www.gnu.org/licenses/gpl-3.0.en.html",
            "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
            "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
            "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
            "CC-BY-ND-4.0": "https://creativecommons.org/licenses/by-nd/4.0/",
            "CC-BY-NC-SA-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "CC-BY-NC-ND-4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
            "LGPL-3.0": "https://www.gnu.org/licenses/lgpl-3.0.en.html",
            "MPL-2.0": "https://www.mozilla.org/en-US/MPL/2.0/",
        }
        
        return license_urls.get(license_id, "https://spdx.org/licenses/")

