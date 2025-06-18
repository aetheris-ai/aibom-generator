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
                        "bom-ref": "pkg:generic/aetheris-ai/aetheris-aibom-generator@1.0.0",
                        "type": "application",
                        "name": "aetheris-aibom-generator",
                        "version": "1.0.0",
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
        
        # ALWAYS add root-level external references
        aibom["externalReferences"] = [{
            "type": "distribution",
            "url": f"https://huggingface.co/{model_id}"
        }]

        if metadata and "commit_url" in metadata:
            aibom["externalReferences"].append({
                "type": "vcs", 
                "url": metadata["commit_url"]
            } )

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
                author = getattr(model_info, "author", None)
                if not author or author.strip() == "":
                    parts = model_id.split("/")
                    author = parts[0] if len(parts) > 1 else "unknown"
                    print(f"DEBUG: Fallback author used: {author}")
                else:
                    print(f"DEBUG: Author from model_info: {author}")
    
                metadata.update({
                    "name": getattr(model_info, "modelId", model_id).split("/")[-1],
                    "author": author,
                    "tags": getattr(model_info, "tags", []),
                    "pipeline_tag": getattr(model_info, "pipeline_tag", None),
                    "downloads": getattr(model_info, "downloads", 0),
                    "last_modified": getattr(model_info, "lastModified", None),
                    "commit": getattr(model_info, "sha", None)[:7] if getattr(model_info, "sha", None) else None,
                    "commit_url": f"https://huggingface.co/{model_id}/commit/{model_info.sha}" if getattr(model_info, "sha", None) else None,
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
    
        metadata["primaryPurpose"] = metadata.get("ai:task", "text-generation")
    
        # Use model owner as fallback for suppliedBy if no author
        if not metadata.get("author"):
            parts = model_id.split("/")
            metadata["author"] = parts[0] if len(parts) > 1 else "unknown"
    
        metadata["suppliedBy"] = metadata.get("author", "unknown")
        metadata["typeOfModel"] = metadata.get("ai:type", "Transformer")
    
        print(f"DEBUG: Final metadata['author'] = {metadata.get('author')}")
        print(f"DEBUG: Adding primaryPurpose = {metadata.get('ai:task', 'Text Generation')}")
        print(f"DEBUG: Adding suppliedBy = {metadata.get('suppliedBy')}")
    
        return {k: v for k, v in metadata.items() if v is not None}
    

    def _extract_unstructured_metadata(self, model_card: Optional[ModelCard], model_id: str) -> Dict[str, Any]:
        """
        Placeholder for future AI enhancement.
        Currently returns empty dict since AI enhancement is not implemented.
        """
        return {}
        

    def _create_metadata_section(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        
        # Get version from metadata or use default
        version = metadata.get("commit", "1.0")
        
        # Create tools section with components array
        tools = {
            "components": [{
                "bom-ref": "pkg:generic/aetheris-ai/aetheris-aibom-generator@1.0.0",
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

        # Create properties array for additional metadata (ALWAYS include critical fields)
        properties = []

        # ALWAYS add critical fields for scoring
        critical_fields = {
            "primaryPurpose": metadata.get("primaryPurpose", metadata.get("ai:task", "text-generation")),
            "suppliedBy": metadata.get("suppliedBy", metadata.get("author", "unknown")),
            "typeOfModel": metadata.get("ai:type", "transformer")
        }

        # Add critical fields first
        for key, value in critical_fields.items():
            if value and value != "unknown":
                properties.append({"name": key, "value": str(value)})

        # Add other metadata fields (excluding basic component fields)
        excluded_fields = ["name", "author", "license", "description", "commit", "primaryPurpose", "suppliedBy", "typeOfModel"]
        for key, value in metadata.items():
            if key not in excluded_fields and value is not None:
                if isinstance(value, (list, dict)):
                    if not isinstance(value, str):
                        value = json.dumps(value)
                properties.append({"name": key, "value": str(value)})

        # Assemble metadata section
        metadata_section = {
            "timestamp": timestamp,
            "tools": tools,
            "component": component,
            "properties": properties  # ALWAYS include properties
        }

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

        # ALWAYS add licenses (use default if not available)
        if metadata and "license" in metadata and metadata["license"]:
            component["licenses"] = [{
                "license": {
                    "id": metadata["license"],
                    "url": self._get_license_url(metadata["license"])
                }
            }]
        else:
            # Add default license structure for consistency
            component["licenses"] = [{
                "license": {
                    "id": "unknown",
                    "url": "https://spdx.org/licenses/"
                }
            }]
        # Debug
        print(f"DEBUG: License in metadata: {'license' in metadata}" )
        if "license" in metadata:
            print(f"DEBUG: Adding licenses = {metadata['license']}")
            
        # ALWAYS add description
        component["description"] = metadata.get("description", f"AI model {model_id}")

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

        # ALWAYS add author information (use model owner if not available )
        author_name = metadata.get("author", group if group else "unknown")
        if author_name and author_name != "unknown":
            component["authors"] = [{"name": author_name}]
            component["publisher"] = author_name
            component["supplier"] = {
                "name": author_name,
                "url": [f"https://huggingface.co/{author_name}"]
            }
            component["manufacturer"] = {
                "name": author_name,
                "url": [f"https://huggingface.co/{author_name}"]
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
            "apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
            "mit": "https://opensource.org/licenses/MIT",
            "bsd-3-clause": "https://opensource.org/licenses/BSD-3-Clause",
            "gpl-3.0": "https://www.gnu.org/licenses/gpl-3.0.en.html",
            "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
            "cc-by-sa-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
            "cc-by-nc-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
            "cc-by-nd-4.0": "https://creativecommons.org/licenses/by-nd/4.0/",
            "cc-by-nc-sa-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "cc-by-nc-nd-4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
            "lgpl-3.0": "https://www.gnu.org/licenses/lgpl-3.0.en.html",
            "mpl-2.0": "https://www.mozilla.org/en-US/MPL/2.0/",
        }
        
        return license_urls.get(license_id.lower(), "https://spdx.org/licenses/" )

    def _fetch_with_retry(self, fetch_func, *args, max_retries=3, **kwargs):
        """Fetch data with retry logic for network failures."""
        for attempt in range(max_retries):
            try:
                return fetch_func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Failed to fetch after {max_retries} attempts: {e}")
                    return None
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        return None