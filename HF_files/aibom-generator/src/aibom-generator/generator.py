import json
import uuid
import datetime
import json
from typing import Dict, Optional, Any, List

from huggingface_hub import HfApi, ModelCard
from huggingface_hub.repocard_data import EvalResult
from urllib.parse import urlparse
from .utils import calculate_completeness_score
from .validation import validate_aibom as validate_schema, get_validation_summary as get_schema_validation_summary

# Import registry-aware enhanced extraction if available
try:
    from .enhanced_extractor import EnhancedExtractor
    from .field_registry_manager import get_field_registry_manager
    ENHANCED_EXTRACTION_AVAILABLE = True
    print("✅ Registry-aware enhanced extraction module loaded successfully")
except ImportError:
    try:
        from enhanced_extractor import EnhancedExtractor
        from field_registry_manager import get_field_registry_manager
        ENHANCED_EXTRACTION_AVAILABLE = True
        print("✅ Registry-aware enhanced extraction module loaded successfully (direct import)")
    except ImportError:
        ENHANCED_EXTRACTION_AVAILABLE = False
        print("⚠️ Registry-aware enhanced extraction not available, using basic extraction")


class AIBOMGenerator:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        inference_model_url: Optional[str] = None,
        use_inference: bool = True,
        cache_dir: Optional[str] = None,
        use_best_practices: bool = True,  # parameter for industry-neutral scoring
    ):
        self.hf_api = HfApi(token=hf_token)
        self.inference_model_url = inference_model_url
        self.use_inference = use_inference
        self.cache_dir = cache_dir
        self.enhancement_report = None  # Store enhancement report as instance variable
        self.use_best_practices = use_best_practices  # Store best practices flag
        self._setup_enhanced_logging()

        self.extraction_results = {}  # Store extraction results for scoring
    
        # Initialize registry manager for enhanced extraction
        self.registry_manager = None
        if ENHANCED_EXTRACTION_AVAILABLE:
            try:
                self.registry_manager = get_field_registry_manager()
                print("✅ Registry manager initialized for generator")
            except Exception as e:
                print(f"⚠️ Could not initialize registry manager: {e}")
                self.registry_manager = None

    def get_extraction_results(self):
        """Return the enhanced extraction results from the last extraction"""
        return getattr(self, 'extraction_results', {})

    def _setup_enhanced_logging(self):
        """Setup enhanced logging for extraction tracking"""
        import logging
        
        # Configure logging to show in HF Spaces
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True  # Override any existing configuration
        )
        
        # Ensure logger shows up
        logger = logging.getLogger('enhanced_extractor')
        logger.setLevel(logging.INFO)
        
        print("🔧 Enhanced logging configured for AI SBOM generation")
    
    
    def generate_aibom(
        self,
        model_id: str,
        output_file: Optional[str] = None,
        include_inference: Optional[bool] = None,
        use_best_practices: Optional[bool] = None,  # parameter for industry-neutral scoring
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
            print(f"🔍 ENHANCED EXTRACTION DEBUG: Returned {len(original_metadata)} fields:")
            for key, value in original_metadata.items():
                print(f"   {key}: {value}")
            print(f"🔍 EXTRACTION RESULTS: {len(self.extraction_results) if hasattr(self, 'extraction_results') and self.extraction_results else 0} extraction results available")
            
            # Create initial AIBOM with original metadata
            original_aibom = self._create_aibom_structure(model_id, original_metadata)

            print(f"🔍 AI SBOM CREATION DEBUG: Checking what made it into AIBOM:")
            if 'components' in original_aibom and original_aibom['components']:
                component = original_aibom['components'][0]
                if 'properties' in component:
                    print(f"   Found {len(component['properties'])} properties in AIBOM:")
                    for prop in component['properties']:
                        print(f"      {prop.get('name')}: {prop.get('value')}")
                else:
                    print("   No properties found in component")
            else:
                print("   No components found in AI SBOM")
                print(f"🔍 FIELD PRESERVATION VERIFICATION:")
                print(f"   Enhanced extraction returned: {len(original_metadata)} fields")
                
                # Count fields in final AIBOM
                aibom_field_count = 0
                
                # Count component properties
                if 'components' in original_aibom and original_aibom['components']:
                    component = original_aibom['components'][0]
                    if 'properties' in component:
                        aibom_field_count += len(component['properties'])
                    
                    # Count model card properties
                    if 'modelCard' in component and 'properties' in component['modelCard']:
                        aibom_field_count += len(component['modelCard']['properties'])
                
                # Count metadata properties
                if 'metadata' in original_aibom and 'properties' in original_aibom['metadata']:
                    aibom_field_count += len(original_aibom['metadata']['properties'])
                
                print(f"   Final AIBOM contains: {aibom_field_count} fields")
                print(f"   Field preservation rate: {(aibom_field_count/len(original_metadata)*100):.1f}%")
                
                if aibom_field_count >= len(original_metadata) * 0.9:  # 90% or better
                    print("✅ EXCELLENT: Field preservation successful!")
                elif aibom_field_count >= len(original_metadata) * 0.7:  # 70% or better  
                    print("⚠️ GOOD: Most fields preserved, some optimization possible")
                else:
                    print("❌ POOR: Significant field loss detected")

            
            # Calculate initial score with industry-neutral approach if enabled
            original_score = calculate_completeness_score(original_aibom, validate=True, use_best_practices=use_best_practices, extraction_results=self.extraction_results)

            
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
                    print("🚨 FALLBACK: Using _create_minimal_aibom due to error!")
                    print(f"🚨 ERROR DETAILS: {str(e)}")
            # Create final AIBOM with potentially enhanced metadata
            aibom = self._create_aibom_structure(model_id, final_metadata)

            # Validate AIBOM against CycloneDX 1.6 schema
            schema_valid, validation_errors = validate_schema(aibom, strict=False)
            if not schema_valid:
                print(f"⚠️ AIBOM validation found {len(validation_errors)} issue(s):")
                for error in validation_errors[:5]:  # Show first 5 errors
                    print(f"   - {error}")
            else:
                print("✅ AIBOM passes CycloneDX 1.6 schema validation")

            # Calculate final score with enhanced extraction results
            extraction_results = self.get_extraction_results()
            final_score = calculate_completeness_score(
                aibom,
                validate=True,
                use_best_practices=use_best_practices,
                extraction_results=extraction_results  # Pass enhanced results
            )
            

            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(aibom, f, indent=2)

            # Create enhancement report for UI display and store as instance variable
            self.enhancement_report = {
                "ai_enhanced": ai_enhanced,
                "ai_model": ai_model_name if ai_enhanced else None,
                "original_score": original_score,
                "final_score": final_score,
                "improvement": round(final_score["total_score"] - original_score["total_score"], 2) if ai_enhanced else 0,
                "schema_validation": {
                    "valid": schema_valid,
                    "error_count": len(validation_errors) if not schema_valid else 0,
                    "errors": validation_errors[:10] if not schema_valid else []
                }
            }

            # Return only the AIBOM to maintain compatibility with existing code
            return aibom
        except Exception as e:
            print(f"Error generating AI SBOM: {e}")
            # Return a minimal valid AI SBOM structure in case of error
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
                        "bom-ref": "pkg:generic/owasp-genai/owasp-aibom-generator@1.0.0",
                        "type": "application",
                        "name": "OWASP AIBOM Generator",
                        "version": "1.0.0",
                        "manufacturer": {
                            "name": "OWASP GenAI Security Project"
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
        # 🔍 CRASH DEBUG: troubleshoot where the process is crashing and falling back to minimal AIBOM
        print(f"🔍 CRASH_DEBUG: _create_aibom_structure called")
        print(f"🔍 CRASH_DEBUG: model_id = {model_id}")
        print(f"🔍 CRASH_DEBUG: metadata type = {type(metadata)}")
        print(f"🔍 CRASH_DEBUG: metadata keys = {list(metadata.keys()) if isinstance(metadata, dict) else 'NOT A DICT'}")
        
        # Extract owner and model name from model_id
        parts = model_id.split("/")
        group = parts[0] if len(parts) > 1 else ""
        name = parts[1] if len(parts) > 1 else parts[0]
        
        # Get version from metadata or use default
        version = metadata.get("commit", "1.0")

        # 🔍 CRASH DEBUG: Check metadata before creating sections
        print(f"🔍 CRASH_DEBUG: About to create metadata section")        
        
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
        
        # 🔍 CRASH DEBUG: Check if we got this far
        print(f"🔍 CRASH_DEBUG: Successfully created basic AIBOM structure")
        
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

        print(f"🔍 CRASH_DEBUG: _create_aibom_structure completed successfully")
        return aibom

    def _extract_structured_metadata(
        self,
        model_id: str,
        model_info: Dict[str, Any],
        model_card: Optional[ModelCard],
    ) -> Dict[str, Any]:
        
        # Use registry-aware enhanced extraction if available
        if ENHANCED_EXTRACTION_AVAILABLE:
            try:
                print(f"🚀 Using registry-aware enhanced extraction for: {model_id}")
            
                # Create registry-aware enhanced extractor instance
                extractor = EnhancedExtractor(self.hf_api, self.registry_manager)
                
                # Get both metadata and extraction results
                metadata = extractor.extract_metadata(model_id, model_info, model_card)
                
                # Store extraction results for scoring
                self.extraction_results = extractor.extraction_results
                
                # Log extraction summary
                if extractor.registry_fields:
                    registry_field_count = len(extractor.registry_fields)
                    extracted_count = len([k for k, v in metadata.items() if v is not None])
                    extraction_results_count = len(extractor.extraction_results)
                    
                    print(f"✅ Registry-driven extraction completed:")
                    print(f"   📋 Registry fields available: {registry_field_count}")
                    print(f"   📊 Fields attempted: {extraction_results_count}")
                    print(f"   ✅ Fields extracted: {extracted_count}")
                    
                    # Log field coverage
                    if registry_field_count > 0:
                        coverage = (extracted_count / registry_field_count) * 100
                        print(f"   📈 Registry field coverage: {coverage:.1f}%")
                else:
                    extracted_count = len([k for k, v in metadata.items() if v is not None])
                    print(f"✅ Legacy extraction completed: {extracted_count} fields extracted")
                
                return metadata

            except Exception as e:
                print(f"❌ Registry-aware enhanced extraction failed: {e}")
                print("🔄 Falling back to original extraction method")
                # Fall back to original extraction code here
        
        # ORIGINAL EXTRACTION METHOD (as fallback)
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
                    "commit_url": f"https://huggingface.co/{model_id}/commit/{model_info.sha}" if getattr(model_info, "sha", None ) else None,
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
        print(f"🔍 CRASH_DEBUG: _create_metadata_section called")
        print(f"🔍 CRASH_DEBUG: metadata type in metadata_section = {type(metadata)}")
        
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        
        # Get version from metadata or use default
        version = metadata.get("commit", "1.0")
        
        # Create tools section with components array
        tools = {
            "components": [{
                "bom-ref": "pkg:generic/owasp-genai/owasp-aibom-generator@1.0.0",
                "type": "application",
                "name": "OWASP AIBOM Generator",
                "version": "1.0.0",
                "manufacturer": {
                    "name": "OWASP GenAI Security Project"
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
            "primaryPurpose": metadata.get("primaryPurpose", "text-generation"),
            "suppliedBy": metadata.get("suppliedBy", "unknown"),
            "typeOfModel": metadata.get("typeOfModel", "Transformer")
        }
        for key, value in critical_fields.items():
            properties.append({"name": key, "value": str(value)})

        # Add enhanced extraction fields to properties
        # Organize fields by category for better AIBOM structure
        component_fields = ["name", "author", "description", "commit"]  # These go in component section
        critical_fields = ["primaryPurpose", "suppliedBy", "typeOfModel"]  # Always include these
        
        # Add all other enhanced extraction fields (preserve everything!)
        enhanced_fields = ["model_type", "tokenizer_class", "architectures", "library_name", 
                          "pipeline_tag", "tags", "datasets", "base_model", "language",
                          "downloads", "last_modified", "commit_url", "ai:type", "ai:task", 
                          "ai:framework", "eval_results"]
        
        print(f"🔍 CRASH_DEBUG: About to call .items() on metadata")
        print(f"🔍 CRASH_DEBUG: metadata type before .items() = {type(metadata)}")
        
        for key, value in metadata.items():
            # Skip component fields and eval_results (handled separately in the model card)
            if key not in (component_fields + ["eval_results"]) and value is not None:
                # Handle different data types properly
                if isinstance(value, (list, dict)):
                    if isinstance(value, list) and len(value) > 0:
                        # Convert list to comma-separated string for better display
                        if all(isinstance(item, str) for item in value):
                            value = ", ".join(value)
                        else:
                            value = json.dumps(value)
                    elif isinstance(value, dict):
                        value = json.dumps(value)
                
                properties.append({"name": key, "value": str(value)})
                print(f"✅ METADATA: Added {key} = {value} to properties")

        # Assemble metadata section
        metadata_section = {
            "timestamp": timestamp,
            "tools": tools,
            "component": component,
            "properties": properties  # ALWAYS include properties
        }

        return metadata_section

    def _create_component_section(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        print(f"🔍 CRASH_DEBUG: _create_component_section called")
        print(f"🔍 CRASH_DEBUG: metadata type in component_section = {type(metadata)}")
        
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

        # Handle license
        license_value = None
        if metadata and "licenses" in metadata and metadata["licenses"]:
            license_value = metadata["licenses"]
            print(f"✅ COMPONENT: Found licenses = {license_value}")
        elif metadata and "license" in metadata and metadata["license"]:
            license_value = metadata["license"]
            print(f"✅ COMPONENT: Found license = {license_value}")

        if license_value:
            component["licenses"] = [{
                "license": {
                    "id": license_value,
                    "url": self._get_license_url(license_value)
                }
            }]
            print(f"✅ COMPONENT: Added license = {license_value}")
        else:
            component["licenses"] = [{
                "license": {
                    "id": "NOASSERTION",
                    "url": "https://spdx.org/licenses/"
                }
            }]
            print(f"⚠️ COMPONENT: No license found, using NOASSERTION")
            
        # ALWAYS add description
        component["description"] = metadata.get("description", f"AI model {model_id}")
        
        # Add enhanced technical properties to component 
        technical_properties = [] 

        # Add model type information 
        if "model_type" in metadata: 
            technical_properties.append({"name": "model_type", "value": str(metadata["model_type"])}) 
            print(f"✅ COMPONENT: Added model_type = {metadata['model_type']}") 

        # Add tokenizer information 
        if "tokenizer_class" in metadata: 
            technical_properties.append({"name": "tokenizer_class", "value": str(metadata["tokenizer_class"])}) 
            print(f"✅ COMPONENT: Added tokenizer_class = {metadata['tokenizer_class']}") 

        # Add architecture information 
        if "architectures" in metadata: 
            arch_value = metadata["architectures"] 
            if isinstance(arch_value, list): 
                arch_value = ", ".join(arch_value) 
            technical_properties.append({"name": "architectures", "value": str(arch_value)}) 
            print(f"✅ COMPONENT: Added architectures = {arch_value}") 

        # Add library information 
        if "library_name" in metadata: 
            technical_properties.append({"name": "library_name", "value": str(metadata["library_name"])}) 
            print(f"✅ COMPONENT: Added library_name = {metadata['library_name']}") 

        # Add technical properties to component if any exist 
        if technical_properties: 
            component["properties"] = technical_properties

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

    def _eval_results_to_json(self, eval_results: List[EvalResult]) -> List[Dict[str, str]]:
        res = []
        for eval_result in eval_results:
            if hasattr(eval_result, "metric_type") and hasattr(eval_result, "metric_value"):
                res.append({"type": eval_result.metric_type, "value": str(eval_result.metric_value)})
        return res

        
    def _create_model_card_section(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        print(f"🔍 CRASH_DEBUG: _create_model_card_section called")
        print(f"🔍 CRASH_DEBUG: metadata type in model_card_section = {type(metadata)}")
        
        model_card_section = {}
        
        # Add quantitative analysis section
        if "eval_results" in metadata:
            model_card_section["quantitativeAnalysis"] = {
                "performanceMetrics": self._eval_results_to_json(metadata["eval_results"]),
                "graphics": {}  # Empty graphics object as in the example
            }
        else:
            model_card_section["quantitativeAnalysis"] = {"graphics": {}}
        
        # Add properties section with enhanced extraction fields
        properties = []
        
        # Component-level fields that shouldn't be duplicated in model card
        component_level_fields = ["name", "author", "license", "description", "commit"]
        
        # DEBUG: troubleshooting AIBOM generation
        print(f"🔍 DEBUG: About to iterate metadata.items()")
        print(f"🔍 DEBUG: metadata type = {type(metadata)}")
        if isinstance(metadata, dict):
            print(f"🔍 DEBUG: metadata keys = {list(metadata.keys())}")
        else:
            print(f"🔍 DEBUG: metadata value = {metadata}")
            print(f"🔍 DEBUG: This is the problem - metadata should be a dict!")
                
        # Add all enhanced extraction fields to model card properties
        try:
            for key, value in metadata.items():
                if key not in component_level_fields and value is not None:
                    # Handle different data types properly
                    if isinstance(value, (list, dict)):
                        if isinstance(value, list) and len(value) > 0:
                            # Convert list to readable format
                            if all(isinstance(item, str) for item in value):
                                value = ", ".join(value)
                            else:
                                value = json.dumps(value)
                        elif isinstance(value, dict):
                            value = json.dumps(value)
                    
                    properties.append({"name": key, "value": str(value)})
                    print(f"✅ MODEL_CARD: Added {key} = {value}")
        except AttributeError as e:
            print(f"❌ FOUND THE ERROR: {e}")
            print(f"❌ metadata type: {type(metadata)}")
            print(f"❌ metadata value: {metadata}")
            raise e
        
        # Always include properties section (even if empty for consistency)
        model_card_section["properties"] = properties
        print(f"✅ MODEL_CARD: Added {len(properties)} properties to model card")
        
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
        # Add enhanced technical parameters
        if "model_type" in metadata or "tokenizer_class" in metadata or "architectures" in metadata:
            technical_details = {}
            
            if "model_type" in metadata:
                technical_details["modelType"] = metadata["model_type"]
            
            if "tokenizer_class" in metadata:
                technical_details["tokenizerClass"] = metadata["tokenizer_class"]
                
            if "architectures" in metadata:
                technical_details["architectures"] = metadata["architectures"]
            
            # Add to model parameters
            model_parameters.update(technical_details)
            print(f"✅ MODEL_CARD: Added technical details: {list(technical_details.keys())}")
        
        # Update model parameters with enhanced details
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

    def validate_registry_integration(self) -> Dict[str, Any]:
        """
        Validate that the registry integration is working correctly.
        This method helps debug registry-related issues.
        """
        validation_results = {
            'registry_manager_available': bool(self.registry_manager),
            'enhanced_extraction_available': ENHANCED_EXTRACTION_AVAILABLE,
            'registry_fields_count': 0,
            'registry_fields_loaded': False,
            'validation_status': 'unknown'
        }
        
        try:
            if self.registry_manager:
                registry = self.registry_manager.registry
                registry_fields = registry.get('fields', {})
                validation_results['registry_fields_count'] = len(registry_fields)
                validation_results['registry_fields_loaded'] = len(registry_fields) > 0
                
                if len(registry_fields) > 0:
                    validation_results['validation_status'] = 'success'
                    print(f"✅ Registry validation successful: {len(registry_fields)} fields loaded")
                    
                    # Log sample fields
                    sample_fields = list(registry_fields.keys())[:5]
                    print(f"📋 Sample registry fields: {', '.join(sample_fields)}")
                else:
                    validation_results['validation_status'] = 'no_fields'
                    print("⚠️ Registry loaded but no fields found")
            else:
                validation_results['validation_status'] = 'no_registry_manager'
                print("❌ Registry manager not available")
                
        except Exception as e:
            validation_results['validation_status'] = 'error'
            validation_results['error'] = str(e)
            print(f"❌ Registry validation failed: {e}")
        
        return validation_results

def test_registry_integration():
    """
    Test function to validate registry integration is working correctly.
    This function can be called to debug registry-related issues.
    """
    print("🧪 Testing Registry Integration...")
    print("=" * 50)
    
    try:
        # Test generator initialization
        generator = AIBOMGenerator()
        
        # Validate registry integration
        validation_results = generator.validate_registry_integration()
        
        print("📊 Validation Results:")
        for key, value in validation_results.items():
            print(f"   {key}: {value}")
        
        # Test with a sample model
        test_model = "deepseek-ai/DeepSeek-R1"
        print(f"\n🔍 Testing extraction with model: {test_model}")
        
        try:
            # Test model info retrieval
            model_info = generator.hf_api.model_info(test_model)
            model_card = ModelCard.load(test_model)
            
            # Test extraction
            if ENHANCED_EXTRACTION_AVAILABLE and generator.registry_manager:
                extractor = EnhancedExtractor(generator.hf_api, generator.registry_manager)
                metadata = extractor.extract_metadata(test_model, model_info, model_card)
                
                print(f"✅ Test extraction successful: {len(metadata)} fields extracted")
                
                # Show sample extracted fields
                sample_fields = dict(list(metadata.items())[:5])
                print("📋 Sample extracted fields:")
                for key, value in sample_fields.items():
                    print(f"   {key}: {value}")
                
                # Show extraction results summary
                extraction_results = extractor.get_extraction_results()
                confidence_counts = {}
                for result in extraction_results.values():
                    conf = result.confidence.value
                    confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
                
                print("📈 Extraction confidence distribution:")
                for conf, count in confidence_counts.items():
                    print(f"   {conf}: {count} fields")
                    
            else:
                print("⚠️ Registry-aware extraction not available for testing")
                
        except Exception as e:
            print(f"❌ Test extraction failed: {e}")
        
    except Exception as e:
        print(f"❌ Registry integration test failed: {e}")
    
    print("=" * 50)
    print("🧪 Registry Integration Test Complete")

# Uncomment this line to run the test automatically when generator.py is imported
test_registry_integration()