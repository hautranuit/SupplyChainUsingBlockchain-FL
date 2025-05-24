"""
Model Repository for Federated Learning integration.
This module manages the storage, versioning, and retrieval of trained FL models.
"""

import os
import json
import logging
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import tensorflow as tf
import numpy as np

# JSON encoder tùy chỉnh để xử lý kiểu dữ liệu NumPy
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder hỗ trợ các kiểu dữ liệu NumPy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../fl_integration_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_repository")

class ModelRepository:
    """
    Model Repository class for managing trained FL models.
    Handles storage, versioning, and retrieval of models.
    """
    
    def __init__(self, 
                 base_dir: str = "./models",
                 metadata_file: str = "model_metadata.json"):
        """
        Initialize the model repository.
        
        Args:
            base_dir: Base directory for storing models
            metadata_file: Name of the metadata file
        """
        self.base_dir = base_dir
        self.metadata_file = metadata_file
        self.metadata_path = os.path.join(base_dir, metadata_file)
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize metadata if it doesn't exist
        if not os.path.exists(self.metadata_path):
            self._initialize_metadata()
        
        logger.info(f"Model repository initialized with base directory: {base_dir}")
    
    def _initialize_metadata(self):
        """Initialize the metadata file with an empty structure."""
        metadata = {
            "models": {},
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
        
        logger.info(f"Initialized metadata file at {self.metadata_path}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from file.
        
        Returns:
            Metadata dictionary
        """
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            return {"models": {}, "last_updated": datetime.now().isoformat()}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """
        Save metadata to file.
        
        Args:
            metadata: Metadata dictionary
        """
        try:
            # Update last_updated timestamp
            metadata["last_updated"] = datetime.now().isoformat()
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
            
            logger.info(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
    
    def save_model(self, 
                  model: tf.keras.Model, 
                  model_name: str,
                  version: str = None,
                  metadata: Dict[str, Any] = None) -> str:
        """
        Save a model to the repository.
        
        Args:
            model: Keras model to save
            model_name: Name of the model
            version: Version string (default: timestamp)
            metadata: Additional metadata for the model
            
        Returns:
            Path to the saved model
        """
        try:
            # Generate version if not provided
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create model directory
            model_dir = os.path.join(self.base_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Create version directory
            version_dir = os.path.join(model_dir, version)
            os.makedirs(version_dir, exist_ok=True)
            
            # Save model in SavedModel format
            model_path = os.path.join(version_dir, "model")
            model.save(model_path)
            
            # Save model in H5 format for compatibility
            h5_path = os.path.join(version_dir, f"{model_name}.h5")
            model.save(h5_path)
            
            # Update metadata
            repo_metadata = self._load_metadata()
            
            if model_name not in repo_metadata["models"]:
                repo_metadata["models"][model_name] = {
                    "versions": [],
                    "latest_version": None,
                    "created_at": datetime.now().isoformat()
                }
            
            # Create version metadata
            version_metadata = {
                "version": version,
                "path": version_dir,
                "h5_path": h5_path,
                "created_at": datetime.now().isoformat(),
                "metrics": metadata.get("metrics", {}) if metadata else {},
                "parameters": metadata.get("parameters", {}) if metadata else {}
            }
            
            # Add version to metadata
            repo_metadata["models"][model_name]["versions"].append(version_metadata)
            repo_metadata["models"][model_name]["latest_version"] = version
            
            # Save metadata
            self._save_metadata(repo_metadata)
            
            logger.info(f"Model {model_name} version {version} saved to {version_dir}")
            
            return h5_path
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {str(e)}")
            return ""
    
    def load_model(self, 
                  model_name: str,
                  version: str = None) -> tf.keras.Model:
        """
        Load a model from the repository.
        
        Args:
            model_name: Name of the model
            version: Version to load (default: latest)
            
        Returns:
            Loaded Keras model
        """
        try:
            # Get model metadata
            repo_metadata = self._load_metadata()
            
            if model_name not in repo_metadata["models"]:
                logger.error(f"Model {model_name} not found in repository")
                return None
            
            model_metadata = repo_metadata["models"][model_name]
            
            # Determine version to load
            if version is None:
                version = model_metadata["latest_version"]
            
            # Find version metadata
            version_metadata = None
            for v in model_metadata["versions"]:
                if v["version"] == version:
                    version_metadata = v
                    break
            
            if version_metadata is None:
                logger.error(f"Version {version} not found for model {model_name}")
                return None
            
            # Load model
            h5_path = version_metadata["h5_path"]
            if os.path.exists(h5_path):
                model = tf.keras.models.load_model(h5_path)
                logger.info(f"Model {model_name} version {version} loaded from {h5_path}")
                return model
            else:
                # Try loading from SavedModel format
                model_path = os.path.join(version_metadata["path"], "model")
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path)
                    logger.info(f"Model {model_name} version {version} loaded from {model_path}")
                    return model
                else:
                    logger.error(f"Model file not found for {model_name} version {version}")
                    return None
        except Exception as e:
            logger.error(f"Failed to load model {model_name} version {version}: {str(e)}")
            return None
    
    def get_model_metadata(self, 
                          model_name: str,
                          version: str = None) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_name: Name of the model
            version: Version to get metadata for (default: latest)
            
        Returns:
            Model metadata
        """
        try:
            # Get model metadata
            repo_metadata = self._load_metadata()
            
            if model_name not in repo_metadata["models"]:
                logger.error(f"Model {model_name} not found in repository")
                return {}
            
            model_metadata = repo_metadata["models"][model_name]
            
            # Return model metadata if no version specified
            if version is None:
                return model_metadata
            
            # Find version metadata
            for v in model_metadata["versions"]:
                if v["version"] == version:
                    return v
            
            logger.error(f"Version {version} not found for model {model_name}")
            return {}
        except Exception as e:
            logger.error(f"Failed to get metadata for model {model_name}: {str(e)}")
            return {}
    
    def list_models(self) -> List[str]:
        """
        List all models in the repository.
        
        Returns:
            List of model names
        """
        try:
            repo_metadata = self._load_metadata()
            return list(repo_metadata["models"].keys())
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
    
    def list_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version strings
        """
        try:
            repo_metadata = self._load_metadata()
            
            if model_name not in repo_metadata["models"]:
                logger.error(f"Model {model_name} not found in repository")
                return []
            
            model_metadata = repo_metadata["models"][model_name]
            
            return [v["version"] for v in model_metadata["versions"]]
        except Exception as e:
            logger.error(f"Failed to list versions for model {model_name}: {str(e)}")
            return []
    
    def get_latest_version(self, model_name: str) -> str:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version string
        """
        try:
            repo_metadata = self._load_metadata()
            
            if model_name not in repo_metadata["models"]:
                logger.error(f"Model {model_name} not found in repository")
                return None
            
            model_metadata = repo_metadata["models"][model_name]
            
            return model_metadata["latest_version"]
        except Exception as e:
            logger.error(f"Failed to get latest version for model {model_name}: {str(e)}")
            return None
    
    def delete_model(self, 
                    model_name: str,
                    version: str = None) -> bool:
        """
        Delete a model from the repository.
        
        Args:
            model_name: Name of the model
            version: Version to delete (default: all versions)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            repo_metadata = self._load_metadata()
            
            if model_name not in repo_metadata["models"]:
                logger.error(f"Model {model_name} not found in repository")
                return False
            
            model_metadata = repo_metadata["models"][model_name]
            
            if version is None:
                # Delete all versions
                model_dir = os.path.join(self.base_dir, model_name)
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                
                # Remove from metadata
                del repo_metadata["models"][model_name]
                
                # Save metadata
                self._save_metadata(repo_metadata)
                
                logger.info(f"Model {model_name} deleted from repository")
                
                return True
            else:
                # Delete specific version
                version_to_delete = None
                for i, v in enumerate(model_metadata["versions"]):
                    if v["version"] == version:
                        version_to_delete = v
                        break
                
                if version_to_delete is None:
                    logger.error(f"Version {version} not found for model {model_name}")
                    return False
                
                # Delete version directory
                version_dir = version_to_delete["path"]
                if os.path.exists(version_dir):
                    shutil.rmtree(version_dir)
                
                # Remove from metadata
                model_metadata["versions"].remove(version_to_delete)
                
                # Update latest version if needed
                if model_metadata["latest_version"] == version:
                    if model_metadata["versions"]:
                        # Set latest to most recent version
                        model_metadata["versions"].sort(key=lambda x: x["created_at"], reverse=True)
                        model_metadata["latest_version"] = model_metadata["versions"][0]["version"]
                    else:
                        model_metadata["latest_version"] = None
                
                # Save metadata
                self._save_metadata(repo_metadata)
                
                logger.info(f"Model {model_name} version {version} deleted from repository")
                
                return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {str(e)}")
            return False
    
    def compare_models(self, 
                      model_name: str,
                      version1: str,
                      version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a model.
        
        Args:
            model_name: Name of the model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results
        """
        try:
            # Get metadata for both versions
            metadata1 = self.get_model_metadata(model_name, version1)
            metadata2 = self.get_model_metadata(model_name, version2)
            
            if not metadata1 or not metadata2:
                logger.error(f"Failed to get metadata for comparison")
                return {}
            
            # Compare metrics
            metrics1 = metadata1.get("metrics", {})
            metrics2 = metadata2.get("metrics", {})
            
            metric_diffs = {}
            all_metrics = set(metrics1.keys()) | set(metrics2.keys())
            
            for metric in all_metrics:
                val1 = metrics1.get(metric)
                val2 = metrics2.get(metric)
                
                if val1 is not None and val2 is not None:
                    diff = val2 - val1
                    pct_change = (diff / val1) * 100 if val1 != 0 else float('inf')
                    metric_diffs[metric] = {
                        "version1": val1,
                        "version2": val2,
                        "absolute_diff": diff,
                        "percent_change": pct_change
                    }
                else:
                    metric_diffs[metric] = {
                        "version1": val1,
                        "version2": val2,
                        "absolute_diff": None,
                        "percent_change": None
                    }
            
            # Compare parameters
            params1 = metadata1.get("parameters", {})
            params2 = metadata2.get("parameters", {})
            
            param_diffs = {}
            all_params = set(params1.keys()) | set(params2.keys())
            
            for param in all_params:
                val1 = params1.get(param)
                val2 = params2.get(param)
                
                param_diffs[param] = {
                    "version1": val1,
                    "version2": val2,
                    "changed": val1 != val2
                }
            
            # Create comparison result
            comparison = {
                "model_name": model_name,
                "version1": version1,
                "version2": version2,
                "metric_differences": metric_diffs,
                "parameter_differences": param_diffs,
                "created_at1": metadata1.get("created_at"),
                "created_at2": metadata2.get("created_at")
            }
            
            return comparison
        except Exception as e:
            logger.error(f"Failed to compare models: {str(e)}")
            return {}
    
    def export_model(self, 
                    model_name: str,
                    version: str = None,
                    export_dir: str = "./exported_models") -> str:
        """
        Export a model to a directory.
        
        Args:
            model_name: Name of the model
            version: Version to export (default: latest)
            export_dir: Directory to export to
            
        Returns:
            Path to the exported model
        """
        try:
            # Create export directory
            os.makedirs(export_dir, exist_ok=True)
            
            # Get model metadata
            repo_metadata = self._load_metadata()
            
            if model_name not in repo_metadata["models"]:
                logger.error(f"Model {model_name} not found in repository")
                return ""
            
            model_metadata = repo_metadata["models"][model_name]
            
            # Determine version to export
            if version is None:
                version = model_metadata["latest_version"]
            
            # Find version metadata
            version_metadata = None
            for v in model_metadata["versions"]:
                if v["version"] == version:
                    version_metadata = v
                    break
            
            if version_metadata is None:
                logger.error(f"Version {version} not found for model {model_name}")
                return ""
            
            # Create export path
            export_path = os.path.join(export_dir, f"{model_name}_{version}")
            os.makedirs(export_path, exist_ok=True)
            
            # Copy model files
            h5_path = version_metadata["h5_path"]
            if os.path.exists(h5_path):
                export_h5_path = os.path.join(export_path, f"{model_name}.h5")
                shutil.copy2(h5_path, export_h5_path)
                
                # Export metadata
                export_metadata = {
                    "model_name": model_name,
                    "version": version,
                    "created_at": version_metadata.get("created_at"),
                    "metrics": version_metadata.get("metrics", {}),
                    "parameters": version_metadata.get("parameters", {})
                }
                
                with open(os.path.join(export_path, "metadata.json"), 'w') as f:
                    json.dump(export_metadata, f, indent=2, cls=NumpyJSONEncoder)
                
                logger.info(f"Model {model_name} version {version} exported to {export_path}")
                
                return export_path
            else:
                logger.error(f"Model file not found for {model_name} version {version}")
                return ""
        except Exception as e:
            logger.error(f"Failed to export model {model_name}: {str(e)}")
            return ""
    
    def import_model(self, 
                    import_path: str,
                    new_model_name: str = None) -> bool:
        """
        Import a model from a directory.
        
        Args:
            import_path: Path to the imported model
            new_model_name: New name for the model (default: use original name)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if import path exists
            if not os.path.exists(import_path):
                logger.error(f"Import path not found: {import_path}")
                return False
            
            # Load metadata
            metadata_path = os.path.join(import_path, "metadata.json")
            if not os.path.exists(metadata_path):
                logger.error(f"Metadata file not found in import path")
                return False
            
            with open(metadata_path, 'r') as f:
                import_metadata = json.load(f)
            
            # Determine model name
            model_name = new_model_name or import_metadata.get("model_name")
            if not model_name:
                logger.error(f"Model name not found in metadata and not provided")
                return False
            
            # Load model
            h5_path = os.path.join(import_path, f"{import_metadata.get('model_name')}.h5")
            if not os.path.exists(h5_path):
                logger.error(f"Model file not found in import path")
                return False
            
            model = tf.keras.models.load_model(h5_path)
            
            # Save model to repository
            version = import_metadata.get("version") or datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.save_model(
                model=model,
                model_name=model_name,
                version=version,
                metadata={
                    "metrics": import_metadata.get("metrics", {}),
                    "parameters": import_metadata.get("parameters", {})
                }
            )
            
            logger.info(f"Model imported from {import_path} as {model_name} version {version}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to import model: {str(e)}")
            return False
