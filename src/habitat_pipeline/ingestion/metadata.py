"""Metadata parsing and management."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class MetadataParser:
    """
    Parser for experimental metadata.
    
    Supports JSON and YAML formats for metadata files.
    """
    
    def __init__(self, metadata_path: Optional[str] = None):
        """
        Initialize metadata parser.
        
        Parameters
        ----------
        metadata_path : str, optional
            Path to metadata file
        """
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.metadata = {}
        
        if self.metadata_path and self.metadata_path.exists():
            self.load()
    
    def load(self) -> Dict[str, Any]:
        """
        Load metadata from file.
        
        Returns
        -------
        dict
            Metadata dictionary
        """
        if not self.metadata_path or not self.metadata_path.exists():
            logger.warning("Metadata path not found")
            return {}
        
        suffix = self.metadata_path.suffix.lower()
        
        try:
            if suffix == '.json':
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported metadata format: {suffix}")
                return {}
            
            logger.info(f"Loaded metadata from {self.metadata_path}")
            return self.metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    def save(self, output_path: str, format: str = 'yaml') -> None:
        """
        Save metadata to file.
        
        Parameters
        ----------
        output_path : str
            Path to output file
        format : str
            Output format ('json' or 'yaml')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
            elif format == 'yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(self.metadata, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved metadata to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set metadata value by key."""
        self.metadata[key] = value
    
    def update(self, metadata: Dict[str, Any]) -> None:
        """Update metadata with new values."""
        self.metadata.update(metadata)
    
    def validate(self, schema: Dict[str, Any]) -> bool:
        """
        Validate metadata against a schema.
        
        Parameters
        ----------
        schema : dict
            Schema dictionary with required fields
            
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        for key, required_type in schema.items():
            if key not in self.metadata:
                logger.warning(f"Missing required field: {key}")
                return False
            
            if not isinstance(self.metadata[key], required_type):
                logger.warning(f"Invalid type for {key}: expected {required_type}")
                return False
        
        return True
