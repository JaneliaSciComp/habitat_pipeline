"""Utility functions and classes for the habitat_pipeline package."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the pipeline.
    
    Parameters
    ----------
    level : str, optional
        Logging level (default: "INFO")
    log_file : str, optional
        Path to log file (default: None, logs to console only)
    format_string : str, optional
        Custom format string for log messages
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger("habitat_pipeline")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save YAML configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Parameters
    ----------
    path : str
        Directory path
        
    Returns
    -------
    Path
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(file_path: str) -> Path:
    """
    Validate that a file exists.
    
    Parameters
    ----------
    file_path : str
        Path to file
        
    Returns
    -------
    Path
        Path object for the file
        
    Raises
    ------
    FileNotFoundError
        If file does not exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path
