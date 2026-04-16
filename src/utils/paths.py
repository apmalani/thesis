"""
Path utility functions for robust path handling.
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory by looking for common markers.
    
    Looks for:
    - .git directory
    - requirements.txt
    - README.md
    
    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    
    # Navigate up from src/utils/paths.py to find project root
    for parent in current.parents:
        # Check for project root markers
        if (parent / '.git').exists() or \
           (parent / 'requirements.txt').exists() or \
           (parent / 'README.md').exists():
            return parent
    
    # Fallback: assume we're in src/utils, go up two levels
    return current.parent.parent


def get_models_dir(state: Optional[str] = None) -> Path:
    """
    Get path to models directory, creating it if it doesn't exist.
    
    Args:
        state: Optional state abbreviation to append
        
    Returns:
        Path to models directory (or state-specific subdirectory)
    """
    root = get_project_root()
    models_dir = root / 'models'
    
    if state:
        models_dir = models_dir / state
    
    # Create directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)
    
    return models_dir


def get_data_dir(state: Optional[str] = None, subdir: str = 'processed') -> Path:
    """
    Get path to data directory, creating it if it doesn't exist.
    
    Args:
        state: Optional state abbreviation to append
        subdir: Subdirectory ('processed' or 'raw')
        
    Returns:
        Path to data directory
    """
    root = get_project_root()
    data_dir = root / 'data' / subdir
    
    if state:
        data_dir = data_dir / state
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return data_dir


def get_logs_dir(state: Optional[str] = None) -> Path:
    """
    Get path to logs directory, creating it if it doesn't exist.
    
    Args:
        state: Optional state abbreviation to append
        
    Returns:
        Path to logs directory (or state-specific subdirectory)
    """
    root = get_project_root()
    logs_dir = root / 'logs'
    
    if state:
        logs_dir = logs_dir / state
    
    # Create directory if it doesn't exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    return logs_dir


def get_outputs_dir(state: Optional[str] = None, subdir: str = 'best_maps') -> Path:
    """
    Get path to outputs directory, creating it if it doesn't exist.
    
    Args:
        state: Optional state abbreviation to append
        subdir: Subdirectory (default: 'best_maps')
        
    Returns:
        Path to outputs directory (or state-specific subdirectory)
    """
    root = get_project_root()
    outputs_dir = root / 'outputs' / subdir
    
    if state:
        outputs_dir = outputs_dir / state
    
    # Create directory if it doesn't exist
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    return outputs_dir

