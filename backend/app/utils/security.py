"""Security utilities and helpers."""
import re
import os
from typing import Optional


def validate_filename(filename: Optional[str]) -> bool:
    """
    Validate filename for security - prevent path traversal attacks.
    
    Args:
        filename: The filename to validate
        
    Returns:
        bool: True if filename is safe, False otherwise
    """
    if not filename:
        return False
    
    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    
    # Check for invalid characters
    if re.search(r'[:*?"<>|]', filename):
        return False
    
    # Allow alphanumeric, Chinese characters, spaces, dots, hyphens, and underscores
    if not re.match(r'^[\w\-. \u4e00-\u9fa5]+$', filename):
        return False
    
    return True


def is_path_within_directory(file_path: str, directory: str) -> bool:
    """
    Check if a file path is within a specified directory.
    
    Args:
        file_path: The file path to check
        directory: The parent directory
        
    Returns:
        bool: True if file_path is within directory
    """
    real_path = os.path.realpath(file_path)
    real_directory = os.path.realpath(directory)
    return real_path.startswith(real_directory)
