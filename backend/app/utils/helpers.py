"""General utility helpers."""


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Human readable size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


def get_param_size(model_name: str) -> str:
    """
    Extract parameter size from model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        str: Parameter size (e.g., "7B", "13B")
    """
    size_patterns = [
        ("70b", "70B"), ("70B", "70B"),
        ("34b", "34B"), ("34B", "34B"),
        ("13b", "13B"), ("13B", "13B"),
        ("8b", "8B"), ("8B", "8B"),
        ("7b", "7B"), ("7B", "7B"),
        ("3b", "3B"), ("3B", "3B"),
        ("1b", "1B"), ("1B", "1B"),
    ]
    
    for pattern, size in size_patterns:
        if pattern in model_name:
            return size
    
    return "Unknown"


def get_quantization(model_name: str) -> str:
    """
    Extract quantization info from model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        str: Quantization type
    """
    quant_patterns = [
        ("-q4_0", "Q4_0"),
        ("-q4_1", "Q4_1"),
        ("-q5_0", "Q5_0"),
        ("-q5_1", "Q5_1"),
        ("-q8_0", "Q8_0"),
        ("-f16", "F16"),
        ("-raw", "Raw"),
    ]
    
    for pattern, quant in quant_patterns:
        if pattern in model_name:
            return quant
    
    return "Default"


def requires_gpu(model_name: str) -> bool:
    """
    Check if model typically requires GPU based on size.
    
    Args:
        model_name: Name of the model
        
    Returns:
        bool: True if GPU is typically required
    """
    large_models = ["70b", "70B", "34b", "34B", "13b", "13B"]
    return any(param in model_name for param in large_models)
