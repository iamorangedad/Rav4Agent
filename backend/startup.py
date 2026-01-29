#!/usr/bin/env python3
"""
Startup script to handle API compatibility for doc-chat
"""
import os
import sys

# Add compatibility patch for OllamaEmbedding
try:
    from llama_index.embeddings.ollama import OllamaEmbedding
    original_init = OllamaEmbedding.__init__
    
    def patched_init(self, **kwargs):
        # Handle both 'model' and 'model_name' parameters
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        return original_init(self, **kwargs)
    
    OllamaEmbedding.__init__ = patched_init
    print("Applied OllamaEmbedding compatibility patch")
except ImportError as e:
    print(f"Could not apply patch: {e}")

# Now import and run the main application
exec(open('/app/main.py').read())