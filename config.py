"""
Configuration settings for the LlamaIndex Memory Agent.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")

# Model Configuration
MODEL_NAME = "gpt-3.5-turbo"  # Options: "gpt-3.5-turbo", "gpt-4"
MODEL_TEMPERATURE = 0.2  # Lower = more deterministic, Higher = more creative

# Document Processing
CHUNK_SIZE = 1000  # Size of text chunks for processing
CHUNK_OVERLAP = 200  # Overlap between chunks for context preservation
DOCUMENTS_DIR = "documents"  # Directory to store documents

# Memory Settings
MEMORY_INDEX_NAME = "memory_index"  # Name of the saved index file
SIMILARITY_TOP_K = 3  # Number of top results to retrieve