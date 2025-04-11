"""
Manages the agent's memory using LlamaIndex.
"""
import os
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, ServiceContext, load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from config import MODEL_NAME, MODEL_TEMPERATURE, MEMORY_INDEX_NAME, SIMILARITY_TOP_K, OPENAI_API_KEY

def create_service_context():
    """
    Create the service context for LlamaIndex with the configured LLM.
    
    Returns:
        ServiceContext: The configured service context
    """
    llm = OpenAI(
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
        api_key=OPENAI_API_KEY
    )
    
    service_context = ServiceContext.from_defaults(llm=llm)
    return service_context

def build_memory_index(nodes: List[Dict[str, Any]]):
    """
    Build and save a vector index from document nodes.
    
    Args:
        nodes (List[Dict[str, Any]]): List of document nodes
    
    Returns:
        VectorStoreIndex: The built index
    """
    service_context = create_service_context()
    
    print("Building memory index...")
    index = VectorStoreIndex(
        nodes=nodes,
        service_context=service_context
    )
    
    # Save the index to disk
    if not os.path.exists(MEMORY_INDEX_NAME):
        os.makedirs(MEMORY_INDEX_NAME)
    
    index.storage_context.persist(persist_dir=MEMORY_INDEX_NAME)
    print(f"Memory index built and saved to {MEMORY_INDEX_NAME}.")
    
    return index

def load_memory_index():
    """
    Load an existing index from disk, or return None if it doesn't exist.
    
    Returns:
        VectorStoreIndex or None: The loaded index, or None if not found
    """
    if not os.path.exists(MEMORY_INDEX_NAME):
        print(f"No existing memory index found at {MEMORY_INDEX_NAME}.")
        return None
    
    print(f"Loading existing memory index from {MEMORY_INDEX_NAME}...")
    try:
        service_context = create_service_context()
        storage_context = StorageContext.from_defaults(persist_dir=MEMORY_INDEX_NAME)
        index = load_index_from_storage(
            storage_context=storage_context,
            service_context=service_context
        )
        print("Memory index loaded successfully.")
        return index
    except Exception as e:
        print(f"Error loading memory index: {e}")
        return None

def query_memory(index, query_text: str) -> str:
    """
    Query the memory index with a natural language query.
    
    Args:
        index: The vector store index
        query_text (str): The query text
        
    Returns:
        str: The query response
    """
    if index is None:
        return "No memory index available. Please add documents first."
    
    # Create a query engine with similarity search
    query_engine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
    )
    
    # Execute the query
    try:
        response = query_engine.query(query_text)
        return str(response)
    except Exception as e:
        return f"Error querying memory: {e}"
