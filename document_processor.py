"""
Handles loading and processing documents for the agent.
"""
import os
from typing import List, Dict, Any
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR
import tqdm

def load_documents() -> List[Document]:
    """
    Load all documents from the documents directory.
    
    Returns:
        List[Document]: List of loaded documents
    """
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"Created documents directory: {DOCUMENTS_DIR}")
        print("Please add your documents to this folder and run the script again.")
        return []
    
    docs_count = len([f for f in os.listdir(DOCUMENTS_DIR) if os.path.isfile(os.path.join(DOCUMENTS_DIR, f))])
    if docs_count == 0:
        print("No documents found in the documents directory.")
        print(f"Please add documents to the {DOCUMENTS_DIR} folder.")
        return []
    
    print(f"Loading {docs_count} documents from {DOCUMENTS_DIR}...")
    try:
        reader = SimpleDirectoryReader(input_dir=DOCUMENTS_DIR)
        documents = reader.load_data()
        print(f"Successfully loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def process_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Process documents into nodes for indexing.
    
    Args:
        documents (List[Document]): List of documents to process
        
    Returns:
        List[Dict[str, Any]]: List of processed nodes
    """
    if not documents:
        return []
    
    print("Processing documents into chunks...")
    parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks from {len(documents)} documents.")
    
    return nodes