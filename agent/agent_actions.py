"""
Define custom actions/tools for the agent to use.
"""
from typing import List, Optional
from datetime import datetime, timedelta
import json
import re
from llama_index.core.tools import FunctionTool

def summarize_document(text: str) -> str:
    """
    Summarize a document or lengthy text into key points.
    
    Args:
        text (str): The text to summarize
    
    Returns:
        str: A concise summary of the text
    """
    # In a real implementation, this would use the LLM to generate a summary
    # For the starter kit, this is simplified
    return f"Summary of document ({len(text)} characters)"

def create_task_list(topic: str, description: str) -> List[str]:
    """
    Create a task list based on document content.
    
    Args:
        topic (str): The topic or project name
        description (str): Description of what needs to be done
    
    Returns:
        List[str]: List of tasks
    """
    # In a real implementation, this would parse the documents to extract tasks
    # For the starter kit, this returns a simple placeholder
    return [
        f"Task 1 for {topic}",
        f"Task 2 for {topic}",
        f"Task 3 for {topic}"
    ]

def schedule_event(title: str, description: str, date_str: Optional[str] = None) -> str:
    """
    Schedule an event based on document information.
    
    Args:
        title (str): Event title
        description (str): Event description
        date_str (str, optional): Date string. Defaults to tomorrow.
    
    Returns:
        str: Confirmation message
    """
    # Default to tomorrow if no date provided
    if not date_str:
        tomorrow = datetime.now() + timedelta(days=1)
        date_str = tomorrow.strftime("%Y-%m-%d")
    
    # In a real implementation, this would integrate with a calendar API
    return f"Scheduled: {title} on {date_str}"

def extract_contact_info(text: str) -> str:
    """
    Extract contact information from document text.
    
    Args:
        text (str): Text containing contact information
    
    Returns:
        str: Structured contact information
    """
    # This is a simplified version
    # In a real implementation, this would use regex or NER to extract contacts
    contact = {
        "name": "Example Person",
        "email": "example@email.com",
        "phone": "123-456-7890"
    }
    
    return json.dumps(contact, indent=2)

def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of a text segment.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        str: Sentiment analysis result
    """
    # Simplified for the starter kit
    # In a real implementation, this would use an NLP model
    return "Sentiment appears positive"

def extract_dates(text: str) -> str:
    """
    Extract dates mentioned in the text.
    
    Args:
        text (str): Text that may contain dates
        
    Returns:
        str: List of dates found in the text
    """
    # Simple regex for dates (MM/DD/YYYY or MM-DD-YYYY)
    date_pattern = r'\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])[\/\-](19|20)\d{2}\b'
    
    dates = re.findall(date_pattern, text)
    
    if not dates:
        return "No dates found in the text."
    
    formatted_dates = []
    for month, day, year in dates:
        try:
            date_obj = datetime(int(year), int(month), int(day))
            formatted_date = date_obj.strftime("%B %d, %Y")
            formatted_dates.append(formatted_date)
        except ValueError:
            # Skip invalid dates
            continue
    
    if not formatted_dates:
        return "No valid dates found in the text."
    
    return "Found dates: " + ", ".join(formatted_dates)

def find_in_documents(query: str, memory_index=None) -> str:
    """
    Search the document memory for specific information.
    
    Args:
        query (str): What to search for
        memory_index: The memory index to search (passed by the agent)
        
    Returns:
        str: The found information or a message if not found
    """
    if memory_index is None:
        return "No memory index available to search."
    
    # Create a query engine
    query_engine = memory_index.as_query_engine()
    
    # Execute the search
    try:
        response = query_engine.query(query)
        if response and len(str(response)) > 0:
            return f"Found in documents: {str(response)}"
        else:
            return f"Could not find information about '{query}' in the documents."
    except Exception as e:
        return f"Error searching documents: {str(e)}"

# Create LlamaIndex function tools
summarize_tool = FunctionTool.from_defaults(
    name="summarize_document",
    fn=summarize_document,
    description="Summarize a document or lengthy text into key points"
)

task_list_tool = FunctionTool.from_defaults(
    name="create_task_list",
    fn=create_task_list,
    description="Create a task list based on document content"
)

schedule_tool = FunctionTool.from_defaults(
    name="schedule_event",
    fn=schedule_event,
    description="Schedule an event based on document information"
)

contact_tool = FunctionTool.from_defaults(
    name="extract_contact_info",
    fn=extract_contact_info,
    description="Extract contact information from document text"
)

sentiment_tool = FunctionTool.from_defaults(
    name="analyze_sentiment",
    fn=analyze_sentiment,
    description="Analyze the sentiment of a text segment"
)

extract_dates_tool = FunctionTool.from_defaults(
    name="extract_dates",
    fn=extract_dates,
    description="Extract dates mentioned in the text"
)

search_tool = FunctionTool.from_defaults(
    name="find_in_documents",
    fn=find_in_documents,
    description="Search the document memory for specific information"
)