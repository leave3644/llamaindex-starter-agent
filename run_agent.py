"""
Main script to run the LlamaIndex Memory Agent.
"""
import os
import sys
from llama_index.agent.openai import OpenAIAgent  # Updated import
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent  # Try the core package


from document_processor import load_documents, process_documents
from memory_handler import build_memory_index, load_memory_index, query_memory, create_service_context
from config import MODEL_NAME, MODEL_TEMPERATURE, OPENAI_API_KEY, DOCUMENTS_DIR
from agent import agent_actions

def initialize_agent(memory_index):
    """
    Initialize the agent with memory and tools.
    
    Args:
        memory_index: The memory index to use
        
    Returns:
        Agent: The initialized agent
    """
    print("Initializing agent...")
    
    # Set up the LLM
    llm = OpenAI(
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
        api_key=OPENAI_API_KEY
    )
    
    # Create a query engine from the memory index
    query_engine = memory_index.as_query_engine()
    
    # Get the tools
    tools = [
        agent_actions.summarize_tool,
        agent_actions.task_list_tool,
        agent_actions.schedule_tool,
        agent_actions.contact_tool,
        agent_actions.sentiment_tool
    ]
    
    # Try the updated agent initialization
    try:
        # Try ReActAgent first
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=True
        )
    except (ImportError, AttributeError):
        # Fallback to SimpleAgent if available
        try:
            from llama_index.core.agent import SimpleAgent
            agent = SimpleAgent.from_tools(
                tools=tools,
                llm=llm,
                verbose=True
            )
        except (ImportError, AttributeError):
            # Final fallback to OpenAIAgent
            try:
                from llama_index.core.agent.openai import OpenAIAgent
                agent = OpenAIAgent.from_tools(
                    tools=tools,
                    llm=llm,
                    verbose=True
                )
            except (ImportError, AttributeError):
                # If all else fails, raise a clear error
                raise ImportError(
                    "Could not initialize an agent with the current LlamaIndex version. "
                    "Please check the LlamaIndex documentation for the current agent implementation."
                )
    
    print("Agent initialized successfully.")
    return agent

def main():
    """Main function to run the agent."""
    print("\n" + "="*50)
    print("Welcome to LlamaIndex Memory Agent!")
    print("="*50 + "\n")
    
    # Check if documents directory exists and has files
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"Created documents directory: {DOCUMENTS_DIR}")
        print(f"Please add your documents to the {DOCUMENTS_DIR} folder and run the script again.")
        sys.exit(0)
    
    # Load an existing index or create a new one
    memory_index = load_memory_index()
    
    if memory_index is None:
        # Load and process documents
        documents = load_documents()
        if not documents:
            print("No documents to process. Exiting.")
            sys.exit(0)
        
        # Process documents and build the index
        nodes = process_documents(documents)
        memory_index = build_memory_index(nodes)
    
    # Initialize the agent
    agent = initialize_agent(memory_index)
    
    # Main interaction loop
    print("\nYour document-powered AI agent is ready!")
    print("Type 'exit' to quit the program.\n")
    
    while True:
        try:
            user_input = input("> ")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Thank you for using LlamaIndex Memory Agent. Goodbye!")
                break
            
            # Process the query with the agent
            response = agent.chat(user_input)
            print(response.response)
            
        except KeyboardInterrupt:
            print("\nProgram interrupted. Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
