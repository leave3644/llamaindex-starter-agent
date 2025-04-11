# Workshop Guide: Building a Document-Powered AI Agent with LlamaIndex

## 🎯 Workshop Objective

In this hands-on workshop, you'll deploy an AI agent powered entirely by LlamaIndex that can:
1. Ingest your documents
2. Build a "smart memory" from those documents
3. Answer questions about your documents
4. Take actions based on document content

---

## 🚀 How to Run It (Setup Guide)

### Part 1: Setup (10 minutes)

#### Prerequisites Check
- [ ] Python 3.9+ installed
- [ ] Git installed
- [ ] OpenAI API key ([Get one here](https://platform.openai.com/api-keys) if you don't have it)
- [ ] Text editor or IDE (VS Code recommended)


#### Installation

1. **Clone the repository**
   
   Open your terminal/command prompt and run:
   
   ```bash
   git clone https://github.com/your-username/llamaindex-memory-agent.git
   cd llamaindex-memory-agent
   ```

2. **Create a virtual environment**
   
   For Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   For macOS/Linux:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
   
   You should see `(venv)` at the beginning of your terminal prompt, indicating the virtual environment is active.

3. **Install dependencies**
   
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install LlamaIndex and other required packages.

4. **Set up your API keys**
   
   Create a `.env` file in the project root directory:
   
   ```bash
   # Windows (Command Prompt)
   echo OPENAI_API_KEY=your-api-key-here > .env
   
   # macOS/Linux
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```
   
   Replace `your-api-key-here` with your actual OpenAI API key.

### Part 2: Add Your Documents (5 minutes)

1. **Prepare sample documents**
   
   For this workshop, you'll need some documents to work with. You can use:
   - Your own documents (PDFs, Word docs, text files)
   - Sample documents provided in the workshop
   
   Place your documents in the `documents` folder.

2. **Supported document types**
   
   The agent works with:
   - PDF files (.pdf)
   - Word documents (.docx)
   - Text files (.txt)
   - Markdown files (.md)

### Part 3: Run Your Agent (5 minutes)

1. **Start the agent**
   
   ```bash
   python run_agent.py
   ```
   
   You should see output like:
   ```
   Welcome to LlamaIndex Memory Agent!
   ================================================
   
   Loading documents from documents/...
   Successfully loaded 3 documents.
   Processing documents into chunks...
   Created 15 chunks from 3 documents.
   Building memory index...
   Memory index built and saved to memory_index/.
   Initializing agent...
   Agent initialized successfully.
   
   Your document-powered AI agent is ready!
   Type 'exit' to quit the program.
   
   >
   ```

2. **Try some basic queries**
   
   Test your agent with questions like:
   ```
   > What documents do you have access to?
   > What are the main topics in my documents?
   > When was the last meeting mentioned in my notes?
   ```

### Part 4: Understanding How It Works (20 minutes)

#### Document Processing Explained

Let's understand how the document processing works:

1. **Opening `document_processor.py`**
   
   Open this file in your editor and notice:
   - `load_documents()` function uses LlamaIndex's `SimpleDirectoryReader`
   - `process_documents()` function chunks documents into smaller pieces

2. **Memory Creation**
   
   Open `memory_handler.py` and see:
   - `build_memory_index()` creates a vector store from documents
   - `query_memory()` searches documents for relevant information

3. **Agent Tools & Actions**
   
   Examine `agent_actions.py`:
   - Each function is a "tool" the agent can use
   - Tools are wrapped using LlamaIndex's `FunctionTool`
   - Sample tools include summarizing, task creation, and scheduling

### Part 5: Customizing Your Agent (30 minutes)

#### Adding a Custom Tool

1. **Add a new tool**
   
   Open `agent_actions.py` and add a new function:
   
   ```python
   def count_mentions(query: str, text: str) -> str:
       """
       Count how many times a term is mentioned in text.
       
       Args:
           query (str): The term to search for
           text (str): Text to search within
           
       Returns:
           str: Number of mentions found
       """
       count = text.lower().count(query.lower())
       return f"'{query}' is mentioned {count} times in the text."
   
   # Create a LlamaIndex function tool
   count_tool = FunctionTool.from_defaults(
       name="count_mentions",
       fn=count_mentions,
       description="Count how many times a term is mentioned in text"
   )
   ```

2. **Register your new tool**
   
   Open `run_agent.py` and update the tools list in the `initialize_agent` function:
   
   ```python
   # Get the tools
   tools = [
       agent_actions.summarize_tool,
       agent_actions.task_list_tool,
       agent_actions.schedule_tool,
       agent_actions.contact_tool,
       agent_actions.sentiment_tool,
       agent_actions.count_tool  # Add your new tool here
   ]
   ```

3. **Test your new tool**
   
   Restart the agent and try your new tool:
   ```
   > How many times is "meeting" mentioned in my documents?
   ```

#### Changing the Agent's Memory

1. **Modify memory settings**
   
   Open `config.py` and update settings:
   
   ```python
   # Memory Settings
   SIMILARITY_TOP_K = 5  # Increase from 3 to 5
   ```

2. **Clear the existing memory**
   
   Delete the `memory_index` folder to force rebuilding the index:
   ```bash
   # Windows
   rmdir /s /q memory_index
   
   # macOS/Linux
   rm -rf memory_index
   ```

3. **Restart the agent**
   
   ```bash
   python run_agent.py
   ```

### Part 6: Advanced Features (20 minutes)

#### Creating a Document Summary

1. **Add a batch processing tool**
   
   In `agent_actions.py`, add:
   
   ```python
   def create_document_summary(memory_index) -> str:
       """
       Create an overall summary of all documents.
       
       Args:
           memory_index: The memory index containing all documents
           
       Returns:
           str: A comprehensive summary
       """
       # Use the query engine to generate a summary
       query_engine = memory_index.as_query_engine()
       response = query_engine.query(
           "Create a comprehensive summary of all documents in 3-5 paragraphs. "
           "Focus on key information, main topics, and important dates."
       )
       return str(response)
   
   # Create a LlamaIndex function tool
   summary_tool = FunctionTool.from_defaults(
       name="create_document_summary",
       fn=create_document_summary,
       description="Create an overall summary of all documents"
   )
   ```

2. **Register and test the tool**
   
   Add to your tools list and restart the agent. Then try:
   ```
   > Create a summary of all my documents.
   ```

## 🎓 Workshop Wrap-Up

Congratulations! You've successfully:
- Set up a document-powered AI agent using LlamaIndex
- Learned how documents are processed and indexed
- Created custom agent tools
- Modified memory settings
- Built advanced document processing features

### Next Steps

1. Try adding more document types to the `documents` folder
2. Create more sophisticated tools
3. Connect to external APIs (like Google Calendar for actual scheduling)
4. Expand the agent's memory capabilities

## 📚 Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-
## Acknowledgements

- LlamaIndex for the core indexing technology
- OpenAI for the language models
- LangChain for agent frameworks

## 📜 License
MIT