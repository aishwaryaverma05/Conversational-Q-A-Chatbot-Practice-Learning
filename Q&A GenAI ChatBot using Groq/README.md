
# Q&A Chatbot with GenAI

## 1. Project Overview
A simple AI chatbot that answers user questions using an LLM. Users can select a model and control temperature and maximum output tokens through a Streamlit interface.

## 2. Tech Stack
- **Python** – Core programming language
- **Streamlit** – Web interface
- **Groq** – LLM API
- **LangChain** – Connects prompt, LLM and output parser
- **LangSmith** – Tracing and monitoring
- **python-dotenv** – Loads environment variables

## 3. Workflow

User Question → Prompt Template → Groq LLM → Output Parser → Response

## 4. Important Components

### `ChatPromptTemplate`
Creates a structured prompt containing:
- **System message:** Defines the assistant's behaviour.
- **User message:** Passes the question using `{question}`.

### `ChatGroq`
Connects to the selected Groq model using the API key, temperature and maximum tokens.

### `StrOutputParser`
Converts the model output into a plain string.

### LangChain Chain
```python
chain = prompt | llm | output_parser
response = chain.invoke({"question": question})
```

- `|` connects components in sequence.
- `invoke()` executes the chain with the input question.

### `generate_response()`
1. Initializes the LLM.
2. Creates the output parser.
3. Connects prompt, LLM and parser.
4. Executes the chain and returns the response.
5. Handles exceptions using `try-except`.

### Streamlit UI
- Takes the Groq API key and question.
- Provides model selection, temperature and max-token controls.
- Validates empty inputs.
- Displays a loading spinner and the generated response.

## 5. Important Parameters

| Parameter | Purpose |
|---|---|
| Model | Selects the LLM |
| Temperature | Controls response randomness |
| Max Tokens | Limits generated output length |
| API Key | Authenticates the Groq API request |

**Temperature:** Lower value = generally more predictable output; higher value = more variation.

## 6. LangSmith Configuration
LangSmith tracing is enabled using environment variables to track LangChain runs.

- `LANGCHAIN_API_KEY` – API key for LangSmith
- `LANGCHAIN_TRACING_V2` – Enables tracing
- `LANGCHAIN_PROJECT` – Sets the project name

## 7. Setup and Run

Install dependencies:

```bash
pip install streamlit langchain-core langchain-groq python-dotenv
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
```

Run the application:

```bash
streamlit run app.py
```

**Security:** Never upload API keys or secrets to GitHub.

## 8. Quick Interview Questions

**Q1. Why use LangChain?**  
To connect prompts, LLMs and output parsers into a single workflow.

**Q2. Why use Streamlit?**  
To build an interactive web interface using Python.

**Q3. What does `StrOutputParser` do?**  
It converts the model output into a plain string.

**Q4. What is LangSmith used for?**  
To trace and monitor LangChain executions.

**Q5. Is this a RAG application?**  
No. It directly sends the question to the LLM without document retrieval or a vector database.

**Q6. Does the chatbot remember previous questions?**  
No. Conversation memory is not implemented.

## 9. Limitations
- No conversation memory.
- No RAG or external knowledge retrieval.
- Error handling is basic.
- Model availability depends on the Groq provider.
