# Conversational PDF Q&A Chatbot

## 1. Project Overview

A conversational chatbot that allows users to upload PDF documents and ask questions about their content. It uses **RAG (Retrieval-Augmented Generation)** to retrieve relevant information and an LLM to generate concise answers while maintaining chat history.

## 2. Tech Stack

- **Python** — Core programming
- **Streamlit** — Web interface
- **LangChain** — Connects the LLM, retriever, prompts and chat history
- **Groq API** — LLM inference
- **HuggingFace Embeddings** — Converts text into vectors
- **ChromaDB** — Stores embeddings and retrieves similar text chunks
- **PyPDFLoader** — Loads PDF content
- **RecursiveCharacterTextSplitter** — Splits documents into chunks
- **ChatMessageHistory** — Stores conversation messages

## 3. How It Works

```text
Upload PDF
    ↓
PyPDFLoader extracts text
    ↓
Split text into chunks
    ↓
Generate embeddings
    ↓
Store vectors in ChromaDB
    ↓
User asks a question
    ↓
Retrieve top 3 relevant chunks
    ↓
LLM generates an answer
    ↓
Display answer + maintain chat history
```

## 4. Important Components

| Component | Purpose |
|---|---|
| `PyPDFLoader` | Loads PDF pages as documents |
| `RecursiveCharacterTextSplitter` | Splits text into manageable chunks |
| `chunk_size=1000` | Target chunk size of 1,000 characters |
| `chunk_overlap=200` | Preserves 200 overlapping characters between chunks |
| `HuggingFaceEmbeddings` | Converts text into numerical vectors |
| `all-MiniLM-L6-v2` | Embedding model used in the project |
| `Chroma.from_documents()` | Stores document chunks and their embeddings |
| `as_retriever(k=3)` | Retrieves the top 3 relevant chunks |
| `ChatPromptTemplate` | Structures instructions and user input |
| `MessagesPlaceholder` | Inserts conversation history into the prompt |
| `create_history_aware_retriever()` | Uses chat history to understand follow-up questions |
| `create_stuff_documents_chain()` | Combines retrieved documents with the answer prompt and LLM |
| `create_retrieval_chain()` | Connects retrieval and answer generation |
| `RunnableWithMessageHistory` | Adds session-based conversation history to the chain |
| `st.session_state` | Maintains session data in Streamlit |

## 5. What Is RAG?

**RAG = Retrieval + Augmented Generation**

1. **Retrieval:** Finds relevant chunks from the uploaded PDFs.
2. **Augmentation:** Adds retrieved information to the LLM's prompt.
3. **Generation:** The LLM generates an answer using the provided context.

**Why RAG?** It allows the chatbot to answer questions using user-provided documents instead of relying only on the LLM's pretrained knowledge.

## 6. How Chat History Works

- A `session_id` identifies a conversation.
- `st.session_state.store` stores histories for different sessions.
- `ChatMessageHistory` maintains messages for each session.
- `MessagesPlaceholder("chat_history")` passes previous messages to the prompts.
- The history-aware retriever rewrites follow-up questions into standalone questions when necessary.

**Example:**

User: What is machine learning?

User: What are its types?

The chatbot can understand that "its" refers to machine learning using the conversation history.

## 7. Important Interview Questions

**Q1. Why use embeddings?**

Embeddings convert text into numerical vectors so that semantically similar text can be retrieved.

**Q2. Why use ChromaDB?**

It stores document embeddings and supports similarity search to find relevant chunks.

**Q3. Why chunk documents?**

Smaller chunks improve retrieval precision and help fit relevant information within the LLM's context window.

**Q4. What is a history-aware retriever?**

It uses previous conversation messages to reformulate a follow-up question before retrieving relevant documents.

**Q5. Difference between a retriever and an LLM?**

A retriever finds relevant information from documents; an LLM generates a natural-language answer.

**Q6. Why use `RunnableWithMessageHistory`?**

It connects a chain with session-based chat history so the chatbot can handle multi-turn conversations.

**Q7. Does this project train an LLM?**

No. It uses an existing LLM and an embedding model. It retrieves relevant document information at query time.

**Q8. What is the role of the system prompt?**

It instructs the LLM to answer using retrieved context, admit when it does not know, and keep answers within three sentences.

## 8. Setup

Install the required packages:

```bash
pip install streamlit langchain-core langchain-groq langchain-community langchain-text-splitters langchain-chroma langchain-classic sentence-transformers pypdf python-dotenv
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

Run the application:

```bash
streamlit run app.py
```

## 9. Limitations

- Answer quality depends on PDF text extraction and retrieval quality.
- The current implementation retrieves only the top 3 chunks.
- Chat history is maintained in application memory and is not persistent across application restarts.
- Answers are not guaranteed to be correct; retrieved context and generated responses should be evaluated.
- The current implementation does not explicitly display source-page citations.

## 10. One-Minute Revision

Remember this flow:

**PDF → Load → Chunk → Embed → Store → Retrieve → Generate → Remember conversation**

- **PyPDFLoader:** Read PDFs
- **Text Splitter:** Create chunks
- **Embeddings:** Convert text to vectors
- **ChromaDB:** Store and search vectors
- **History-aware Retriever:** Understand follow-up questions
- **LLM:** Generate answers from retrieved context
- **RunnableWithMessageHistory:** Maintain conversational context

**Core concept:** RAG provides relevant external information to an LLM at query time, while chat history helps it understand the ongoing conversation.
