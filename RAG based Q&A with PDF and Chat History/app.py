import os                          
import tempfile    #temporary file storage for uploaded PDFs
import streamlit as st            
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader  
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma   
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()  
from langchain_core.runnables.history import RunnableWithMessageHistory


os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational Q&A Chatbot with PDF and Chat History")
st.write("Upload a PDF file and ask questions about its content. The chatbot will provide answers based on the information in the PDF, while maintaining the context of the conversation.")

# groq api
api_key=st.text_input("Enter your Groq API Key", type="password", value=os.getenv("GROQ_API_KEY"))


# chk if api key is provided
if api_key:
    llm=ChatGroq(api_key=api_key, model="gpt-4o")

    session_id=st.text_input("Session ID",value="default_session")  


    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_file=st.file_uploader("Upload a PDF file", type=["pdf"],accept_multiple_files=True)


    if uploaded_file:
        documents=[]
        for file in uploaded_file:
            temppdf=f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(file.getvalue())
                file_name=file.name
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)


        # split and create embeddings
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever(search_kwargs={"k":3})


        context=(
            "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        )

        context_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", context),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,retriever,context_prompt
        )

        # answer ques prompt

        system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer=create_stuff_documents_chain(llm,qa_prompt)

        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer)



        def get_session_history(session:str)->BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session]=ChatMessageHistory()
            return st.session_state.store[session]


        conversational_rag_chain=RunnableWithMessageHistory(rag_chain,get_session_history,input_key="input",history_key="chat_history",output_key="answer")




        user_input=st.text_input("Ask a question about the PDF content:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke({"input":"user_input"}, config={
                "configurable":{"session":session_id}
            },)


            st.write(st.session_state.store)
            st.write("Assistant:", response["answer"])
            st.write("Chat History:",session_history.messages)


else:
    st.warning("Please enter your Groq API Key to use the chatbot.")