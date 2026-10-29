import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()


# langsmith tracking->used for tracking your project ki har cheze 
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A ChatBOT with GenAI"


# prompt template 
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant that answers questions."),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,llm_model,tempreature,max_tokens):
    try:
        #  initialise the groq llm with the user input
        llm=ChatGroq(api_key=api_key, model_name=llm_model,temperature=tempreature,max_tokens=max_tokens)

        output_parser=StrOutputParser()

        # create the langchain chain( prompt->llm->parser)
        chain=prompt|llm|output_parser

        # invoke the chain with the user input
        response= chain.invoke(
            {"question":question}
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"


# Streamlit part 
st.title("Q&A ChatBOT with GenAI")

# sidebar
st.sidebar.title("Configuration")
api_key=st.sidebar.text_input("Enter your Groq API key:",type="password",value=os.getenv('GROQ_API_KEY'))


# dropdown for llm model selection
llm_model=st.sidebar.selectbox(
    "Select LLm Model:",
    [
        "openai/gpt-oss-120b",          
        "openai/gpt-oss-20b",           
        "qwen/qwen3.8-27b",             
        "openai/gpt-oss-safeguard-20b"  
    ]
)

tempreature=st.sidebar.slider("Select Temperature:",0.0,1.0,0.7,0.1)

max_tokens=st.sidebar.slider("Max Tokens:", 100, 4000, 1000)

# user input
user_question=st.text_input("Ask you question here:")


if st.button("Generate Response"):
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar!")
    elif not user_question:
        st.warning("Please type a question first!")

    else:
        with st.spinner("Thinking..."):
            response=generate_response(user_question,api_key,llm_model,tempreature,max_tokens)
            st.write(response)