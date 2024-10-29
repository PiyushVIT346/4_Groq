import streamlit as st 
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

# Load Groq API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# Ensure the API key exists
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()

# Initialize session state variables
if "vectors" not in st.session_state:
    try:
        # Initialize embeddings (using Ollama)
        st.session_state.embeddings = OllamaEmbeddings()

        # Load documents from the specified URL
        st.session_state.loader = WebBaseLoader('https://docs.smith.langchain.com/')
        st.session_state.docs = st.session_state.loader.load()

        # Split documents into chunks for better processing
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Create FAISS vectors from the document embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    except Exception as e:
        st.error(f"Error initializing session state: {str(e)}")
        st.stop()

# App title
st.title("ChatGroq Demo")

# Initialize the Groq LLM (replace with appropriate model if necessary)
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")
except Exception as e:
    st.error(f"Error initializing ChatGroq model: {str(e)}")
    st.stop()

# Define a prompt template for the chatbot
prompt_template = ChatMessagePromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    
    Questions: {input}
    """
)

# Create a chain for handling document-based responses
try:
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
except Exception as e:
    st.error(f"Error creating retrieval chain: {str(e)}")
    st.stop()

# Input prompt from the user
user_prompt = st.text_input("Input your prompt here")

if user_prompt:
    start_time = time.process_time()

    # Invoke the chain to get a response
    try:
        response = retriever_chain.invoke({"input": user_prompt})
        st.write(f"Response: {response['answer']}")
        st.write(f"Response time: {time.process_time() - start_time:.2f} seconds")

    except Exception as e:
        st.error(f"Error occurred during retrieval: {str(e)}")
