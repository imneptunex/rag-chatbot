import streamlit as st
from document_processor import DocumentProcessor
from embedding_indexer import EmbeddingIndexer
from rag_chain import RAGChain
from chatbot import Chatbot

@st.cache_resource
def initialize_chatbot():
    # Gets the dataset 
    processor = DocumentProcessor("data/movies.txt")
    texts = processor.load_and_split()
    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)
    rag_chain = RAGChain(vectorstore)
    return Chatbot(rag_chain.create_chain())

st.title("🎬 Movie Recommender Chatbot")

chatbot = initialize_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Input for user request
if prompt := st.chat_input("What kind of movie do you feel like watching today?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chatbot.get_response(prompt)
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})