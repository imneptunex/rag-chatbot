# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Chroma. This application allows users to interact with a chatbot that retrieves relevant information from documents to provide accurate and context-aware responses.

## Features

- **Streamlit Interface**: User-friendly web interface for interacting with the chatbot.
- **Document Processing**: Supports processing of Markdown files to build a knowledge base.
- **Embeddings**: Utilizes sentence-transformers to create embeddings for document chunks.
- **Vector Store**: Stores embeddings in Chroma for efficient retrieval.

## Installation

### Clone the Repository

git clone https://github.com/imneptunex/rag-chatbot.git
cd rag-chatbot

**Set Up Virtual Environment**
python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate

**Install Dependencies**
pip install -r requirements.txt

**Run the Streamlit App**
streamlit run app.py
**This will start the Streamlit server and provide a local URL to access the chatbot interface.**


Contributing

Contributions are welcome! Please fork the repository, create a new branch, make your changes, and submit a pull request.

