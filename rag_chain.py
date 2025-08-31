# rag_chain.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI as Gemini
from dotenv import load_dotenv
import os

load_dotenv()  # load GEMINI_API_KEY from .env

class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()

    def get_llm(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No valid API key found! Please set one in .env file.")
        
        # Gemini LLM with slightly higher temperature for friendly responses
        return Gemini(
            api_key=api_key,
            model="gemini-2.0-flash",
            temperature=0.7,
            streaming=True
        )

    def create_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Friendly prompt including both {context} and {question}
        template = """
        You are a friendly AI series recommender.
        Always greet the user warmly and answer in a casual, helpful way.
        Use the following context to help you answer:

        {context}

        User Question: {question}

        Friendly Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}  # use the custom prompt
        )

        return qa_chain