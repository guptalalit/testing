import os
import json
from typing import List, Dict
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "mongodb")  # Default to MongoDB
FAISS_STORAGE_PATH = os.getenv("FAISS_STORAGE_PATH", "./faiss_store")

class RAG:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def retrieve_documents(self, query: str, application_name: str, k: int = 3) -> List[Dict]:
        """
        Retrieve documents from either MongoDB or FAISS based on configuration
        """
        print("DATABASE_TYPE: ", DATABASE_TYPE)
        if DATABASE_TYPE == "mongodb":
            return self._retrieve_from_mongodb(query, application_name, k)
        elif DATABASE_TYPE == "local":
            return self._retrieve_from_faiss(query, application_name, k)
        else:
            raise ValueError(f"Unsupported database type: {DATABASE_TYPE}")

    def _retrieve_from_mongodb(self, query: str, application_name: str, k: int) -> List[Dict]:
        """Original MongoDB implementation"""
        MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
        COLLECTION_NAME = f"{application_name}_docs"
        ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
        collection = client[application_name][COLLECTION_NAME]
        
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=self.embedding_model,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
        )
        
        results = vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def _retrieve_from_faiss(self, query: str, application_name: str, k: int) -> List[Dict]:
        """FAISS implementation"""
        print("faiss results: ", FAISS_STORAGE_PATH, application_name)
        store_path = os.path.join(FAISS_STORAGE_PATH, application_name, "docs")
        
        if not os.path.exists(store_path):
            raise FileNotFoundError(f"No FAISS index found at {store_path}")
            
        vector_store = FAISS.load_local(
            store_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        results = vector_store.similarity_search(query, k=k)
        print("results: ", results)
        return [doc.page_content for doc in results]

    def rag(self, query: str, sessionid: str, application_name: str) -> str:
        """
        Generate answer using the configured storage backend
        """
        try:
            context_docs = self.retrieve_documents(query, application_name, k=9)
            # Add your existing RAG pipeline logic here
            return context_docs
        except Exception as e:
            return f"Error in RAG pipeline: {e}"