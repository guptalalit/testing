import os
import json
from typing import List, Dict
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import google.generativeai as genai
from memory.memory import Memory
from dotenv import load_dotenv
load_dotenv()




# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class RAG:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        
       

    def retrieve_documents(self, query: str,application_name, k: int = 3) -> List[Dict]:
        """
        Retrieve the top-k most relevant documents for the given query using MongoDB Vector Search.

        Args:
            query (str): The query to search for.
            k (int): The number of documents to retrieve.

        Returns:
            list: A list of dictionaries containing the retrieved documents and their metadata.
        """
        MONGODB_ATLAS_CLUSTER_URI=os.getenv("MONGODB_ATLAS_CLUSTER_URI")
        DB_NAME=application_name
        COLLECTION_NAME=f"{application_name}_docs"
        ATLAS_VECTOR_SEARCH_INDEX_NAME=os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")
        
        # Initialize MongoDB client
        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        
        # Perform vector search
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=self.embedding_model,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
        )
        
        results = vector_store.similarity_search(query, k=k)
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append(doc.page_content)
        return formatted_results

    def rag(self, query: str, sessionid: str,application_name:str) -> str:
        """
        Generate an answer for the given query using the RAG pipeline.

        Args:
            query (str): The query to answer.
            sessionid (str): The session ID for memory retrieval.

        Returns:
            str: The generated answer.
        """
        # Step 1: Retrieve relevant documents
        try:
            context_docs = self.retrieve_documents(query, k=3,application_name=application_name)
        except Exception as e:
            return f"Error retrieving documents: {e}"

        
        return context_docs