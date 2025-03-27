from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from datetime import datetime
import os


class Memory:
    def __init__(self, application_name: str):
        """
        Initialize MongoDB client and embeddings.
        """
        # Fetch MongoDB connection URI and index name from environment variables
        self.mongodb_uri = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
        self.atlas_vector_search_index_name = "qa_vector_index"
        
        # Initialize embedding model once during object creation
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize MongoDB client and store the application name for later use
        self.client = MongoClient(self.mongodb_uri)
        self.db_name = application_name
        self.collection_name = f"{application_name}_qa"
        
        # Ensure collections exist in the MongoDB database
        self.collection = self.client[self.db_name][self.collection_name]
        
        # Initialize the vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            embedding=self.embeddings,
            collection=self.collection,
            index_name=self.atlas_vector_search_index_name,
            relevance_score_fn="cosine"
        )

    def add_conversation(self, query: str, answer: str, sessionid: str):
        """
        Add a conversation (query and answer) to the QnA database.
        """
        # Create a document with metadata for the conversation
        conversation_doc = Document(
            page_content=f"Query: {query}\nAnswer: {answer}",
            metadata={
                "query": query,
                "answer": answer,
                "sessionid": sessionid,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Add the conversation document to the vector store (MongoDB)
        self.vector_store.add_documents(documents=[conversation_doc])
        print(f"Conversation added to QnA database for session {sessionid}.")

    def search_conversations(self, query: str, k: int = 3):
        """
        Search for similar conversations in the QnA database.
        """
        # Perform a similarity search to find the most relevant conversations
        results = self.vector_store.similarity_search(query, k=k)
        
        # Format the results to return just the content of each document
        formatted_results = [doc.page_content for doc in results]
        
        return formatted_results
