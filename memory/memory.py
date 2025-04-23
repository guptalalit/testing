from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from datetime import datetime
import os
from typing import List

class Memory:
    def __init__(self, application_name: str):
        """
        Unified memory storage supporting both MongoDB and FAISS
        """
        self.storage_type = os.getenv("DATABASE_TYPE", "mongodb")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.application_name = application_name

        if self.storage_type == "mongodb":
            self._init_mongodb()
        elif self.storage_type == "local":
            self._init_faiss()
        else:
            raise ValueError(f"Unsupported database type: {self.storage_type}")

    def _init_mongodb(self):
        """Initialize MongoDB connection"""
        self.mongodb_uri = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
        self.client = MongoClient(self.mongodb_uri)
        self.collection = self.client[self.application_name][f"{self.application_name}_qa"]
        
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="qa_vector_index",
            relevance_score_fn="cosine"
        )

    def _init_faiss(self):
        """Initialize FAISS vector store"""
        self.storage_path = os.path.join(
            os.getenv("FAISS_STORAGE_PATH", "./memory_store"),
            self.application_name
        )
        os.makedirs(self.storage_path, exist_ok=True)

        try:
            self.vector_store = FAISS.load_local(
                self.storage_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except:
            # Create new store with empty document
            self.vector_store = FAISS.from_texts([""], self.embeddings)
            # Remove dummy document
            self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])
            self.vector_store.save_local(self.storage_path)

    def add_conversation(self, query: str, answer: str, sessionid: str):
        """
        Add conversation with consistent format for both storage types
        """
        # Maintain original document format
        conversation_doc = Document(
            page_content=f"Query: {query}\nAnswer: {answer}",
            metadata={
                "query": query,
                "answer": answer,
                "sessionid": sessionid,
                "timestamp": datetime.now().isoformat(),
                "feedback":"positive"
            }
        )
        
        # Add to vector store
        self.vector_store.add_documents([conversation_doc])
        
        # Explicit save for FAISS
        if self.storage_type == "local":
            self.vector_store.save_local(self.storage_path)
        
        print(f"Added conversation to {self.storage_type.upper()} storage")

    def search_conversations(self, query: str, k: int = 3) -> List[str]:
        """
        Search conversations with identical interface for both storage types
        """
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def update_feedback(self, query: str, answer: str, sessionid: str, feedback: str):
        """
        Update feedback for a specific conversation using query/answer/sessionid
        """
        if self.storage_type == "mongodb":
            # MongoDB update
            result = self.collection.update_one(
                {
                    "query": query,
                    "answer": answer,
                    "sessionid": sessionid
                },
                {"$set": {"feedback": feedback}},
                upsert=False  # Avoid creating a new document if none matches
            )
            if result.modified_count > 0:
                print("Updated feedback successfully!")
            else:
                print("No matching document found to update.")
        else:
            # FAISS update without deletion
            updated = False

            # Access the docstore dictionary
            docstore_dict = self.vector_store.docstore._dict

            for doc_id, doc in docstore_dict.items():
                meta = doc.metadata
                if (
                    meta.get("query") == query and
                    meta.get("answer") == answer and
                    meta.get("sessionid") == sessionid
                ):
                    print("Data: ", meta)
                    # Update feedback in metadata
                    meta["answer"] = answer + "\nFeedback for the answer: " + feedback
                    updated = True
                    break

            if updated:
                self.vector_store.save_local(self.storage_path)
                print("Updated feedback in FAISS successfully!")
            else:
                print("No matching document found in FAISS to update.")
