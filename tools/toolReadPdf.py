import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from memory.memory import Memory

class RAG:
    def __init__(self):
        # Initialize paths and models
        self.index_file = "faiss_index.index"
        self.metadata_file = "metadata.json"
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Configure Gemini API
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def retrieve_documents(self, query: str, k: int = 3) -> list:
        """
        Retrieve the top-k most relevant documents for the given query using FAISS.

        Args:
            query (str): The query to search for.
            k (int): The number of documents to retrieve.

        Returns:
            list: A list of dictionaries containing the retrieved documents and their metadata.
        """
        # Load the FAISS index
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"FAISS index file '{self.index_file}' not found.")
        index = faiss.read_index(self.index_file)

        # Load the metadata
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file '{self.metadata_file}' not found.")
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Generate the query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)

        # Perform the similarity search
        distances, indices = index.search(query_embedding, k)

        # Retrieve the top-k documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                result = metadata[idx]
                result["distance"] = float(distances[0][i])  # Add the distance to the result
                results.append(result)

        return results

    def rag(self, query: str) -> str:
        """
        Generate an answer for the given query using the RAG pipeline.

        Args:
            query (str): The query to answer.

        Returns:
            str: The generated answer.
        """
        # Step 1: Retrieve relevant documents
        try:
            context_docs = self.retrieve_documents(query, k=3)
        except Exception as e:
            return f"Error retrieving documents: {e}"

        # Step 2: Format the context for the prompt
        context = "\n\n".join([doc["text"] for doc in context_docs])
        # Create an instance of Memory
        memory = Memory()
        prev_conversation=memory.search_conversations(query=query,k=1)
        print("\n Prev Conversation: ",prev_conversation)
        # Step 3: Generate the prompt
        prompt = f"""Provide the answer for the given question based on the context.
            Question: {query}
            Context: {context}
            Previous Conversation:{prev_conversation}
            """

        # Step 4: Generate the answer using Gemini
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text,context
        except Exception as e:
            return f"Error generating answer: {e}"


