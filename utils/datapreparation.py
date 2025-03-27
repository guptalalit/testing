import os
import numpy as np
from typing import List, Dict
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import utils.google_llm_services as googleserv


MONGODB_ATLAS_CLUSTER_URI=os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME=os.getenv("DB_NAME")
COLLECTION_NAME=os.getenv("COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME=os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384  # For all-MiniLM-L6-v2 model

def load_text_files(folder_path: str) -> Dict[str, str]:
    """Same as original implementation"""
    file_contents = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                file_contents[file_name] = file.read()
        else:
            file_path = os.path.join(folder_path, file_name)
            file_contents[file_name] = googleserv.GoogleLLM().getResults(file_path)
    return file_contents

def create_chunks(text: str, chunk_size: int) -> List[str]:
    """Same as original implementation"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def prepare_data(file_contents: Dict[str, str]) -> List[Document]:
    """Convert file contents to LangChain Documents"""
    documents = []
    for file_name, content in file_contents.items():
        chunks = create_chunks(content, CHUNK_SIZE)
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"file_name": file_name}
            ))
    return documents


def execute_data(data_folder, application_name):
    # Step 1: Load and prepare documents
    file_contents = load_text_files(data_folder)
    documents = prepare_data(file_contents)
    print("Total Documents: ",len(documents))
    # Step 2: MongoDB connection
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    print("application_name", application_name)
    # Check if the database for the specific application exists
    db = client[application_name]  # Use the application name as the database name
    
    # Create collection names dynamically based on application name
    main_collection_name = f"{application_name}_docs"
    qa_collection_name = f"{application_name}_qa"

    # Check if the main collection exists
    main_collection = db[main_collection_name]
    main_collection_exists = main_collection_name in db.list_collection_names()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if not main_collection_exists:
        # If the main collection doesn't exist, create it and the vector index
        vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=main_collection,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine"
        )
        vector_store.create_vector_search_index(dimensions=384)
        print(f"Created new vector index for {main_collection_name} in application {application_name}")
    else:
        # If the collection exists, use the existing one
        vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=main_collection,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine"
        )
        print(f"Using existing main collection {main_collection_name} for application {application_name}")

    # Always add/update documents in the main collection
    vector_store.add_documents(documents=documents)
    print(f"Updated {len(documents)} documents in collection {main_collection_name} for application {application_name}")
    
    # Check if the QA collection exists
    qa_collection = db[qa_collection_name]
    qa_collection_exists = qa_collection_name in db.list_collection_names()
    
    if not qa_collection_exists:
        # If the QA collection doesn't exist, create it and the vector index
        qa_vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=qa_collection,
            index_name="qa_vector_index",
            relevance_score_fn="cosine"
        )
        qa_vector_store.create_vector_search_index(dimensions=384)
        print(f"Created new QA vector index for {qa_collection_name} in application {application_name}")
    
    print(f"Data update completed for application {application_name} without index recreation")

