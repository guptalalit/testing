import os
import numpy as np
from typing import List, Dict
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import utils.google_llm_services as googleserv
from langchain_community.vectorstores import FAISS

MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "mongodb")  # Default to MongoDB
# FAISS_STORAGE_PATH = "./faiss_store"
FAISS_STORAGE_PATH = os.getenv("FAISS_STORAGE_PATH", "./faiss_store")
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
                metadata={"file_name": file_name, "type": "document"}
            ))
    return documents

def initialize_mongodb(application_name: str, documents: List[Document]):
    """Original MongoDB implementation"""
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    print("application_name", application_name)
    db = client[application_name]
    
    # Document collection setup
    main_collection_name = f"{application_name}_docs"
    main_collection = db[main_collection_name]
    main_collection_exists = main_collection_name in db.list_collection_names()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if not main_collection_exists:
        vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=main_collection,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine"
        )
        vector_store.create_vector_search_index(dimensions=384)
        print(f"Created new vector index for {main_collection_name} in application {application_name}")
    else:
        vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=main_collection,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine"
        )
        print(f"Using existing main collection {main_collection_name} for application {application_name}")

    vector_store.add_documents(documents=documents)
    print(f"Updated {len(documents)} documents in collection {main_collection_name}")
    
    # QA collection setup
    qa_collection_name = f"{application_name}_qa"
    qa_collection = db[qa_collection_name]
    qa_collection_exists = qa_collection_name in db.list_collection_names()
    
    if not qa_collection_exists:
        qa_vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=qa_collection,
            index_name="qa_vector_index",
            relevance_score_fn="cosine"
        )
        qa_vector_store.create_vector_search_index(dimensions=384)
        print(f"Created new QA vector index for {qa_collection_name}")

def initialize_faiss(application_name: str, documents: List[Document]):
    """Local FAISS implementation"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    app_storage_path = os.path.join(FAISS_STORAGE_PATH, application_name)
    
    # Document store
    docs_path = os.path.join(app_storage_path, "docs")
    index_file_path = os.path.join(docs_path, "index.faiss")
    if os.path.exists(index_file_path):
        docs_store = FAISS.load_local(docs_path, embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded existing FAISS documents store from {docs_path}")
    else:
        os.makedirs(docs_path, exist_ok=True)
        docs_store = FAISS.from_documents(documents, embeddings)
        docs_store.save_local(docs_path)
        print(f"Created new FAISS documents store at {docs_path}")
    
    # Add/update documents
    docs_store.add_documents(documents)
    docs_store.save_local(docs_path)
    print(f"Updated FAISS documents store with {len(documents)} documents")
    
    # QA store
    qa_path = os.path.join(app_storage_path, "qa")
    if os.path.exists(qa_path):
        qa_store = FAISS.load_local(qa_path, embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded existing FAISS QA store from {qa_path}")
    else:
        os.makedirs(qa_path, exist_ok=True)
        # Create empty QA store
        qa_store = FAISS.from_texts([""], embeddings)
        qa_store.delete([qa_store.index_to_docstore_id[0]])  # Remove dummy document
        print(f"Created new FAISS QA store at {qa_path}")
    
    qa_store.save_local(qa_path)

def execute_data(data_folder, application_name):
    # Step 1: Load and prepare documents
    file_contents = load_text_files(data_folder)
    documents = prepare_data(file_contents)
    print("Total Documents: ", len(documents))
    
    if DATABASE_TYPE == "mongodb":
        initialize_mongodb(application_name, documents)
    elif DATABASE_TYPE == "local":
        initialize_faiss(application_name, documents)
    else:
        raise ValueError(f"Unsupported database type: {DATABASE_TYPE}")
    
    print(f"Data update completed for application {application_name} using {DATABASE_TYPE}")