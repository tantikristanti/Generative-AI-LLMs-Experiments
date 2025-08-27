from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
from chromadb.config import Settings
from chromadb import Client

def generate_embedding(chunk, model_embedding):
    """
    Generate embeddings for a given chunk of text.

    Args:
        chunk (str): The chunk of text
        model_embedding (str): The name of the embedding model to use (e.g. "nomic-embed-text")

    Returns:
        embedding: The embedding for each chunk
    """
    
    return model_embedding.embed_query(chunk.page_content)

def generate_embeddings(doc_splits, model_name):
    """
    Generate embeddings for the entire text.

    Args:
        doc_splits (list): The list of LangChain Document for storing a piece of text and associated metadata
        model_name (str): The name of the LLM to use (e.g. "deepseek-r1")

    Returns:
        embeddings (list): The generated embeddings
    """
    
    # Initialize the embedding
    model_embedding = OllamaEmbeddings(model=model_name)
    # Parallelize embedding generation
    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(generate_embedding, doc_splits, [model_embedding] * len(doc_splits)))
    return embeddings

def create_collection(doc_splits, embeddings, collection_name):
    """
    Create a collection consisting of documents, embeddings, and additional metadata.

    Args:
        doc_splits (list): The list of LangChain Document for storing a piece of text and associated metadata
        embeddings (str): The name of the LLM to use (e.g. "deepseek-r1")
        collection_name (str): The name of the collection to create or reset if it already exists
    """
    
    # Initialize Chroma client
    client = Client(Settings())
    
    # Create or reset the collection
    collection = client.get_or_create_collection(collection_name)
    
    # Add documents and embeddings to the collection
    for idx, chunk in enumerate(doc_splits):
        collection.add(
            documents=[chunk.page_content], 
            metadatas=[{'id': idx}], 
            embeddings=[embeddings[idx]], 
            ids=[str(idx)]  
        )
        
def create_vectorstore(doc_splits, collection_name, model_name):
    """
    Create a vector store to store the collection of documents and their embeddings.

    Args:
        doc_splits (list): The list of LangChain Document for storing a piece of text and associated metadata
        collection_name (str): The name of the collection to create or reset if it already exists
        model_name (str): The name of the LLM to use (e.g. "deepseek-r1")

    Returns:
        vectorStore: The vector store 
    """
    
    # Create a vector store
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=OllamaEmbeddings(model=model_name),
        persist_directory="./chroma_db"
    )
    
    return vectorstore

