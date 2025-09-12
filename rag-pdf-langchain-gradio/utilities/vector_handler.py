import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from concurrent.futures import ThreadPoolExecutor
from chromadb.config import Settings
from chromadb import Client

def generate_embedding(chunk, embedding_model):
    """
    Generate embeddings for a given chunk of text.

    Args:
        chunk (str): The chunk of text
        embedding_model (str): The name of the embedding model to use (e.g. "nomic-embed-text")

    Returns:
        embedding: The embedding for each chunk
    """
    
    return embedding_model.embed_query(chunk.page_content)

def generate_embeddings(doc_splits, embedding_model):
    """
    Generate embeddings for the entire text.

    Args:
        doc_splits (list): The list of LangChain Document for storing a piece of text and associated metadata
        embedding_model (str): The name of the LLM to use (e.g. "deepseek-r1")

    Returns:
        embeddings (list): The generated embeddings
    """
    
    # Generate embeddings
    embed_model = HuggingFaceEmbeddings(model_name=embedding_model,
                                        #multi_process=True, # for multi-processing
                                        #model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu" },
                                        #encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
                                        )

    # Parallelize embedding generation
    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(generate_embedding, doc_splits, [embed_model] * len(doc_splits)))
    return embeddings

def create_collection(doc_splits, embeddings, collection_name="rag-chroma"):
    """
    Create a collection consisting of documents, embeddings, and additional metadata.

    Args:
        doc_splits (list): The list of LangChain Document for storing a piece of text and associated metadata
        collection_name (str): The name of the collection to create or reset if it already exists
        embeddings (str): The name of the LLM to use (e.g. "deepseek-r1")
        
    """
    
    # Initialize Chroma client using a persist directory
    #client = Client(Settings(persist_directory='./chroma_db'))
    client = Client()
    
    # Caution: errors when creating a collection may be caused by the presence of a previous collection with a different embedding model. If this occurs, delete the previous collection
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)
    
    # `get_or_create_collection` to avoid creating a new collection every time
    collection = client.get_or_create_collection(collection_name)
    
    # Add documents and embeddings to the collection 
    # If the collection does not exist yet or if the embeddings is built using a different embedding model, use 'add'
    # Otherwise, use 'upsert', instead of 'add', to avoid adding the same documents every time
    
    for idx, chunk in enumerate(doc_splits):
        collection.add(
        #collection.upsert(
            documents=[chunk.page_content], 
            metadatas=[{'id': idx}], 
            embeddings=[embeddings[idx]], 
            ids=[str(idx)]  
        )
        
def create_vector_store(doc_splits, model_name, collection_name="rag-chroma"):
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
        embedding=HuggingFaceEmbeddings(model_name=model_name),
        persist_directory='./chroma_db' #use previously persisted DB (if it exists)
    )
    
    return vectorstore