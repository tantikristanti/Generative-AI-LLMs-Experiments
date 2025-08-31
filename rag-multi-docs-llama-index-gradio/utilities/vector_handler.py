from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from typing import Optional

def generate_nodes_embedding(nodes, embed_model):
    """
    Generate embeddings for each node using a sentence transformers model.

    Args:
        nodes (list): The list of LlamaIndex nodes containing the text chunks and their corresponding source document metadata
        embed_model (str): The embedding model (e.g. "nomic-embed-text")

    Returns:
        nodes (list): The embedded nodes
    """

    # Generate embeddings for each node using a sentence transformers model
    for node in nodes:
        # Each node contains the embedding of the text and the text metadata
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
    
    return nodes

def initiate_vector_store(db_name, 
                        host, 
                        user, 
                        password, 
                        port, 
                        table_name="llama3_document",
                        embed_dim=384): # check the embedding model dimensions (384, 768) 
    
    """
    Initiate a Postgres vector store.

    Args:
        db_name, host, password, port, user, table_name (str): The data for creating a vector store
        embed_dim (int): The embedding dimension. The embedding model has different dimensional size. For example, the sentences transformer 'all-mpnet-base-v2' has a dimension size of 768, while 'all-MiniLM-L6-v2' and 'bge-small-en-v1.5' have a dimension size of 384.

    Returns:
        vector_store: The Postgres vector store
    """
    
    vector_store = PGVectorStore.from_params(database=db_name,
                                            host=host,
                                            password=password,
                                            port=port,
                                            user=user,
                                            table_name=table_name,
                                            embed_dim=embed_dim,  
                                            )
    return vector_store
        
def vectorstore_nodes(vector_store, nodes):
    """
    Save the embedded nodes into a vector store.

    Args:
        vector_store: The Postgres vector store
        nodes (list): The embedded nodes

    Returns:
        vectorStore: The vector store 
    """
    
    # Save the nodes into a vector store
    vector_store.add(nodes)
    
    return vector_store


    
