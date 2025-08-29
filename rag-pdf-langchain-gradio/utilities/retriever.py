def get_retriever(vector_store):
    """
    Create a retriever component to fetch relevant chunks from the stored embeddings in the vector store based on the end user queries.

    Args:
        vector_store: The Chroma vector store 

    Returns:
        retriever: The retriever
    """

    try:
        # Use search_type="similarity" for using Similarity Based Search and search_type="mmr" for MMR based Hybrid search
        # MMR (Maximum Marginal Relevance) search is used for balancing relevance and diversity 
        # Define the 'k' for the number of relevant document returned
        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    except Exception as e:
        print(f"Error when initializing the vector store: {e}")
        return None