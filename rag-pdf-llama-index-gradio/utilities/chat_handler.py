def generate_response(query_engine, query):
    """
    Maintain communication between the user and the chatbot.

    Args:
        query_engine (RetrieverQueryEngine): The query engine
        query (str): The question from the end-user
        
    Returns:
        response (str): The LLM responses  
    """
    
    # Generate the response 
    response = query_engine.query(query)
    
    return str(response)