def generate_prompt():
    """
    Create a template prompt for the chatbot.

    Returns:
        formatted_prompt (str): The formatted prompt 
    """
    
    # Create a prompt template
    # The context is the relevant context retrieved from the document and the question is the question from the end-user
    chat_prompt = """Provide a precise and well-structured answer based solely on the following context, without any speculation or assumption:
    {context}
    Question: {question}
    """
    return chat_prompt
