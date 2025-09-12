import re
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utilities.prompt_format import generate_prompt

def generate_response(model_name, question, retriever):
    """
    Maintain communication between the user and the chatbot.

    Args:
        model_name (str): The name of the LLM to use (e.g. "deepseek-r1")
        question (str): The question from the end-user
        retriever (object): The component to retrieve the documents from the vector store
        
    Returns:
        final_response (str): The LLM responses after cleaning 
    """
    
    # Instantiate the conversation object
    chat_model = ChatOllama(model=model_name)

    # Create a chat prompt template from a template string
    chat_prompt = ChatPromptTemplate.from_template(generate_prompt())
    
    # Create a chat prompt chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt
        | chat_model
        | StrOutputParser()
    )
    
    # Generate the response 
    response = rag_chain.invoke(question)
    
    # Remove the <think> tags from the response  
    final_response = re.sub(r'<think>.*?</think>', '', str(response), flags=re.DOTALL).strip()
    return final_response