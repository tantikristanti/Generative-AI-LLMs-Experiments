from llama_index.core.prompts import RichPromptTemplate

def generate_prompt_standard():
    """
    Create a customizing prompt template for the chatbot.
    The default existing prompts by LlamaIndex can be accessed from the query engine: query_engine.get_prompts()

    Source: https://docs.llamaindex.ai/en/stable/examples/customization/prompts/chat_prompts/
    
    Returns:
        formatted_prompt (RichPromptTemplate): The formatted prompt 
    """
    
    # Create a prompt template from a string
    qa_prompt = """
                {% chat role="system" %}
                Always answer the question, even if the context isn't helpful.
                {% endchat %}

                {% chat role="user" %}
                The following is some retrieved context:

                <context>
                {{ context_str }}
                </context>

                Using both the retrieved context and the following question, provide precise and well-structured answers based solely on the context, without any speculation or assumptions.
                {{ query_str }}
                {% endchat %}
                              """
    formatted_prompt = RichPromptTemplate(qa_prompt)

    return formatted_prompt

def generate_prompt_refine():
    """
    Create a customizing prompt template for the chatbot.
    The default existing prompts by LlamaIndex can be accessed from the query engine: query_engine.get_prompts()

    Source: https://docs.llamaindex.ai/en/stable/examples/customization/prompts/chat_prompts/
    
    Returns:
        formatted_prompt (RichPromptTemplate): The formatted prompt 
    """
    
    # Create a prompt template from a string
    qa_prompt = """
                {% chat role="system" %}
                Always answer the question, even if the context isn't helpful.
                {% endchat %}

                {% chat role="user" %}
                The following is some new retrieved context:

                <context>
                {{ context_msg }}
                </context>

                And here is an existing answer to the query:
                <existing_answer>
                {{ existing_answer }}
                </existing_answer>

                Using both the new retrieved context and the existing answer, either update or repeat the existing answer to this query:
                {{ query_str }}
                {% endchat %}
                """
    formatted_prompt = RichPromptTemplate(qa_prompt)

    return formatted_prompt