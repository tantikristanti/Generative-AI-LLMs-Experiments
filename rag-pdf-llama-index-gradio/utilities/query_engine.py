from llama_index.core.query_engine import RetrieverQueryEngine
from utilities.prompt_format import generate_prompt_standard, generate_prompt_refine

def construct_query_engine(retriever, llm):
    """
    Construct a query engine.

    Args:
        retriever (VectorDBRetriever): The retriever
        llm (LlamaCPP): The Llama CPP models

    Returns:
        query_engine (RetrieverQueryEngine): The query engine
    """

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    
    query_engine.update_prompts(
                                    {
                                        # list the templates 
                                        "response_synthesizer:generate_prompt_standard": generate_prompt_standard(),
                                        "response_synthesizer:generate_prompt_refine": generate_prompt_refine(),
                                    }
                                )
    
    return query_engine