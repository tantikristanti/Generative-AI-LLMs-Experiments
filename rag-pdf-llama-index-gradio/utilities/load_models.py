from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP

def load_embedding_model(model_path):
    """
    Load the pre-trained embedding model. 
    If the model_name is a filepath on disc, it loads the model from that path. 
    If it is not a path, it first tries to download a pre-trained SentenceTransformer model.

    Args:
        model_path (str): The path to the embedding model or the model name

    Returns:
        HuggingFaceEmbedding: The HuggingFace embedding model
    """
    
    embed_model = HuggingFaceEmbedding(model_name=model_path)
    return embed_model

def load_llm(#model_url, 
             model_path, 
             temperature=0.1, 
             max_new_tokens=256, 
             context_window=3000, 
             n_gpu_layers=1):
    """
    Load an LLM. 
    If the model_name is a filepath on disc, it loads the model from that path. 
    If it is not a path, it first tries to download a pre-trained SentenceTransformer model.

    Args:
        model_path (str): The path to the embedding model or the model name
        temperature (float): Temperature in LLM; the higher the temperature, the more creative the LLM response
        max_new_tokens (int): The max output tokens
        context_window (int): The maximum capacity for storing conversation history and generating new output 
        n_gpu_layers (int): The number of GPUs used 

    Returns:
        LlamaCPP: The LlamaCPP LLM
    """
    
    llm = LlamaCPP(
        # Pass the URL to a GGML model to download it automatically
        # model_url=model_url,
        # OR, set the path to load a pre-downloaded model 
        model_path=model_path,#None,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        # llama3.2 has a context window of 8k tokens, but it will be set lower due to limited computational resources
        context_window=context_window,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 1},
        verbose=True,
    )
    return llm