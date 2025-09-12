from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def quantize_llm(model_name):
    """
    Maintain communication between the user and the chatbot.

    Args:
        model_name (str): The name of the LLM to be quantized
        
    Returns:
        model: The quantized model
        tokenizer: The quantized tokenizer
    """
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer