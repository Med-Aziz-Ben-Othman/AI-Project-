# llms/mistral_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer

def get_mistral_model():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    return model, tokenizer
