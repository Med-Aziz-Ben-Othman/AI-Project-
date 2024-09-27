import os
import json
from datetime import datetime
from llms.mistral_model import get_mistral_model
from constants.config import DEFAULT_MAX_TOKENS

# Function to ensure output folder exists
def create_output_folder(folder_name="output_GenAI"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Function to save extracted data to a file
def save_extracted_info_to_file(info, folder="output_GenAI", file_name="extracted_info"):
    create_output_folder(folder)
    
    # Create a unique filename with timestamp to avoid overwriting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(folder, f"{file_name}_{timestamp}.json")
    
    # Save as JSON for structured data storage
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    
    print(f"Extracted information saved to {file_path}")
    return file_path

# Function to extract information using the LLM model
def extract_prm_information(text):
    model, tokenizer = get_mistral_model()
    
    # Define the system message for guiding the model
    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled at extracting specific information about project risk management (PRM). Extract key attributes, motivations, and challenges mentioned in the provided text."},
        {"role": "user", "content": f"Please extract all relevant information about project risk management from the following text:\n\n{text}"}
    ]
    
    # Tokenize and prepare inputs
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    # Generate response using the model
    outputs = model.generate(inputs, max_new_tokens=DEFAULT_MAX_TOKENS)

    # Decode the output
    extracted_info = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return extracted_info

if __name__ == "__main__":
    # Example text (can be loaded from a file or other source)
    text = """
    Introduction
    This chapter presents the project’s context, focusing on challenges in project risk management
    and our proposed solution using deep learning and NLP.
    1.1 Project Context
    Motivation: The lack of standardization and consistency in project risk management (PRM)
    terminologies is a major challenge...
    """
    
    # Call the function to extract PRM information
    result = extract_prm_information(text)
    
    # Save the extracted information into the output folder
    save_extracted_info_to_file({"extracted_info": result})
