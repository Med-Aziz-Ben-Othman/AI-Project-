import openai
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Constants.config import config  # Ensure the correct import path

class LlmDataExtractor:
    def __init__(self):
        config_instance = config()  # Renaming to avoid conflict
        config_instance.setup_openai()  # Setup OpenAI API
        
    def generate_response(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Choose the appropriate OpenAI model
                messages=[
                    {"role": "system", "content": '''You are a highly capable assistant focused on extracting knowledge from project management text to support graph-based learning models, such as Graph Convolutional Networks (GCNs). 

Your task is to analyze the provided project management text semantically, identifying key concepts, entities, and relationships between them relevant to project management and project risks. Structure your findings in a way that reflects a natural understanding of the domain, extracting entities (nodes), the relationships between them, and any attributes or characteristics.

Do not assume any predefined rules for extraction; instead, rely on your understanding of the text and the semantic relationships therein.'''},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Control creativity
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
