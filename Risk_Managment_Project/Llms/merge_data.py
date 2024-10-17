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

Your task is to analyze the provided project management text hierarchically and return it in a JSON format that includes:
1. **Nodes**: An array of nodes with properties like name, definition, and attributes. Ensure no nodes are omitted.
2. **Relationships**: An array of relationships that illustrate how nodes connect, especially:
   - Each chapter node should be connected to its corresponding parent node (full context).
   - Each sentence node should be connected to its corresponding chapter node.
3. **Attributes**: A dictionary of attributes related to the nodes.

Please ensure the output is valid JSON and clearly structured with no redundancy. 
'''},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Control creativity
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
