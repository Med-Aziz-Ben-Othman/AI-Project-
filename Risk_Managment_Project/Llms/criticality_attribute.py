# llm_data_extractor.py
import sys
import os
import openai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Constants.config import config

class LlmDataExtractor:
    def __init__(self):
        config_instance = config()  # Renaming to avoid conflict
        config_instance.setup_openai()  # Setup OpenAI API
        
    def generate_response(self, node_name, attributes):
        # Create a suitable system prompt for the task
        system_prompt = '''You are a project management assistant. Based on the provided information about a project risk management node, your task is to assess and classify the risk criticality of the node. 

        Given the node's name and its attributes, determine the risk criticality level: Low, Medium, or High. Respond only with one of these three classifications. 
        Avoid providing explanations or justifications, and ensure that your response is clear and concise. 
        '''
        
        try:
            prompt = f"{system_prompt}\nNode name: {node_name}\nAttributes: {attributes}"
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Choose the appropriate OpenAI model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more deterministic output
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
