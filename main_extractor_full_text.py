import pandas as pd
import os
import sys
import json
from tqdm import tqdm
import spacy

nlp = spacy.load('en_core_web_lg')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Risk_Managment_Project', 'Llms')))
from data_extractor import LlmDataExtractor

def extract_conceptual_graph_data(text):
    """
    Extract nodes, relationships, and attributes from the full text for GCN model and save LLM responses.
    
    Args:
        text (str): The entire text from the PMI PDF book as one document.
        
    Returns:
        dict: A dictionary with nodes, relationships, and attributes extracted from the entire text.
    """
    extractor = LlmDataExtractor()

    # Ensure 'respons' folder exists
    output_folder = 'respons_all_text'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Send the whole text as a single prompt to the LLM
    prompt = f"Analyze the following text and extract nodes, relationships, and attributes related to project management risks:\n\n{text}"
    
    # Get response from LLM
    response = extractor.generate_response(prompt)
    print(f"Response for the entire text:\n{response}\n")

    # Save the LLM response to a file in the 'respons' folder
    if response:
        response_file = os.path.join(output_folder, 'response_full_text.txt')
        with open(response_file, 'w', encoding='utf-8') as file:
            file.write(response)
        
        # Process the response into structured data
        nodes = []
        relationships = []
        attributes = {}

        # Split response into lines and parse nodes, relationships, and attributes
        lines = response.splitlines()
        current_section = None

        for line in lines:
            line = line.strip()

            # Identify section headers
            if line.startswith("### Nodes"):
                current_section = 'nodes'
            elif line.startswith("### Relationships"):
                current_section = 'relationships'
            elif line.startswith("### Attributes"):
                current_section = 'attributes'
            elif line:
                if current_section == 'nodes':
                    node = line.split(". ")[-1].strip()
                    nodes.append(node)
                elif current_section == 'relationships':
                    relationships.append(line)
                elif current_section == 'attributes':
                    key_value = line.split(":", 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        attributes[key] = value

        # Return a structured dictionary containing the extracted data
        return {
            'nodes': nodes,
            'relationships': relationships,
            'attributes': attributes
        }

    return None

def main():
    # Load the sentences DataFrame
    df_pmi = pd.read_csv('Risk_Managment_Project/data/df_pmi_sent.csv')
    
    # Concatenate all sentences into one large text
    full_text = ". ".join(df_pmi['sentence'].tolist())
    docPMI = nlp(full_text)  # Process the full text with SpaCy if needed
    
    # Extract conceptual graph data from the entire document
    conceptual_graph_data = extract_conceptual_graph_data(docPMI.text)
    
    if conceptual_graph_data:
        # Save the structured output to a JSON file with the keys 'nodes', 'relationships', 'attributes'
        output_file = "extracted_conceptual_graph_data_full_text.json"
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(conceptual_graph_data, file, indent=4)
        print(f"\nData has been written to {output_file}")
    else:
        print("No data was extracted from the text.")

if __name__ == "__main__":
    main()
