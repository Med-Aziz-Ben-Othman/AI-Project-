import pandas as pd
import os
import sys
from tqdm import tqdm  # Import tqdm for progress visualization
import spacy
import json

nlp = spacy.load('en_core_web_lg')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Risk_Managment_Project', 'Llms')))
from data_extractor import LlmDataExtractor  # Import the LlmDataExtractor class

def extract_conceptual_graph_data(df):
    """
    Extract nodes, relationships, and attributes from the DataFrame for GCN model and save LLM responses.
    
    Args:
        df (pd.DataFrame): DataFrame containing sentences from the PMI PDF book.
        
    Returns:
        list: A list of dictionaries with nodes, relationships, and attributes for each sentence.
    """
    extractor = LlmDataExtractor()
    structured_data = []

    # Ensure 'respons' folder exists
    output_folder = 'respons'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each sentence in the DataFrame with tqdm progress bar
    for idx, sentence in tqdm(enumerate(df['sentence']), desc="Processing sentences", total=len(df)):
        prompt = f"Analyze the following sentence and extract nodes, relationships, and attributes related to project management risks:\n\n{sentence}"
        
        # Get response from LLM
        response = extractor.generate_response(prompt)
        print(f"Response for sentence {idx}: '{sentence}'\n{response}\n")
        
        # Save the LLM response to a file in the 'respons' folder
        if response:
            response_file = os.path.join(output_folder, f'response_{idx}.txt')
            with open(response_file, 'w', encoding='utf-8') as file:
                file.write(response)
            
            # Process the response into structured data
            nodes = []
            relationships = []
            attributes = {}

            # Split response into lines and parse nodes, relationships, and attributes
            lines = response.splitlines()
            current_section = None  # Track which section we're in (Nodes, Relationships, or Attributes)

            for line in lines:
                line = line.strip()

                # Identify section headers
                if line.startswith("### Nodes"):
                    current_section = 'nodes'
                elif line.startswith("### Relationships"):
                    current_section = 'relationships'
                elif line.startswith("### Attributes"):
                    current_section = 'attributes'
                elif line:  # Process line based on the current section
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
            
            # Append the structured data for this sentence to the list
            structured_data.append({
                'sentence': sentence,
                'nodes': nodes,
                'relationships': relationships,
                'attributes': attributes
            })

    return structured_data


def main():
    # Load your DataFrame (assuming it's stored in a CSV for this example)
    df_pmi = pd.read_csv('Risk_Managment_Project/data/df_pmi_sent.csv')  # Adjust path as necessary
    docPMI = nlp(". ".join(df_pmi.sentence))
    
    # Extract conceptual graph data
    conceptual_graph_data = extract_conceptual_graph_data(df_pmi)
    
    # Optionally, save the structured output to a JSON file
    output_file = "extracted_conceptual_graph_data.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(conceptual_graph_data, file, indent=4)
    print(f"\nData has been written to {output_file}")

if __name__ == "__main__":
    main()
