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
        dict: A structured representation of nodes, relationships, and attributes.
    """
    extractor = LlmDataExtractor()
    structured_data = {
        'nodes': set(),  # Use sets to avoid duplicates
        'relationships': [],
        'attributes': {}
    }
    
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
            with open('output_file.txt', 'w', encoding='utf-8') as file:
                file.write(response)

            
            # Process the response into structured data by splitting and identifying patterns.
            lines = response.splitlines()
            for line in lines:
                line = line.strip()
                
                # Use more flexible extraction, assuming LLM identifies nodes, relationships, and attributes
                if " --" in line and "--> " in line:  # Check if the line looks like a relationship
                    relationship = line.strip()
                    structured_data['relationships'].append(relationship)
                    
                    # Extract nodes from the relationship pattern
                    nodes = relationship.split('--')
                    if len(nodes) == 2:
                        node_a = nodes[0].strip()
                        node_b = nodes[1].split('>')[-1].strip()
                        structured_data['nodes'].update([node_a, node_b])  # Add both nodes
                
                elif ':' in line:  # Attribute pattern
                    attribute = line.split(':', 1)
                    key = attribute[0].strip()
                    value = attribute[1].strip()
                    structured_data['attributes'][key] = value
                    
                elif "Node" in line:  # If LLM explicitly mentions nodes
                    node = line.replace("Node:", "").strip()
                    structured_data['nodes'].add(node)

    return {
        'nodes': list(structured_data['nodes']),  # Convert sets back to lists for JSON serialization
        'relationships': structured_data['relationships'],
        'attributes': structured_data['attributes']
    }


def main():
    # Load your DataFrame (assuming it's stored in a CSV for this example)
    df_pmi = pd.read_csv('Risk_Managment_Project/data/df_pmi_sent.csv')  # Adjust path as necessary
    docPMI = nlp(". ".join(df_pmi.sentence))
    
    # Extract conceptual graph data
    conceptual_graph_data = extract_conceptual_graph_data(df_pmi)

    # Output the extracted data
    print("Extracted Nodes:")
    print(conceptual_graph_data['nodes'])
    
    print("\nExtracted Relationships:")
    print(conceptual_graph_data['relationships'])
    
    print("\nExtracted Attributes:")
    print(conceptual_graph_data['attributes'])

    # Optionally, save the structured output to a JSON file
    output_file = "extracted_conceptual_graph_data.json"
    with open(output_file, "w") as file:
        json.dump(conceptual_graph_data, file, indent=4)
    print(f"\nData has been written to {output_file}")

if __name__ == "__main__":
    main()
    