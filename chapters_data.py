import pandas as pd
import os
import sys
from tqdm import tqdm  # Import tqdm for progress visualization
import spacy
import json

nlp = spacy.load('en_core_web_lg')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Risk_Managment_Project', 'Llms')))
from data_extractor import LlmDataExtractor  # Import the LlmDataExtractor class

def extract_conceptual_graph_data(full_chapter_text, chapter_name):
    """
    Extract nodes, relationships, and attributes from the entire chapter text using LLM and save responses.
    
    Args:
        full_chapter_text (str): Full text of the chapter.
        chapter_name (str): Name of the chapter being processed.
        
    Returns:
        dict: A dictionary with nodes, relationships, and attributes for the chapter.
    """
    extractor = LlmDataExtractor()

    # Ensure 'output/output_chapters' folder exists
    output_folder = 'output/output_chapters'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    prompt = f"Analyze the following chapter and extract nodes, relationships, and attributes related to project management risks:\n\n{full_chapter_text}"
    
    # Get response from LLM for the whole chapter
    response = extractor.generate_response(prompt)
    print(f"Response for chapter {chapter_name}:\n{response}\n")
    
    # Process the response into structured data
    nodes = []
    relationships = []
    attributes = {}

    # Split response into lines and parse nodes, relationships, and attributes
    if response:
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
    
    # Create structured data for the whole chapter
    structured_data = {
        'chapter': chapter_name,
        'nodes': nodes,
        'relationships': relationships,
        'attributes': attributes
    }

    # Save the output of the chapter to a JSON file in the output folder
    chapter_output_file = os.path.join(output_folder, f'extracted_data_{chapter_name}.json')
    with open(chapter_output_file, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=4)
    print(f"Data for {chapter_name} saved to {chapter_output_file}")

    return structured_data

def main():
    # Folder where your chapter DataFrames are stored
    chapter_folder = 'Risk_Managment_Project/data/chapters'
    
    # Ensure the folder exists
    if not os.path.exists(chapter_folder):
        print(f"The folder '{chapter_folder}' does not exist.")
        return

    # Loop through each chapter CSV file in the folder
    for chapter_file in os.listdir(chapter_folder):
        if chapter_file.endswith('.csv'):  # Process only CSV files
            chapter_path = os.path.join(chapter_folder, chapter_file)
            chapter_name = os.path.splitext(chapter_file)[0]  # Get the chapter name (without .csv extension)
            
            # Load the DataFrame for this chapter
            df_chapter = pd.read_csv(chapter_path)
            
            # Combine all sentences in the chapter into one large text
            full_chapter_text = " ".join(df_chapter['Sentence'].tolist())
            print(f"Processing chapter: {chapter_name}")
            
            # Process the whole chapter and extract conceptual graph data
            extract_conceptual_graph_data(full_chapter_text, chapter_name)

if __name__ == "__main__":
    main()
