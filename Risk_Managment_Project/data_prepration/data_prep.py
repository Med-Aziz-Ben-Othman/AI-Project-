import json
import os

# Helper function to merge attributes with string cleanup and deduplication
def merge_attributes(existing_attrs, new_attrs):
    def clean_string(s):
        # Lowercase, strip extra spaces, and remove duplicate substrings separated by semicolons
        return "; ".join(sorted(set(part.strip() for part in s.lower().split(';') if part.strip())))

    for key, value in new_attrs.items():
        if key in existing_attrs:
            if isinstance(existing_attrs[key], list) and isinstance(value, list):
                # Merge lists by appending unique lowercase items, removing duplicates
                existing_attrs[key] = sorted(set(item.strip().lower() for item in existing_attrs[key] + value))
            elif isinstance(existing_attrs[key], str) and isinstance(value, str):
                # Merge strings with semicolon separation, removing duplicates and cleaning
                combined_value = existing_attrs[key] + "; " + value
                existing_attrs[key] = clean_string(combined_value)
        else:
            # Add new attribute, clean it if it's a string or clean its list if it's a list
            if isinstance(value, list):
                existing_attrs[key] = sorted(set(item.strip().lower() for item in value))
            elif isinstance(value, str):
                existing_attrs[key] = clean_string(value)
            else:
                existing_attrs[key] = value
    return existing_attrs

# Function to merge nodes with attribute cleanup
def merge_nodes(first_nodes, second_nodes):
    merged_nodes = {node['name']: node for node in first_nodes}  # Dictionary for fast lookup

    for node in second_nodes:
        if node['name'] in merged_nodes:
            # Merge attributes of existing nodes with cleanup
            merged_nodes[node['name']]['attributes'] = merge_attributes(merged_nodes[node['name']]['attributes'], node['attributes'])
        else:
            # If the node doesn't exist, add it
            merged_nodes[node['name']] = node

    return list(merged_nodes.values())  # Convert back to list

# Function to merge relationships with redundancy removal
def merge_relationships(first_relationships, second_relationships):
    # Combine relationships, removing duplicates by using a set of tuples
    merged_relationships = first_relationships + second_relationships
    seen = set()
    unique_relationships = []

    for rel in merged_relationships:
        rel_tuple = (rel['source'], rel['relationship'], rel['target'])
        if rel_tuple not in seen:
            unique_relationships.append(rel)
            seen.add(rel_tuple)

    return unique_relationships

# Function to extract nodes and relationships from second_data (which is a list of dictionaries)
def extract_nodes_relationships(data_list):
    all_nodes = []
    all_relationships = []
    for data in data_list:
        if 'nodes' in data:
            all_nodes.extend(data['nodes'])
        if 'relationships' in data:
            all_relationships.extend(data['relationships'])
    return all_nodes, all_relationships

# Main function to load, merge, and save the data
def main():
    # Define the paths to your input JSON files
    first_file = "/content/extracted_conceptual_graph_data_full_text_refined.json"
    second_file = "/content/extracted_conceptual_graph_data_refined.json"
    output_file = "/content/merged_data.json"
    
    # Check if the files exist
    if not os.path.exists(first_file):
        print(f"File not found: {first_file}")
        return
    if not os.path.exists(second_file):
        print(f"File not found: {second_file}")
        return

    # Load your two JSON data
    with open(first_file, 'r') as f:
        first_data = json.load(f)

    with open(second_file, 'r') as f:
        second_data = json.load(f)

    # Extract nodes and relationships from the second JSON (handling the list structure)
    second_nodes, second_relationships = extract_nodes_relationships(second_data)

    # Merge the nodes and relationships with cleanup
    merged_nodes = merge_nodes(first_data['nodes'], second_nodes)
    merged_relationships = merge_relationships(first_data['relationships'], second_relationships)

    # Create the merged data structure
    merged_data = {
        "nodes": merged_nodes,
        "relationships": merged_relationships
    }

    # Save to output JSON file
    with open(output_file, "w") as outfile:
        json.dump(merged_data, outfile, indent=4)

    print(f"Merging completed successfully. Merged data saved to '{output_file}'.")

if __name__ == "__main__":
    main()
