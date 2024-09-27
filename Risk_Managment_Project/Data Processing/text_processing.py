import os
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from datetime import datetime

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to create the output folder if it doesn't exist
def create_output_folder(folder_name="output_normal"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Function to save extracted text into the output folder
def save_text_to_file(text, folder="output_normal", file_name="extracted_text"):
    create_output_folder(folder)
    
    # Create a unique filename with timestamp to avoid overwriting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(folder, f"{file_name}_{timestamp}.txt")
    
    # Save the text in the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Extracted text saved to {file_path}")
    return file_path

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Tokenization and normalization
def tokenize_and_clean(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove punctuation and make lowercase
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# Lemmatization
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

if __name__ == "__main__":
    # Example usage
    pdf_path = 'path_to_your_pdf.pdf'  # Replace with your actual PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Save the extracted text to the output folder
    save_text_to_file(extracted_text)
    
    # Perform tokenization and lemmatization
    cleaned_tokens = tokenize_and_clean(extracted_text)
    lemmatized_tokens = lemmatize_tokens(cleaned_tokens)
    
    print("Cleaned and lemmatized tokens:", lemmatized_tokens[:20])
