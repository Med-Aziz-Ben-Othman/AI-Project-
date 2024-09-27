from data_processing.text_processing import extract_text_from_pdf, tokenize_and_clean, lemmatize_tokens, save_text_to_file
from data_processing.tfidf import compute_tfidf, levenshtein_distance
from data_processing.visualization import plot_tfidf

# Path to your PDF
pdf_path = 'data/PMI_RM standard.pdf'

if __name__ == "__main__":
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Save the extracted text into the output_normal folder
    saved_file_path = save_text_to_file(text)

    # Tokenization, cleaning, and lemmatization
    tokens = tokenize_and_clean(text)
    lemmatized_tokens = lemmatize_tokens(tokens)

    # Compute TF-IDF
    term_scores = compute_tfidf(lemmatized_tokens)
    
    # Visualize the top 20 TF-IDF terms
    plot_tfidf(term_scores)

    # Example Levenshtein distance
    word1, word2 = 'management', 'risks'
    lev_distance = levenshtein_distance(word1, word2)
    print(f"Levenshtein distance between '{word1}' and '{word2}': {lev_distance}")
    
    # Show where the extracted text was saved
    print(f"Extracted text has been saved to: {saved_file_path}")
