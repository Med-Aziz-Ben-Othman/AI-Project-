# data_processing/mesures.py

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.metrics import edit_distance

# Compute TF-IDF
def compute_tfidf(tokens):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(tokens)])
    terms = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    
    # Associate each term with its score
    term_scores = list(zip(terms, scores))
    term_scores = sorted(term_scores, key=lambda x: x[1], reverse=True)
    
    return term_scores

# Compute Levenshtein distance
def levenshtein_distance(word1, word2):
    return edit_distance(word1, word2)

