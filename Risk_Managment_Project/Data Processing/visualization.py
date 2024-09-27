# data_processing/visualization.py

import matplotlib.pyplot as plt
from collections import Counter

# Plot POS counts
def plot_pos_counts(doc):
    pos_counts = Counter(token.pos_ for token in doc[:50])
    labels, counts = zip(*pos_counts.items())

    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Part of Speech')
    plt.ylabel('Count')
    plt.title('Count of Tokens by Part of Speech')
    plt.xticks(rotation=45)
    plt.show()

# Plot most frequent words
def plot_most_common_words(freq_dist):
    most_common_words = freq_dist.most_common(10)
    df_most_common = pd.DataFrame(most_common_words, columns=['Mot', 'Fréquence'])
    words = [word for word, freq in most_common_words]
    frequencies = [freq for word, freq in most_common_words]
    
    plt.figure(figsize=(7, 7))
    plt.pie(frequencies, labels=words, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Top 10 des mots les plus fréquents')
    plt.show()

# Plot TF-IDF results
def plot_tfidf(term_scores):
    terms = [term for term, _ in term_scores[:20]]
    scores = [score for _, score in term_scores[:20]]

    plt.barh(terms, scores)
    plt.xlabel('Score')
    plt.ylabel('Concepts')
    plt.title('Concepts les plus pertinents')
    plt.show()
