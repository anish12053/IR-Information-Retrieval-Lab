import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

def preprocess_text(text):
    # Tokenization and converting to lowercase
    tokens = text.lower().split()
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the dataset
preprocessed_data = [preprocess_text(text) for text in newsgroups.data]

# Print a preprocessed document as an example
print(preprocessed_data[0])


# Create a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed data
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_data)

# Print the shape of the term-document matrix
print("Shape of the term-document matrix:", tfidf_matrix.shape)


# Define the number of topics (components) to retain
num_topics = 100

# Apply SVD to the term-document matrix
svd = TruncatedSVD(n_components=num_topics)
lsa_matrix = svd.fit_transform(tfidf_matrix)
# Print the shape of the LSA matrix
print("Shape of the LSA matrix:", lsa_matrix.shape)


# Get singular vectors and terms
singular_vectors = svd.components_
feature_names = tfidf_vectorizer.get_feature_names_out()

# Find the indices of the top singular values
top_topics = [i for i in range(num_topics)]
topics_to_analyze = [0, 1, 2, 3, 4]
# Analyze the topics and their top terms
for topic_num in topics_to_analyze:
    # Get the top terms for this topic
    top_term_indices = singular_vectors[topic_num].argsort()[::-1][:10]
    top_terms = [feature_names[i] for i in top_term_indices]
    
    print(f"Topic {topic_num}:")
    print(", ".join(top_terms))
    print()


query = ["Hello world"]  # Place the query text inside a list

# Preprocess the query
query_tfidf = tfidf_vectorizer.transform(query)

# Project the query into LSI space using the singular vectors
query_lsi = svd.transform(query_tfidf)

# Compute cosine similarity between the query and all documents
cosine_similarities = cosine_similarity(query_lsi, lsa_matrix)  # Use lsa_matrix, which contains LSI-transformed documents

# Sort documents by similarity in descending order
most_similar_indices = cosine_similarities[0].argsort()[::-1]

# Print the most relevant documents
top_k = 10  # You can adjust the number of top documents you want to retrieve
for i, idx in enumerate(most_similar_indices[:top_k]):
    print(f"Top-{i + 1} Document:")
    print(newsgroups.data[idx])  # Print the content of the document
    print(f"Cosine Similarity: {cosine_similarities[0][idx]}")
    print()


# Clustering using K-Means and Evaluation
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
# Apply K-Means clustering to the LSI-transformed data
n_clusters = 20  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
predicted_labels = kmeans.fit_predict(lsa_matrix)
silhouette_avg = silhouette_score(lsa_matrix, predicted_labels)

print(f"Silhouette Score:, {silhouette_avg:.4f}")
# Calculate Normalized Mutual Information (NMI) with the true labels (true_labels) and predicted cluster assignments
nmi = normalized_mutual_info_score(true_labels, predicted_labels)

print(f"Normalized Mutual Information (NMI):, {nmi:.4f}")

