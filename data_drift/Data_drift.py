%%time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from google.cloud import storage
import requests
import pickle

# Public URL of the reference embeddings file
public_url = "https://storage.googleapis.com/ref_embeddings/reference_embeddings.pkl"
# Download the file
response = requests.get(public_url)
if response.status_code == 200:
    with open("reference_embeddings.pkl", "wb") as f:
        f.write(response.content)
    print("Reference embeddings downloaded successfully.")
else:
    print(f"Failed to download file. HTTP Status Code: {response.status_code}")

# Load the embeddings
with open("reference_embeddings.pkl", "rb") as f:
    ref_embeddings = pickle.load(f)
print("Reference embeddings loaded.")

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Current text to compare
current_text = ["I have a complain with my dentist. he borrowed momey from me and never returned."]

# Generate embedding for the current text
current_embeddings = model.encode(current_text, show_progress_bar=True)

import numpy as np
# Compute cosine similarities
cos_similarities = cosine_similarity(ref_embeddings, current_embeddings)

# Find the maximum similarity and corresponding reference text
max_cos_sim_index = np.argmax(cos_similarities)
max_cos_sim = cos_similarities[max_cos_sim_index][0]

# Drift detection threshold
cosine_threshold = 0.55

if max_cos_sim < cosine_threshold:
    print("Drift detected!")
else:
    print("No drift detected!")

# Display the most similar reference text
print(f"Max Cosine Similarity: {max_cos_sim:.4f}")
