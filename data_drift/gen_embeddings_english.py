import pandas as pd
df = pd.read_parquet('preprocessed_dataset.parquet')

reference_texts = []
for i in range(43383):
    reference_texts.append(df['abuse_free_complaint'][i])

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2', device = device)

# Generate embeddings for reference texts
ref_embeddings = model.encode(reference_texts, show_progress_bar=True)

import pickle

# Save embeddings to a pickle file
def save_embeddings_to_pickle(embeddings, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(embeddings, file)
    print(f"Embeddings successfully saved to {file_name}")

# Load embeddings from a pickle file
def load_embeddings_from_pickle(file_name):
    with open(file_name, 'rb') as file:
        embeddings = pickle.load(file)
    print(f"Embeddings successfully loaded from {file_name}")
    return embeddings

# Example usage
pickle_file_name = "ref_embeddings_english.pkl"

# Save the baseline embeddings
save_embeddings_to_pickle(ref_embeddings, pickle_file_name)
