import pandas as pd
df = pd.read_parquet('preprocessed_dataset.parquet')

reference_texts = []
for i in range(43382):
    reference_texts.append(df['abuse_free_complaint_hindi'][i])

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm  # Progress bar

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the model and send it to the GPU
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = model.to(device)

# Generate embeddings in batches with progress monitoring
def encode_with_progress(data, batch_size=8):
    embeddings = []
    for i in tqdm(range(0, len(data), batch_size), desc="Encoding batches"):
        batch = data[i:i + batch_size]
        batch_embeddings = model.encode(batch, device=device)
        embeddings.extend(batch_embeddings)
    return embeddings

# Encode baseline and new data
print("Encoding baseline data...")
baseline_embeddings = encode_with_progress(reference_texts)

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
pickle_file_name = "ref_embeddings_hindi.pkl"

# Save the baseline embeddings
save_embeddings_to_pickle(baseline_embeddings, pickle_file_name)
