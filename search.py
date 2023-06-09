import os
import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer
import torch
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

pinecone.init(api_key='f6ff4ba3-0003-4171-a3f5-8504143e990f', environment='northamerica-northeast1-gcp')

index_name = 'document-search'
index = pinecone.Index(index_name)

query = "autogpt"
xq = model.encode(query).tolist()

results = index.query(queries=[xq], top_k=5, include_metadata=True)

# Initialize lists to store results
ids = []
scores = []
metadatas = []

# Loop over the results and append to lists
for result in results.results[0].matches:
    ids.append(result.id)
    scores.append(result.score)
    metadatas.append(result.metadata)

# Create a dataframe from the lists
df = pd.DataFrame({
    'ID': ids,
    'Score': scores,
    'Metadata': metadatas
})

# Expand the metadata dictionary into separate columns
metadata_df = pd.json_normalize(df['Metadata'])

# Concatenate the original dataframe with the metadata dataframe
df = pd.concat([df.drop('Metadata', axis=1), metadata_df], axis=1)
df.to_csv('results.csv',index=False)

print(df['Extracted_Text'].to_list())
