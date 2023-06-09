import os
import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer
import torch
import json

class DocumentSearcher:
    def __init__(self, model_name='all-MiniLM-L6-v2', api_key='f6ff4ba3-0003-4171-a3f5-8504143e990f', 
                 environment='northamerica-northeast1-gcp', index_name='document-search'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    def search(self, query, top_k=5):
        xq = self.model.encode(query).tolist()
        results = self.index.query(queries=[xq], top_k=top_k, include_metadata=True)

        ids = []
        scores = []
        metadatas = []

        for result in results.results[0].matches:
            ids.append(result.id)
            scores.append(result.score)
            metadatas.append(result.metadata)

        df = pd.DataFrame({
            'ID': ids,
            'Score': scores,
            'Metadata': metadatas
        })

        metadata_df = pd.json_normalize(df['Metadata'])
        df = pd.concat([df.drop('Metadata', axis=1), metadata_df], axis=1)

        return df
