import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from typing import List, Dict

# --- Text Chunker ---
class TextChunker:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i + self.chunk_size]
            if not chunk:
                break
            chunks.append(' '.join(chunk))
            i += self.chunk_size - self.chunk_overlap
        return chunks

# --- Embedder ---
class Embedder:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True))

# --- Vector Indexer ---
class VectorIndexer:
    def __init__(self, dim: int, store_dir: str = '../vector_store'):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []  # List of dicts
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadatas)

    def save(self, index_name: str = 'faiss.index', meta_name: str = 'metadata.json'):
        faiss.write_index(self.index, os.path.join(self.store_dir, index_name))
        with open(os.path.join(self.store_dir, meta_name), 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

# --- Main Pipeline ---
def main():
    # Params
    chunk_size = 256
    chunk_overlap = 32
    embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
    
    data_path = 'data/processed/filtered_complaints.csv'
    store_dir = '../vector_store'

    # Load data
    df = pd.read_csv(data_path)
    if 'cleaned_narrative' not in df.columns:
        raise ValueError('Expected "cleaned_narrative" column in input CSV.')

    chunker = TextChunker(chunk_size, chunk_overlap)
    embedder = Embedder(embedding_model)

    all_chunks = []
    all_metadata = []
    for idx, row in df.iterrows():
        complaint_id = row.get('Complaint ID', idx)
        product = row.get('Product', None)
        narrative = row['cleaned_narrative']
        chunks = chunker.chunk_text(narrative)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                'complaint_id': complaint_id,
                'product': product,
                'chunk_index': chunk_idx,
                'original_length': len(narrative.split()),
            })

    print(f'Total text chunks: {len(all_chunks)}')
    # Embed
    embeddings = embedder.embed(all_chunks)
    print(f'Embeddings shape: {embeddings.shape}')
    # Index
    indexer = VectorIndexer(dim=embeddings.shape[1], store_dir=store_dir)
    indexer.add(embeddings, all_metadata)
    indexer.save()
    print(f'Vector store and metadata saved to {store_dir}')

if __name__ == '__main__':
    main()
