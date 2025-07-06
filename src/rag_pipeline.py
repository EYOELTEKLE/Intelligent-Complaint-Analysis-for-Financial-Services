import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from transformers import pipeline

class ComplaintRetriever:
    def __init__(self, index_path: str, meta_path: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', top_k: int = 5):
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k

    def embed_query(self, query: str) -> np.ndarray:
        return np.array(self.model.encode([query], normalize_embeddings=True)).astype('float32')

    def retrieve(self, query: str) -> List[Dict]:
        query_vec = self.embed_query(query)
        D, I = self.index.search(query_vec, self.top_k)
        results = []
        for idx in I[0]:
            meta = self.metadata[idx]
            # Try to get the actual chunk text from metadata, fallback to cleaned_narrative, else empty string
            chunk_text = meta.get('chunk_text') or meta.get('cleaned_narrative') or ""
            meta['retrieved_text'] = chunk_text
            results.append(meta)
        return results

class PromptEngineer:
    def __init__(self, template: str = None):
        self.template = template or (
            """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. "
            "Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, "
            "state that you don't have enough information.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""
        )

    def build_prompt(self, context_chunks: List[str], question: str) -> str:
        # Ensure all context_chunks are strings and not None
        context_chunks = [c if isinstance(c, str) and c else "" for c in context_chunks]
        context = "\n---\n".join(context_chunks)
        return self.template.format(context=context, question=question)

class ComplaintGenerator:
    def __init__(self, model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2', device: int = -1):
        # device: -1 for CPU, 0+ for GPU
        self.generator = pipeline("text-generation", model=model_name, device=device, max_new_tokens=256)

    def generate(self, prompt: str) -> str:
        output = self.generator(prompt, do_sample=True, temperature=0.7)
        return output[0]['generated_text'].split('Answer:')[-1].strip()

class RAGPipeline:
    def __init__(self, retriever: ComplaintRetriever, prompt_engineer: PromptEngineer, generator: ComplaintGenerator):
        self.retriever = retriever
        self.prompt_engineer = prompt_engineer
        self.generator = generator

    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        retrieved = self.retriever.retrieve(question)
        context_chunks = [meta.get('retrieved_text', '') for meta in retrieved]
        prompt = self.prompt_engineer.build_prompt(context_chunks, question)
        answer = self.generator.generate(prompt)
        return answer, context_chunks

# --- Example usage ---
if __name__ == '__main__':
    # Update these paths as needed
    print(os.getcwd())
    index_path = 'vector_store/faiss.index'
    meta_path = 'vector_store/metadata.json'
    retriever = ComplaintRetriever(index_path, meta_path)
    prompt_engineer = PromptEngineer()
    generator = ComplaintGenerator()  # You may need to adjust model_name/device
    rag = RAGPipeline(retriever, prompt_engineer, generator)

    # Example question
    question = "How do customers feel about credit card late fees?"
    answer, sources = rag.answer_question(question)
    print(f"Q: {question}\nA: {answer}\n\nTop retrieved sources:\n")
    for i, src in enumerate(sources[:2]):
        print(f"Source {i+1}: {src}\n")
