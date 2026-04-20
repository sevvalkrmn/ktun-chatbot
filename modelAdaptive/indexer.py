import json
import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

class AdaptiveRAGIndexer:
    def __init__(self, data_path="CategorizedGeneralQuestions.json", 
                 model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 index_dir="rag_index"):
        self.data_path = data_path
        self.model_name = model_name
        self.index_dir = index_dir
        self.docs = []
        self.bm25 = None
        self.index = None
        self.model = None

        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Combine instruction and output for better retrieval
        self.docs = [
            {
                "id": d["id"],
                "text": f"Soru: {d['instruction']}\nCevap: {d['output']}",
                "category": d["category"],
                "difficulty": d["difficulty"]
            }
            for d in data
        ]
        print(f"Loaded {len(self.docs)} documents.")

    def build_bm25(self):
        print("Building BM25 index...")
        tokenized_docs = [doc["text"].lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        with open(os.path.join(self.index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)
        print("BM25 index built.")

    def build_faiss(self):
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        print("Generating embeddings (this may take a few minutes)...")
        texts = [doc["text"] for doc in self.docs]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))
        
        faiss.write_index(self.index, os.path.join(self.index_dir, "faiss.index"))
        
        # Save docs for later retrieval
        with open(os.path.join(self.index_dir, "docs.pkl"), "wb") as f:
            pickle.dump(self.docs, f)
        print("FAISS index built.")

    def run(self):
        self.load_data()
        self.build_bm25()
        self.build_faiss()

if __name__ == "__main__":
    indexer = AdaptiveRAGIndexer()
    indexer.run()
