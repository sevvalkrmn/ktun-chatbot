import os
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from airllm_wrapper import QwenAirLLMWrapper

class AdaptiveRAGSystem:
    def __init__(self, index_dir="rag_index", model_id="Qwen/Qwen2.5-32B-Instruct"):
        self.index_dir = index_dir
        self.model_id = model_id
        
        print("Loading RAG components...")
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(index_dir, "docs.pkl"), "rb") as f:
            self.docs = pickle.load(f)
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        
        print("Loading embedding model (CPU)...")
        self.embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
        
        print("Loading reranker model (CPU)...")
        # BGE-Reranker-v2-m3 is excellent for multilingual tasks
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
        
        # Initialize AirLLM
        self.llm = QwenAirLLMWrapper(model_id=self.model_id)

    def hybrid_retrieve(self, query, top_k=5, rerank_k=15):
        # 1. BM25 Retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. FAISS Retrieval
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        faiss_distances, faiss_indices = self.faiss_index.search(query_embedding.astype("float32"), rerank_k)
        
        # 3. Reciprocal Rank Fusion for initial candidates
        combined_results = {}
        bm25_indices = np.argsort(bm25_scores)[::-1][:rerank_k]
        for rank, idx in enumerate(bm25_indices):
            combined_results[idx] = combined_results.get(idx, 0) + 1 / (rank + 60)
        for rank, idx in enumerate(faiss_indices[0]):
            combined_results[idx] = combined_results.get(idx, 0) + 1 / (rank + 60)
            
        initial_candidates_idx = sorted(combined_results.keys(), key=lambda x: combined_results[x], reverse=True)[:rerank_k]
        initial_docs = [self.docs[i] for i in initial_candidates_idx]
        
        # 4. Cross-Encoder Reranking
        print(f"Reranking {len(initial_docs)} candidates...")
        pairs = [[query, doc['text']] for doc in initial_docs]
        rerank_scores = self.reranker.predict(pairs)
        
        reranked_docs = [doc for _, doc in sorted(zip(rerank_scores, initial_docs), key=lambda x: x[0], reverse=True)]
        
        return reranked_docs[:top_k]

    def analyze_query(self, query):
        prompt = f"""
Sorguyu analiz et ve aşağıdaki JSON formatında yanıt ver:
{{
  "intent": "Ders içeriği | Genel Bilgi | Kampüs | Diğer",
  "complexity": "Basit | Orta | Karmaşık",
  "needs_retrieval": true/false,
  "multi_hop": true/false
}}

Sorgu: {query}
JSON:"""
        # Note: In real Adaptive RAG, we'd use a faster model for this.
        # But per user request, we use the 32B model.
        analysis_raw = self.llm.generate(prompt, max_new_tokens=50)
        try:
            # Simple cleanup for JSON parsing
            json_str = analysis_raw.strip()
            if "{" in json_str and "}" in json_str:
                json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]
            import json
            return json.loads(json_str)
        except:
            return {"intent": "Genel Bilgi", "complexity": "Orta", "needs_retrieval": True, "multi_hop": False}

    def generate_grounded_answer(self, query, docs):
        context = "\n---\n".join([d['text'] for d in docs])
        prompt = f"""
Aşağıdaki dökümanları kullanarak soruya yanıt ver. Eğer dökümanlarda bilgi yoksa "Bilmiyorum" de.
Dökümanlar:
{context}

Soru: {query}
Yanıt (Alıntı yaparak):"""
        return self.llm.generate(prompt, max_new_tokens=512)

    def critique_answer(self, query, answer, docs):
        context = "\n---\n".join([d['text'] for d in docs])
        prompt = f"""
Verilen yanıtın dökümanlar tarafından desteklenip desteklenmediğini kontrol et.
Dökümanlar:
{context}

Soru: {query}
Yanıt: {answer}

Yanıt dökümanlarla tutarlı mı? (EVET/HAYIR)
Neden:"""
        critique = self.llm.generate(prompt, max_new_tokens=100)
        return "EVET" in critique.upper()

    def rewrite_query(self, query, critique_reason):
        prompt = f"""
Soru için verilen yanıt yetersiz bulundu. Sorguyu daha iyi sonuç alabilmek için yeniden düzenle.
Soru: {query}
Neden yetersiz: {critique_reason}

Yeniden düzenlenmiş sorgu:"""
        return self.llm.generate(prompt, max_new_tokens=50)

    def run(self, query):
        print(f"\n--- İşleniyor: {query} ---")
        
        # 1. Analyze
        print("Sorgu analiz ediliyor...")
        analysis = self.analyze_query(query)
        print(f"Analiz: {analysis}")
        
        if not analysis.get("needs_retrieval", True):
            return self.llm.generate(query)

        # 2. First Retrieval & Generation
        print("Dökümanlar getiriliyor (1. Aşama)...")
        retrieved_docs = self.hybrid_retrieve(query)
        print("Yanıt oluşturuluyor...")
        answer = self.generate_grounded_answer(query, retrieved_docs)
        
        # 3. Critique & Verify
        print("Yanıt doğrulanıyor...")
        is_faithful = self.critique_answer(query, answer, retrieved_docs)
        
        if not is_faithful:
            print("Uyarı: Yanıt güvenilir bulunmadı. Re-retrieval (Yeniden Arama) başlatılıyor...")
            # 4. Rewrite & Re-retrieve
            improved_query = self.rewrite_query(query, "Yanıt dökümanlarda tam desteklenmiyor.")
            print(f"Yeni Sorgu: {improved_query}")
            
            extra_docs = self.hybrid_retrieve(improved_query)
            # Merge and Filter docs (Context Filter)
            all_docs = {d['id']: d for d in (retrieved_docs + extra_docs)}.values()
            all_docs = list(all_docs)[:7] # Keep top 7 unique docs
            
            print("Yanıt yeniden oluşturuluyor...")
            answer = self.generate_grounded_answer(query, all_docs)
            
        return answer

if __name__ == "__main__":
    # Test with ONE question as requested
    import json
    
    try:
        with open("test_set.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        # Taking the first question
        test_q = test_data["sorular"][0]
        query = test_q["instruction"]
        ground_truth = test_q["ground_truth"]
        
        print(f"Sistem Başlatılıyor...")
        system = AdaptiveRAGSystem()
        
        print(f"\nSoru: {query}")
        print(f"Beklenen (Ground Truth): {ground_truth}")
        
        result = system.run(query)
        
        print("\n" + "="*30)
        print("SİSTEM YANITI:")
        print(result)
        print("="*30)
        
    except FileNotFoundError:
        print("Hata: test_set.json veya rag_index bulunamadı. Lütfen önce indexer.py çalıştırın.")
