import json
from langchain_core.language_models import LLM
from typing import Any, List, Optional
from pydantic import Field
from adaptive_rag import AdaptiveRAGSystem
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

class AirLLMLangChain(LLM):
    wrapper: Any = Field(default=None)

    @property
    def _llm_type(self) -> str:
        return "airllm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.wrapper.generate(prompt)

def run_evaluation(num_samples=5):
    # Initialize system
    rag_system = AdaptiveRAGSystem()
    
    # Load test set
    with open("test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    questions = test_data["sorular"][:num_samples]
    
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print(f"Evaluating {num_samples} samples...")
    for q in questions:
        query = q["instruction"]
        # Get retrieval results for context recall
        docs = rag_system.hybrid_retrieve(query)
        context_texts = [d["text"] for d in docs]
        
        # Get system answer
        answer = rag_system.run(query)
        
        eval_data["question"].append(query)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(context_texts)
        eval_data["ground_truth"].append(q["ground_truth"])
    
    # Convert to RAGAS dataset
    dataset = Dataset.from_dict(eval_data)
    
    # In a real scenario, we'd pass the LLM to RAGAS metrics
    # But RAGAS 0.4+ expects specific LLM wrappers. 
    # For now, we'll just output the collected data.
    print("\n--- Evaluation Data Collected ---")
    for i in range(len(eval_data["question"])):
        print(f"Q: {eval_data['question'][i]}")
        print(f"A: {eval_data['answer'][i]}")
        print(f"GT: {eval_data['ground_truth'][i]}")
        print("-" * 20)
    
    # Note: RAGAS metrics calculation requires a judge LLM.
    # To run this locally with AirLLM, one would need to configure:
    # result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall], llm=...)
    # print(result)

if __name__ == "__main__":
    # We use a small number of samples because AirLLM is slow
    run_evaluation(num_samples=3)
