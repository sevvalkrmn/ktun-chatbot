# KTÜN Computer Engineering Chatbot

A Turkish academic chatbot for Konya Technical University (KTÜN) Computer Engineering students. The system answers questions about courses, internships, lateral transfer, Erasmus, and other academic topics — running entirely locally without internet or cloud dependency.

> **Advisor:** Doç. Dr. Sait Ali Uymaz  
> **Institution:** Konya Technical University — Computer Engineering Department

---

## Overview

This project benchmarks multiple RAG (Retrieval-Augmented Generation) architectures on the same Turkish domain-specific dataset. Each team member implements a different retrieval strategy; all architectures are evaluated with RAGAS metrics under identical conditions.

**Stack**
- LLM: Qwen3.5-9B / Qwen2.5-7B (auto-selected based on available VRAM)
- Embedding: `paraphrase-multilingual-MiniLM-L12-v2`
- Orchestration: LangGraph
- Evaluation: RAGAS

---

## Repository Structure

```
ktun-chatbot/
├── data/
│   ├── dataset_chatbot_v2/    # Source JSON files by category
│   ├── ktun_dataset.json      # Merged dataset (255 entries)
│   └── test_set.json          # RAGAS test set — DO NOT MODIFY
├── architectures/
│   └── hybrid_rag/            # Baseline implementation
├── agents/                    # Shared LangGraph agent system
├── evaluation/
│   ├── evaluate.py            # Shared RAGAS evaluation script
│   └── results/               # Per-architecture results
├── config.py                  # Shared configuration
├── main.py
└── requirements.txt
```

---

## Getting Started

```bash
git clone https://github.com/sevvalkrmn/ktun-chatbot.git
cd ktun-chatbot
git checkout feature/<your-branch>

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

pip install -r requirements.txt
python main.py
```

---

## Dataset Schema

```json
{
  "id": "staj_sure_kosullar_001",
  "category": "staj",
  "subcategory": "sure_ve_kosullar",
  "canonical_question": "Zorunlu staj süresi ne kadardır?",
  "answer_short": "Toplam 40 iş günüdür.",
  "answer_full": "Zorunlu staj, her biri 20 iş günü olmak üzere iki oturumda yapılır...",
  "question_variants": ["Staj kaç gün?", "Staj ne kadar sürer?"],
  "keywords": ["staj", "40 iş günü"],
  "source": "ktun_bm_staj_yonergesi_2024",
  "verified": false
}
```

**255 entries** across 8 categories: `bolum_dersleri`, `staj`, `yatay_gecis`, `erasmus`, `ingilizce_muafiyet`, `genel_universite`, `topluluk`, `yemekhane`

---

## Adding a New Architecture

Create your folder under `architectures/` and implement the standard interface:

```python
# architectures/your_arch/retriever.py

def search(self, query: str) -> dict:
    return {
        "top_answer": str,    # Best answer text
        "top_score": float,   # Confidence score 0-1
        "top_results": list   # Top-k results
    }
```

Always use shared config:

```python
from config import DATA_PATH, MODEL_NAME, EMBEDDING_MODEL, HIGH_CONFIDENCE, MID_CONFIDENCE
```

---

## Evaluation

```bash
python -m evaluation.evaluate --architecture <arch_name> --n 100
```

Results are saved to `evaluation/results/<arch_name>_results.json`.

Metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`

---

## Contribution Rules

- `data/test_set.json` — shared test set, never modify
- `config.py` — do not change `MODEL_NAME` or `EMBEDDING_MODEL`
- Only write to your own `architectures/<your_arch>/` folder
- Commit messages must be in English: `<scope>: <description>`

```bash
# Daily workflow
git pull origin master
git checkout feature/<your-branch>

# ... work ...

git add architectures/<your_arch>/
git commit -m "your_arch: describe what you did"
git push origin feature/<your-branch>
```

---

## License

MIT License
