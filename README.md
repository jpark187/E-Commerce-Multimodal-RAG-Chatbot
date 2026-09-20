# Multimodal RAG Product Assistant
A multimodal Retrieval-Augmented Generation (RAG) system that answers natural-language questions about Amazon products using both **text and image understanding**. The system embeds product data with CLIP, retrieves relevant products via ChromaDB with cross-encoder reranking, and generates grounded answers with an LLM — with guardrails for PII redaction and hallucination detection.

## Overview
Given a question like *"What are good wireless earbuds with noise cancellation?"* — or an uploaded photo of a product — the system retrieves the most relevant products from an embedded product catalog and generates a natural-language answer grounded in that retrieved context, citing product names, prices, and specs rather than fabricating them.

## Architecture
**1. Data preparation**
Raw Amazon product data is cleaned, validated, and normalized into structured product records (name, category, price, features, specs, dimensions, etc.), with a quality gate requiring either a real description or a valid product image.

**2. Multimodal embedding**
Each product is embedded using **CLIP** (`openai/clip-vit-base-patch32`), combining a text embedding of the product description with an image embedding of the product photo (averaged into a single multimodal vector). This lets the system retrieve products from either a text query, an image query, or both.

**3. Retrieval**
Embeddings are stored and queried via **ChromaDB** (cosine similarity, top-20 candidates), then reranked using a **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) against the raw query text for higher-precision top-k results.

**4. Generation**
An open-source LLM (Meta-Llama-3-8B-Instruct, via HuggingFace's Inference API) generates the final answer, using a prompt constrained to only the retrieved product context — with few-shot examples to guide tone and format, and per-session chat memory via LangChain.

**5. Responsible AI guardrails**
- **PII redaction** (Presidio) scrubs sensitive info from user queries before processing.
- **Topic filtering** blocks off-topic/inappropriate queries.
- **Groundedness scoring** — a lightweight n-gram overlap check between the generated answer and retrieved context, flagging potentially hallucinated answers.

**6. Interface**
A Streamlit chat interface (`app.py`) supports text queries, image uploads, and combined text+image queries, with per-session conversation memory.

## Tech Stack
- **Embeddings:** CLIP (ViT-B/32)
- **Vector store:** ChromaDB
- **Reranking:** Sentence-Transformers cross-encoder
- **LLM:** Meta-Llama-3-8B-Instruct (HuggingFace Inference API)
- **Orchestration:** LangChain
- **Guardrails:** Presidio (PII detection/anonymization)
- **UI:** Streamlit
- **Evaluation:** RAGAS (faithfulness, answer relevancy, context precision/recall), custom Recall@K

## Project Structure
├── app.py # Streamlit chat interface
├── rag_utils.py # Core RAG pipeline (retrieval, generation, guardrails)
├── amazon_chroma_db/ # Pre-built vector index (CLIP embeddings of ~1,000 products)
├── notebooks/
│ └── data_pipeline_and_eval.ipynb # Full data cleaning, embedding, and evaluation pipeline
├── requirements.txt
└── .gitignore


## Evaluation
The system was evaluated on a hand-built test set spanning factual, semantic, multi-hop, and out-of-scope queries, using:
- **RAGAS** metrics (faithfulness, answer relevancy, context precision/recall) with GPT-3.5 as an LLM judge
- **Recall@K** (K = 1, 5, 10) against known-relevant products
- A custom lightweight groundedness heuristic for real-time hallucination flagging

Full evaluation code and results are in the notebook under `/notebooks`.

## Running Locally
```bash
git clone https://github.com/jpark187/E-Commerce-Multimodal-RAG-Chatbot.git
cd amazon-rag-assistant
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "your_huggingface_token_here"
```

Then run:
```bash
streamlit run app.py
```

> Note: `meta-llama/Meta-Llama-3-8B-Instruct` requires requesting access on HuggingFace. `HuggingFaceH4/zephyr-7b-beta` is a drop-in ungated alternative for testing.

## Known Limitations
- **Embedding fusion is a simple average of text/image vectors** — a more sophisticated cross-modal fusion (e.g., learned weighting) would likely improve retrieval precision.
- **Recall@K evaluation uses a small (5-query) test set** — useful as a smoke test, not a statistically rigorous benchmark.
- **Groundedness check is a lightweight n-gram heuristic**, not a semantic entailment model — it catches obvious hallucinations but isn't exhaustive.
- **No live-hosted demo** — the vector index (`amazon_chroma_db/`) is pre-built and included in this repo; running the app requires a HuggingFace API token.

## Dataset
[Amazon Product Dataset 2020](https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020) — a sample of ~1,000 products was used for embedding and indexing.
