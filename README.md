
## README — Problem statement

**Problem statement**: Build a reliable agent that can answer user questions by combining a local open-source LLM for reasoning, a semantic retriever (embeddings + vector DB), and external tools exposed as HTTP APIs (for up-to-date facts). The agent should be orchestrated with LangGraph to support tool-calls and durable orchestration and evaluated with DeepEval to measure correctness, hallucination, and retrieval accuracy.

**Why this project**: Many production workloads require a mix of local models (for privacy and cost), retrieval, and tools (for live data). The stack below is designed to be reproducible and easy to swap components.

---

## Tools used

- **LangGraph** — low-level orchestration for building stateful agents and durable executions. We use it to wire the reasoning steps and tool calls.
- **Transformers / accelerate** — to load and run Mistral-7B (or another model) locally or via Hugging Face Inference.
- **sentence-transformers** (`all-MiniLM-L6-v2`) — compact, fast embeddings for semantic search.
- **Chroma / FAISS (optional)** — vector store for retrieval (examples will show Chroma usage).
- **Requests / httpx** — to call external HTTP APIs (tools).
- **DeepEval** — evaluation framework for testing LLM outputs with unit-test style benchmarks.

---

## Why these models

- **Mistral-7B (open-source, Apache-2.0)**: excellent performance-to-size tradeoff. It is known to outperform many larger models on text benchmarks while remaining small enough to run on commodity GPUs or inference services. See Mistral release notes and Hugging Face model card.

- **`all-MiniLM-L6-v2`** embeddings: small (fast, low memory), accurate for semantic search and clustering for many applications. Good default when latency and cost matter.

(References: Mistral announcement on Mistral.ai, Hugging Face model card, sentence-transformers docs.)

---




## ⚙️ Setup Instructions

### 1️⃣ Prerequisites
- Python 3.12+
- (Optional) [uv](https://github.com/astral-sh/uv) for fast environment setup.

### 2️⃣ Clone the repo
```bash
git clone https://github.com/dandesunil/ChatApp.git
cd ChatApp
```

### 3️⃣ Create and activate environment
Using **uv**:
```bash
uv venv
source .venv/bin/activate
```
Or using traditional Python:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 4️⃣ Install dependencies
```bash
uv sync
```

### 5️⃣ Run the application
```bash
uvicorn main:app --reload
```

The app will be available at:  
👉 **http://localhost:8000**