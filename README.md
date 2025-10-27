
## README — Problem statement

**Problem statement**: Build a reliable agent that can answer user questions by combining a local open-source LLM for reasoning and external tools exposed as HTTP APIs (for up-to-date facts). The agent should be orchestrated with LangGraph to support tool-calls and durable orchestration and evaluated with DeepEval to measure correctness, hallucination, and retrieval accuracy.

**Why this project**: Many production workloads require a mix of local models (for privacy and cost), retrieval, and tools (for live data). The stack below is designed to be reproducible and easy to swap components.

---

## Tools used

- **LangGraph** — low-level orchestration for building stateful agents and durable executions. We use it to wire the reasoning steps and tool calls.
- **LLM Model** —deepseek-ai/DeepSeek-R1-Distill-Qwen-7B[https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B] huggingface model. Best local reasoning model. Handles CoT, math, logic very well.
- **Requests / httpx** — to call external HTTP APIs (tools).
- **DeepEval** — evaluation framework for testing LLM outputs with unit-test style benchmarks.

---

## Why these models

- **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (open-source, Apache-2.0)**: excellent performance-to-size tradeoff. It is known to outperform many larger models on text benchmarks while remaining small enough to run on commodity GPUs or inference services. See [https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B] release notes and Hugging Face model card.

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