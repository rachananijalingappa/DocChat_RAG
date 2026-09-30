---
title: DocChat
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# DocChat: Multi-Agent RAG Service

DocChat is a decoupled RAG application that uses LangGraph to route queries between local document retrieval and live web search. 

## Architecture Overview

The backend is built with FastAPI and operates as a state graph:
- **Router:** Evaluates query intent using GPT-4o-mini.
- **RAG Node:** Embeds documents into ChromaDB and applies Cohere contextual re-ranking to the retrieved chunks.
- **Web Node:** Falls back to DuckDuckGo search for queries unrelated to the uploaded document.
- **Evaluator:** An LLM-as-a-judge scores the final response for faithfulness and relevance to flag hallucinations.
- **Guardrails:** Pre-processing layer to block prompt injections before pipeline execution.

## Tech Stack
- **Backend:** FastAPI, Python
- **Frontend:** Streamlit
- **Orchestration:** LangGraph
- **LLMs:** OpenAI (GPT-4o-mini)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector DB:** ChromaDB (in-memory)
- **Re-ranking:** Cohere (`rerank-english-v3.0`)

## Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with the required API keys:
   ```env
   OPENAI_API_KEY=your_openai_key
   COHERE_API_KEY=your_cohere_key
   ```
3. Start the FastAPI server:
   ```bash
   python -m uvicorn api:app --port 8000
   ```
4. Start the Streamlit frontend (in a separate terminal):
   ```bash
   streamlit run app.py
   ```
