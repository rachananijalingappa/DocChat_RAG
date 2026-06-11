import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

# Embeddings Configuration
# sentence-transformers runs locally and is free
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# RAG Configuration
RETRIEVER_K = 3  # Number of chunks to retrieve
