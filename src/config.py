import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embeddings Configuration
# sentence-transformers runs locally and is free
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# RAG Configuration
RETRIEVER_K = 3  # Number of chunks to retrieve
