import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import EMBEDDING_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def create_in_memory_vector_store(chunks):
    """
    Takes document chunks and creates an ephemeral (in-memory) Chroma vector store.

    Each call gets its own uniquely named collection. Without a name, Chroma reuses
    its default collection, so chunks from earlier uploads stayed searchable after
    a new document was uploaded.
    """
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"doc_{uuid.uuid4().hex}",
    )
    return vector_store
