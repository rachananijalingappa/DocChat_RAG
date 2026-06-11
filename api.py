from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
import tempfile
import logging

logger = logging.getLogger(__name__)
from src.document_processor import process_pdf
from src.vector_store import create_in_memory_vector_store
from src.rag_engine import setup_rag_pipeline, ask_rag_question
from src.evaluator import evaluate_response

app = FastAPI(title="DocChat RAG Microservice")

class AppState:
    def __init__(self):
        self.vector_store = None
        self.rag_chain = None

state = AppState()

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        class UploadedFileAdapter:
            def __init__(self, path):
                self.path = path
            def getvalue(self):
                with open(self.path, 'rb') as f:
                    return f.read()

        adapted_file = UploadedFileAdapter(tmp_path)
        chunks = process_pdf(adapted_file)
        
        state.vector_store = create_in_memory_vector_store(chunks)
        state.rag_chain = setup_rag_pipeline(state.vector_store)
        
        return {
            "message": "Document processed successfully", 
            "chunks": len(chunks)
        }
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during document processing")
    finally:
        os.remove(tmp_path)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not state.rag_chain:
        raise HTTPException(status_code=400, detail="Document not initialized")
    
    result = ask_rag_question(state.rag_chain, request.question)
    
    serialized_chunks = [
        {
            "content": chunk.page_content, 
            "page": chunk.metadata.get("page", "Unknown")
        } for chunk in result.get("context", [])
    ]
    
    full_context_str = "\n\n".join([chunk["content"] for chunk in serialized_chunks])
    eval_scores = evaluate_response(request.question, full_context_str, result.get("answer", ""))
    
    return {
        "answer": result.get("answer", ""),
        "retrieved_chunks": serialized_chunks,
        "evaluation": eval_scores
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_ready": state.rag_chain is not None}
