from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
import tempfile
import logging
from src.config import LANGCHAIN_API_KEY, LANGCHAIN_ENDPOINT

# Enable LangSmith Tracing BEFORE importing Langchain modules
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_PROJECT"] = "DocChat_RAG"

logger = logging.getLogger(__name__)

from src.document_processor import process_pdf
from src.vector_store import create_in_memory_vector_store
from src.graph import create_agent_graph, build_document_profile
from src.evaluator import evaluate_response
from src.guardrails import check_input_safety

app = FastAPI(title="DocChat RAG Microservice")

class AppState:
    def __init__(self):
        self.vector_store = None
        self.agent_graph = None

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
        
        # Build the new store and graph first, then swap, so a failed upload
        # leaves the previous document working.
        new_store = create_in_memory_vector_store(chunks)
        document_profile = build_document_profile(file.filename, chunks)
        new_graph = create_agent_graph(new_store, document_profile)

        old_store = state.vector_store
        state.vector_store = new_store
        state.agent_graph = new_graph

        # Delete the previous document's collection so its chunks can no longer be retrieved.
        if old_store is not None:
            try:
                old_store.delete_collection()
            except Exception as e:
                logger.warning(f"Could not delete previous collection: {e}")
        
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
    if not state.agent_graph:
        raise HTTPException(status_code=400, detail="Document not initialized")
    
    guardrail_result = check_input_safety(request.question)
    if not guardrail_result["is_safe"]:
        return {
            "answer": f"🛡️ **Security Alert:** Query blocked by Guardrails.\n\n**Reason:** {guardrail_result['reason']}",
            "retrieved_chunks": [],
            "evaluation": {"faithfulness": None, "relevance": None, "note": "Blocked by guardrail"}
        }
    
    try:
        initial_state = {"question": request.question, "chat_history": []}
        result = state.agent_graph.invoke(initial_state)
        
        final_answer = result.get("final_answer", "Error generating answer.")
        retrieved_context = result.get("retrieved_context", [])
        source_used = result.get("source_used", "unknown")
        
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        final_answer = f"Error executing workflow: {str(e)}"
        retrieved_context = []
        source_used = "error"
    
    serialized_chunks = [
        {
            "content": chunk.page_content, 
            "page": chunk.metadata.get("page", "Unknown") if hasattr(chunk, 'metadata') else "Unknown"
        } for chunk in retrieved_context
    ]
    
    full_context_str = "\n\n".join([chunk["content"] for chunk in serialized_chunks])
    
    if source_used == "document":
        eval_scores = evaluate_response(request.question, full_context_str, final_answer)
    else:
        # Web answers are not scored; returning None avoids showing fake 100% scores in the UI.
        eval_scores = {"faithfulness": None, "relevance": None, "note": "Web search answers are not evaluated"}
    
    return {
        "answer": f"**[Source: {source_used.upper()}]**\n\n{final_answer}",
        "retrieved_chunks": serialized_chunks,
        "evaluation": eval_scores
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_ready": state.agent_graph is not None}
