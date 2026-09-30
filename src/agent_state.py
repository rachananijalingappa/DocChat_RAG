from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    The state dictionary that is passed between nodes in our LangGraph.
    """
    question: str
    chat_history: List[Dict[str, str]]
    next_step: Optional[str]        # 'rag' or 'web_search'
    retrieved_context: List[Any]    # Can be Document objects from RAG or string snippets from Web Search
    final_answer: Optional[str]
    source_used: Optional[str]      # Tracks whether we used 'document' or 'web' for the UI
