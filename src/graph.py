import json
import logging
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from ddgs import DDGS

from src.agent_state import AgentState
from src.config import OPENAI_API_KEY, COHERE_API_KEY, RETRIEVER_K

logger = logging.getLogger(__name__)

class WebSearchTool:
    def run(self, query: str) -> str:
        try:
            results = DDGS().text(query, max_results=3)
            return str(results) if results else "No results found."
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return "Search failed."

def create_agent_graph(vector_store):
    router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
    generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
    web_search = WebSearchTool()

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    cohere_rerank = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-english-v3.0", top_n=RETRIEVER_K)
    retriever = ContextualCompressionRetriever(base_compressor=cohere_rerank, base_retriever=base_retriever)

    def route_question(state: AgentState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Route the user's question to the appropriate data source.\n"
                       "If it relates to an uploaded document, route to 'rag'.\n"
                       "If it requires current events or general knowledge, route to 'web_search'.\n"
                       "Output strictly as JSON: {{\"route\": \"rag\"}} or {{\"route\": \"web_search\"}}"),
            ("human", "{question}")
        ])
        
        response = (prompt | router_llm).invoke({"question": state["question"]}).content.strip()
        
        # Strip potential markdown formatting
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("\n", 1)[0]
            
        try:
            decision = json.loads(response).get("route", "rag")
        except json.JSONDecodeError:
            decision = "rag"
            
        logger.debug(f"Question routed to: {decision}")
        return {"next_step": decision}

    def rag_agent(state: AgentState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"retrieved_context": docs, "source_used": "document"}

    def web_search_agent(state: AgentState) -> dict:
        search_result = web_search.run(state["question"])
        
        class SearchResultAdapter:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {"page": "Web Search"}
                
        return {"retrieved_context": [SearchResultAdapter(search_result)], "source_used": "web"}

    def generate_answer(state: AgentState) -> dict:
        context_str = "\n\n".join([doc.page_content for doc in state["retrieved_context"]])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using ONLY the provided context.\n"
                       "If the context does not contain the answer, output exactly: 'I don't know.'\n\n"
                       "Context: {context}"),
            ("human", "{question}")
        ])
        
        response = (prompt | generator_llm).invoke({
            "question": state["question"], 
            "context": context_str
        })
        return {"final_answer": response.content}

    def decide_next_step(state: AgentState) -> str:
        return state["next_step"]

    workflow = StateGraph(AgentState)
    workflow.add_node("router", route_question)
    workflow.add_node("rag", rag_agent)
    workflow.add_node("web_search", web_search_agent)
    workflow.add_node("generator", generate_answer)
    
    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", decide_next_step, {"rag": "rag", "web_search": "web_search"})
    workflow.add_edge("rag", "generator")
    workflow.add_edge("web_search", "generator")
    workflow.add_edge("generator", END)
    
    return workflow.compile()
