import json
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config import GROQ_API_KEY, LLM_MODEL_NAME

def evaluate_response(question: str, context: str, answer: str) -> dict:
    """
    Evaluates the RAG response on Faithfulness and Relevance using an LLM-as-a-judge.
    Returns a dictionary with scores between 0 and 100.
    """
    if not GROQ_API_KEY:
        return {"faithfulness": 0, "relevance": 0, "error": "API Key missing"}

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.0
    )

    evaluation_prompt = """
    You are an impartial judge evaluating a RAG (Retrieval-Augmented Generation) system.
    You will be given a User Question, the Retrieved Context from a document, and the System's Answer.
    
    You must evaluate the system based on two metrics:
    1. Faithfulness (0-100): Is the answer strictly derived from the provided context? Deduct points if the answer hallucinates information not present in the context.
    2. Relevance (0-100): Does the answer directly address the user's question? Deduct points if it goes off on a tangent or fails to answer the question.
    
    User Question: {question}
    
    Retrieved Context:
    {context}
    
    System Answer: {answer}
    
    You must return ONLY a raw JSON object with the two scores as integers. Do not include any markdown formatting, backticks, or explanation.
    Example output:
    {{"faithfulness": 95, "relevance": 100}}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", evaluation_prompt)
    ])
    
    chain = prompt | llm

    try:
        response = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })
        
        # Clean up potential markdown formatting from the LLM
        output = response.content.strip()
        if output.startswith("```json"):
            output = output[7:]
        if output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]
            
        scores = json.loads(output.strip())
        return {
            "faithfulness": scores.get("faithfulness", 0),
            "relevance": scores.get("relevance", 0)
        }
    except Exception as e:
        logging.getLogger(__name__).error(f"Evaluation failed: {e}")
        return {"faithfulness": 0, "relevance": 0, "error": str(e)}
