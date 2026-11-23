import os
import random
import logging
from statistics import mean
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from src.document_processor import process_pdf
from src.vector_store import create_in_memory_vector_store
from src.evaluator import evaluate_response
from src.config import OPENAI_API_KEY, COHERE_API_KEY

load_dotenv()
logging.basicConfig(level=logging.ERROR)

def generate_synthetic_questions(chunks: list, num_questions: int = 15) -> list[str]:
    print(f"Generating {num_questions} synthetic test questions...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract a single complex question that can be answered using only the provided chunk. Return only the question string."),
        ("human", "Chunk:\n{chunk}")
    ])
    chain = prompt | llm

    valid_chunks = [c for c in chunks if len(c.page_content) > 200]
    sample_chunks = random.sample(valid_chunks, min(num_questions, len(valid_chunks)))
    
    questions = []
    for i, chunk in enumerate(sample_chunks):
        response = chain.invoke({"chunk": chunk.page_content}).content.strip()
        questions.append(response)
        
    return questions

def run_evaluation(pdf_path: str):
    print(f"Processing {pdf_path}...")
    
    class LocalFileAdapter:
        def __init__(self, path):
            self.path = path
        def getvalue(self):
            with open(self.path, 'rb') as f:
                return f.read()

    chunks = process_pdf(LocalFileAdapter(pdf_path))
    vector_store = create_in_memory_vector_store(chunks)
    
    questions = generate_synthetic_questions(chunks, num_questions=15)
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    wide_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    cohere_rerank = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-english-v3.0", top_n=3)
    advanced_retriever = ContextualCompressionRetriever(base_compressor=cohere_rerank, base_retriever=wide_retriever)
    
    generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using ONLY the provided context. If you don't know, say 'I don't know.'\n\nContext: {context}"),
        ("human", "{question}")
    ])
    chain = prompt | generator_llm

    results = {"base_rel": [], "base_faith": [], "adv_rel": [], "adv_faith": []}

    print(f"Running evaluation on {len(questions)} queries...")

    for i, q in enumerate(questions):
        # Baseline
        base_docs = base_retriever.invoke(q)
        base_context = "\n\n".join([d.page_content for d in base_docs])
        base_answer = chain.invoke({"question": q, "context": base_context}).content
        base_eval = evaluate_response(q, base_context, base_answer)
        
        results["base_rel"].append(base_eval.get("relevance", 0))
        results["base_faith"].append(base_eval.get("faithfulness", 0))

        # Advanced
        adv_docs = advanced_retriever.invoke(q)
        adv_context = "\n\n".join([d.page_content for d in adv_docs])
        adv_answer = chain.invoke({"question": q, "context": adv_context}).content
        adv_eval = evaluate_response(q, adv_context, adv_answer)
        
        results["adv_rel"].append(adv_eval.get("relevance", 0))
        results["adv_faith"].append(adv_eval.get("faithfulness", 0))

    print("\nEvaluation Results:")
    print("Baseline:")
    print(f"  Relevance:    {mean(results['base_rel']):.1f}%")
    print(f"  Faithfulness: {mean(results['base_faith']):.1f}%")
    print("Cohere Re-ranking:")
    print(f"  Relevance:    {mean(results['adv_rel']):.1f}%")
    print(f"  Faithfulness: {mean(results['adv_faith']):.1f}%")

if __name__ == "__main__":
    TEST_PDF = "test_document.pdf" 
    
    if not os.path.exists(TEST_PDF):
        print(f"Error: {TEST_PDF} not found.")
    else:
        run_evaluation(TEST_PDF)
