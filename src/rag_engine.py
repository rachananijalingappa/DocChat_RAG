from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from src.config import GROQ_API_KEY, COHERE_API_KEY, LLM_MODEL_NAME, RETRIEVER_K

def setup_rag_pipeline(vector_store):
    """
    Sets up the LangChain RAG pipeline with Cohere Re-ranking.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please set it in the .env file.")
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY is not set. Please set it in the .env file.")
        
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.0
    )

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    cohere_rerank = CohereRerank(
        cohere_api_key=COHERE_API_KEY, 
        model="rerank-english-v3.0", 
        top_n=RETRIEVER_K
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank,
        base_retriever=base_retriever
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Keep the answer clear and concise."
        "\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)
    
    return rag_chain

def ask_rag_question(rag_chain, question: str):
    """Executes the RAG pipeline for a given question."""
    try:
        response = rag_chain.invoke({"input": question})
        return {
            "answer": response["answer"],
            "context": response["context"]
        }
    except Exception as e:
        return {"answer": f"Error querying the model: {str(e)}", "context": []}
