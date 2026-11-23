import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocChat RAG", page_icon="📚", layout="wide")

def main():
    st.title("DocChat: Multi-Agent RAG")
    st.write("Upload PDFs and interact with the content via a FastAPI & LangGraph microservice architecture.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False

    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Select a PDF file", type="pdf")
        
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing document and building vector store..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success("Document processed successfully.")
                        st.session_state.rag_ready = True
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI backend. Is it running?")

    if not st.session_state.rag_ready:
        st.info("Please upload and process a PDF from the sidebar to begin.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your PDF..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    response = requests.post(f"{API_URL}/ask", json={"question": prompt})
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        retrieved_chunks = data["retrieved_chunks"]
                        
                        st.markdown(answer)
                        
                        if "evaluation" in data:
                            eval_data = data["evaluation"]
                            if eval_data.get("faithfulness") is None or "error" in eval_data:
                                # Unscored (web search, blocked) or the judge call failed: don't show a number.
                                reason = eval_data.get("note") or f"evaluation failed ({eval_data.get('error')})"
                                st.caption(f"Answer not scored: {reason}")
                            else:
                                cols = st.columns(2)
                                with cols[0]:
                                    st.metric("Faithfulness Score", f"{eval_data['faithfulness']}%")
                                with cols[1]:
                                    st.metric("Relevance Score", f"{eval_data['relevance']}%")
                        
                        if retrieved_chunks:
                            with st.expander("View Sources"):
                                for i, chunk in enumerate(retrieved_chunks):
                                    page_num = chunk.get("page", "Unknown")
                                    if isinstance(page_num, int):
                                        page_num += 1
                                        
                                    st.markdown(f"**Chunk {i+1} (Page {page_num}):**")
                                    st.text(chunk['content'])
                                    st.markdown("---")
                                    
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"API Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI backend. Is it running?")

if __name__ == "__main__":
    main()
