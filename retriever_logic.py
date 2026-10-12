import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

@st.cache_resource
def load_retrievers():
    index_dir = "Index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_vectorstore = FAISS.load_local(
        f"{index_dir}/faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

# Custom Chain Class to completely avoid LangChain version errors
class CustomRAGChain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        
    def invoke(self, inputs):
        query = inputs["input"]
        
        # Get relevant documents (Supports all Streamlit environments)
        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(query)
        else:
            docs = self.retriever.get_relevant_documents(query)
            
        # Combine text for the AI
        context = "\n\n".join([f"Document: {d.metadata.get('document_name', 'Unknown')}\n{d.page_content}" for d in docs])
        
        # System Prompt
        prompt = f"""You are an Expert Contract Administrator and Legal Assistant for the Communication & Works (C&W) Department, KPK. 
You answer questions based strictly on the provided context which includes KPPRA Rules and FIDIC Conditions of Contract.

Follow these strict rules:
1. DIAGNOSE THE PHASE: First, determine if the query is about 'Tendering/Procurement' or 'Execution/Site'.
2. CHOOSE THE LAW: Apply KPPRA Rules for Procurement. Apply FIDIC for Execution/Site.
3. CITE YOUR SOURCES: Explicitly state the Rule Number or Clause Number.
4. NO HALLUCINATION: If not in context, state: 'This is not covered in the provided KPPRA or FIDIC documents.'

Context: 
{context}

User Query: {query}"""

        # Direct AI call
        response = self.llm.invoke(prompt)
        
        # Return exact format expected by app.py
        return {
            "answer": response.content,
            "context": docs
        }

def setup_rag_chain(api_key, retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.1, 
        google_api_key=api_key
    )
    return CustomRAGChain(llm, retriever)
