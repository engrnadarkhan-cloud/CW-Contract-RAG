import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

@st.cache_resource
def load_retrievers():
    index_dir = "Index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_vectorstore = FAISS.load_local(f"{index_dir}/faiss_index", embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
    
    with open(f"{index_dir}/bm25_retriever.pkl", 'rb') as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 3
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever

def setup_rag_chain(api_key, retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=api_key)
    system_prompt = (
        "You are an Expert Contract Administrator and Legal Assistant for the Communication & Works (C&W) Department, KPK. "
        "You answer questions based strictly on the provided context which includes KPPRA Rules and FIDIC Conditions of Contract.\n\n"
        "Follow these strict rules:\n"
        "1. DIAGNOSE THE PHASE: First, determine if the user's query is about the 'Tendering/Procurement' phase or the 'Execution/Site' phase.\n"
        "2. CHOOSE THE LAW: \n"
        "   - For Tendering/Procurement queries, apply KPPRA Rules.\n"
        "   - For Execution/Site queries, apply FIDIC Conditions of Contract.\n"
        "3. CITE YOUR SOURCES: You must explicitly state the Rule Number (for KPPRA) or Clause Number (for FIDIC) in your answer.\n"
        "4. NO HALLUCINATION: If the exact answer is not in the provided context, clearly state: 'This is not covered in the provided KPPRA or FIDIC documents.' Do not guess.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain
