import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

@st.cache_resource
def load_retrievers():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = []
    
    pdf_files = ["KPPRA_Rules.pdf", "FIDIC_Conditions.pdf"]
    
    for file in pdf_files:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            docs = loader.load()
            for doc in docs:
                doc.metadata['document_name'] = file
                doc.metadata['source_type'] = "FIDIC" if "fidic" in file.lower() else "KPPRA"
            documents.extend(docs)
            
    if not documents:
        for file in os.listdir("."):
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(file)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['document_name'] = file
                    doc.metadata['source_type'] = "General"
                documents.extend(docs)

    if not documents:
        raise ValueError("❌ Error: No PDF files found in the repository root.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=250,
        separators=["\n\n", "\n", "Clause", "Rule", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

class CustomRAGChain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        
    def invoke(self, inputs):
        query = inputs["input"]
        
        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(query)
        else:
            docs = self.retriever.get_relevant_documents(query)
            
        context = "\n\n".join([f"Document Source: {d.metadata.get('document_name', 'Unknown')}\n{d.page_content}" for d in docs])
        
        prompt = f"""You are an Expert Contract Administrator and Legal Assistant for the Communication & Works (C&W) Department, KPK. 
You answer questions based strictly on the provided context which includes KPPRA Rules and FIDIC Conditions of Contract.

Follow these strict rules:
1. DIAGNOSE THE PHASE: First, determine if the user's query is about the 'Tendering/Procurement' phase or the 'Execution/Site' phase.
2. CHOOSE THE LAW: 
   - For Tendering/Procurement queries, apply KPPRA Rules.
   - For Execution/Site queries, apply FIDIC Conditions of Contract.
3. CITE YOUR SOURCES: You must explicitly state the Rule Number (for KPPRA) or Clause Number (for FIDIC) in your answer.
4. NO HALLUCINATION: If the exact answer is not in the provided context, clearly state: 'This is not covered in the provided KPPRA or FIDIC documents.' Do not guess.

Context: 
{context}

User Query: {query}"""

        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "context": docs
        }

def setup_rag_chain(api_key, retriever):
    # Updated to Gemini 3.8 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash", 
        temperature=0.1, 
        google_api_key=api_key
    )
    return CustomRAGChain(llm, retriever)
