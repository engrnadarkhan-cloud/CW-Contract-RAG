import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

@st.cache_resource
def load_retrievers():
    # 1. Direct PDFs se text load karo jo GitHub par mojood hain
    # (Apne GitHub folder me check kar lein ke PDFs ka naam yehi hai ya unki jagah sahi path hai)
    pdf_files = ["KPPRA_Rules.pdf", "FIDIC_Conditions.pdf"] 
    documents = []
    
    for path in pdf_files:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['document_name'] = path
            documents.extend(docs)
            
    # Fallback agar file name thora mukhtalif ho
    if not documents:
        for file in os.listdir("."):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['document_name'] = file
                documents.extend(docs)

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Direct Memory me Vector Store banana (No local file read error!)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})

class CustomRAGChain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        
    def invoke(self, inputs):
        query = inputs["input"]
        docs = self.retriever.invoke(query) if hasattr(self.retriever, "invoke") else self.retriever.get_relevant_documents(query)
        
        context = "\n\n".join([f"Document: {d.metadata.get('document_name', 'Unknown')}\n{d.page_content}" for d in docs])
        
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

        response = self.llm.invoke(prompt)
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
