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
            documents.extend(docs)
            
    if not documents:
        for file in os.listdir("."):
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(file)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['document_name'] = file
                documents.extend(docs)

    if not documents:
        raise ValueError("❌ Error: No PDF files found in repository root.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
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
1. DIAGNOSE THE PHASE: First, determine if the query is about 'Tendering/Procurement' or 'Execution/Site'.
2. CHOOSE THE LAW: Apply KPPRA Rules for Procurement. Apply FIDIC for Execution/Site.
3. CITE YOUR SOURCES: Explicitly state the Rule Number or Clause Number.
4. NO HALLUCINATION: If not in context, state: 'This is not covered in the provided KPPRA or FIDIC documents.'

Context: 
{context}

User Query: {query}"""

        # Direct AI invocation and robust text extraction
        raw_response = self.llm.invoke(prompt)
        
        # Ensure we extract pure string text, handling any dictionary/object wrapping
        if hasattr(raw_response, "content"):
            answer_text = raw_response.content
        else:
            answer_text = str(raw_response)
            
        # If it accidentally returns a list representation as string, clean it
        if isinstance(answer_text, list):
            answer_text = "".join([str(item) for item in answer_text])

        return {
            "answer": answer_text,
            "context": docs
        }

def setup_rag_chain(api_key, retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash", 
        temperature=0.1, 
        google_api_key=api_key
    )
    return CustomRAGChain(llm, retriever)
