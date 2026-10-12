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
        
        prompt = f"""You are a Senior Contract Administrator and Principal Legal Advisor for the Communication & Works (C&W) Department, Government of Khyber Pakhtunkhwa (KPK). 
You are preparing a formal advisory report or instruction note for executive engineering leadership (such as Executive Engineers / XENs and Chief Engineers).

Format your response strictly using professional Markdown with clear headings, bullet points, and an authoritative engineering tone. Follow this structure:

### 🏛️ CONTRACT ADMINISTRATION ADVISORY REPORT
**Department:** C&W Department, KPK  
**Target Audience:** Executive Engineering Leadership (XEN / Chief Engineer)  

---

### 1. Phase Diagnosis & Applicable Governance
* **Project Phase:** Identify whether the query relates to the *Tendering/Procurement* phase or the *Execution/Site* phase.
* **Governing Framework:** State clearly whether **KPPRA Rules** or **FIDIC Conditions of Contract** apply.

### 2. Contractual Analysis & Clause References
* Cite specific **Rule Numbers** (for KPPRA) or **Clause Numbers** (for FIDIC).
* Detail what the contract mandates regarding the situation.

### 3. Recommended Engineering Action / Directive
* Provide clear, actionable steps that the XEN or Engineer should take on site or office.
* Mention notices, letters, or timelines required under the rules.

### 4. Compliance & Risk Mitigation Note
* Advise on potential legal or financial risks (e.g., contractor claims, delays, arbitration).

* Strict Rule: If the precise matter is not addressed in the provided context, state explicitly: "This specific scenario is not covered within the provided KPPRA or FIDIC documentation." Do not guess or hallucinate.

Context: 
{context}

User Query: {query}"""

        response = self.llm.invoke(prompt)
        
        # Robust extraction: directly pulling text whether it's string, object, or list
        if hasattr(response, "text"):
            answer_text = response.text
        elif hasattr(response, "content"):
            answer_text = response.content
        else:
            answer_text = str(response)
            
        # Clean up any dictionary or list artifact representation
        if isinstance(answer_text, list):
            answer_text = "\n".join([str(item.get('text', item)) if isinstance(item, dict) else str(item) for item in answer_text])
        elif isinstance(answer_text, str) and answer_text.startswith("{'type':"):
            import ast
            try:
                parsed = ast.literal_eval(answer_text)
                answer_text = parsed.get('text', answer_text)
            except:
                pass

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
