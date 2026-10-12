import streamlit as st
from retriever_logic import load_retrievers, setup_rag_chain

st.set_page_config(page_title="C&W Contract Assistant", page_icon="🏗️", layout="centered")

st.title("🏗️ C&W Contract Assistant (KPK)")
st.markdown("**Powered by KPPRA Rules & FIDIC Conditions**")
st.markdown("Ask contract administration, dispute resolution, or procurement queries.")

with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Enter your Gemini API Key:", type="password")
        st.info("Get a free key from Google AI Studio.")
    
    st.divider()
    st.write("👨‍💻 Developed by **Engineer Nadir Khan**")
    st.write("Gold Medalist BS Civil | MS CEM NUST")

retriever = load_retrievers()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("E.g., What is the procedure if a contractor claims extension of time due to rain?"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar first.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal documents..."):
            try:
                rag_chain = setup_rag_chain(api_key, retriever)
                response = rag_chain.invoke({"input": query})
                
                answer = response["answer"]
                sources = response.get("context", [])
                
                st.markdown(answer)
                
                with st.expander("📄 View Retrieved Source Clauses"):
                    for i, doc in enumerate(sources):
                        st.write(f"**Source {i+1}:** {doc.metadata.get('document_name', 'Unknown Document')}")
                        st.caption(doc.page_content[:300] + "...")
                        st.divider()
                        
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
