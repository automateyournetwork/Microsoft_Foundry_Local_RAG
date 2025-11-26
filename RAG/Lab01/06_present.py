import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from foundry_local import FoundryLocalManager
from langchain_openai import ChatOpenAI

#foundry setup
# Choose an alias from the Foundry Local catalog
# e.g. "qwen2.5-0.5b", "phi-4-mini-instruct", etc.
alias = "qwen2.5-0.5b"   # or a phi-* alias when you pick one

# Start Foundry Local and load the model
manager = FoundryLocalManager(alias)

# Get the concrete model id (the optimized ONNX variant)
model_id = manager.get_model_info(alias).id
print(f"🤖 Using Foundry Local model: {model_id}"
      )

# --- Load + Embed + Index (cache this to avoid reloading every time) ---
@st.cache_resource
def setup_rag_chain():
    loader = PyPDFLoader("2312_10997v5.pdf")
    documents = loader.load()

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    splitter = SemanticChunker(embedding)
    chunks = splitter.split_documents(documents)

    vector_store = Chroma.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Now configure ChatOpenAI to talk to the *local* OpenAI-compatible endpoint
    llm = ChatOpenAI(
        model=model_id,
        base_url=manager.endpoint,   # Foundry Local REST endpoint
        api_key=manager.api_key,     # Fake key used locally, but required by SDK
        temperature=0,
        streaming=False,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# --- Streamlit UI ---
st.set_page_config(page_title="Ask the RAG Paper", page_icon="📄")
st.title("📄 Ask the RAG Paper")
st.markdown("Type your question below to explore the Retrieval-Augmented Generation for Large Language Models Survey")

qa_chain = setup_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("💬 Your question:", placeholder="e.g. What is RAG?")

if question:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({
            "question": question,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append((question, response["answer"]))

# --- Display chat history ---
for user_q, answer in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑‍💻 You:** {user_q}")
    st.markdown(f"**🤖 RAGBot:** {answer}")
    st.markdown("---")
