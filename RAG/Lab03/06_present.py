import streamlit as st
from genie.testbed import load
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import tempfile, os, json, uuid
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

# --- UI Setup ---
st.set_page_config(page_title="Chat with CAT9k_AO Interface Table", page_icon="🛣️")
st.title("🛣️ Chat with Your CAT9k_AO Interface Table")
st.markdown("Ask anything about the live Interface table retrieved from CAT9k_AO using pyATS!")

# --- Cached RAG Pipeline Setup ---
def setup_routing_chain():
    # Step 1: Connect to CAT9k_AO and get routing table
    testbed = load("testbed.yaml")
    device = testbed.devices["CAT9k_AO"]
    print("🔌 Connecting to CAT9k_AO...")
    device.connect(log_stdout=True)
    parsed_output = device.parse("show ip interface brief")

    # Step 2: Write JSON to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
        json.dump(parsed_output, tmp, indent=2)
        tmp_path = tmp.name

    # Step 3: Load into LangChain Documents
    loader = JSONLoader(
        file_path=tmp_path,
        jq_schema='.',  # 1 route per document
        text_content=False
    )
    documents = loader.load()
    os.remove(tmp_path)

    # Step 4: Embed & Split
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    splitter = SemanticChunker(embedding)
    chunks = splitter.split_documents(documents)

    # Step 5: Build Chroma vector store
    vector_store = Chroma.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Step 6: Set up RAG chain
    # Now configure ChatOpenAI to talk to the *local* OpenAI-compatible endpoint
    llm = ChatOpenAI(
        model=model_id,
        base_url=manager.endpoint,   # Foundry Local REST endpoint
        api_key=manager.api_key,     # Fake key used locally, but required by SDK
        temperature=0,
        streaming=False,
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

qa_chain = setup_routing_chain()

# --- Chat Interaction ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("💬 Ask a question about CAT9k_AO's Interface table:")

if question:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({
            "question": question,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append((question, response["answer"]))

# --- Display Chat History ---
for user_q, answer in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑‍💻 You:** {user_q}")
    st.markdown(f"**🤖 CAT9k_AO Interface Bot:** {answer}")
    st.markdown("---")
