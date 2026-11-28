import streamlit as st
import torch
import numpy as np
import faiss

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "knowledge_base" / "Customer_Support_PDF.pdf"


@st.cache_resource(show_spinner="Loading the PDF, embeddings and language model...")
def load_rag_pipeline(pdf_path_string, pdf_modified_time):
    global embedding_model, chunks, index, tokenizer, llm, device

    pdf_file = Path(pdf_path_string)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_file}\n"
            "Place Customer_Support_PDF.pdf inside the knowledge_base folder."
        )

    # PDF loading
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # Embedding model
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    chunk_texts = [chunk.page_content for chunk in chunks]

    # Embedding generation
    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # FAISS index construction
    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # FLAN-T5-base model
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = llm.to(device)
    llm.eval()

    return (
        embedding_model,
        chunks,
        index,
        tokenizer,
        llm,
        device
    )


def retrieve_context(query, k=3):
    # Convert the user question into an embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype(np.float32)

    # Search FAISS for the top-k relevant chunks
    distances, indices = index.search(query_embedding, k)

    # Collect the retrieved document content
    retrieved_chunks = []

    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            retrieved_chunks.append(chunks[int(idx)].page_content)

    # Combine the retrieved chunks into one context
    context = "\n\n".join(retrieved_chunks)

    return context


def rag_response(query):
    # Retrieve relevant documents
    context = retrieve_context(query, k=3)

    # Exact prompt supplied from the working Google Colab code
    prompt = f"""
        Answer the customer's question using the information in the context.

        Context:
        {context}

        Customer Question:
        {query}

        Provide a helpful, clear answer with the relevant steps.
            Do not repeat instructions or invent information.

        Answer:
        """

    # Exact tokenizer settings supplied from Google Colab
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # Exact generation settings supplied from Google Colab
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            num_beams=4
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response

st.set_page_config(
    page_title="E-commerce Customer Support",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ E-commerce Customer Support")
st.caption("Ask a question about refunds, returns, orders, shipping or payments.")

if not PDF_PATH.exists():
    st.error(
        "Customer_Support_PDF.pdf was not found. "
        "Place it inside the knowledge_base folder."
    )
    st.stop()

try:
    (
        embedding_model,
        chunks,
        index,
        tokenizer,
        llm,
        device
    ) = load_rag_pipeline(
        str(PDF_PATH),
        PDF_PATH.stat().st_mtime
    )
except Exception as error:
    st.error("The RAG pipeline could not be loaded.")
    st.exception(error)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept one question at a time; retain earlier conversation in the UI
query = st.chat_input("How can I get a refund for my order?")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Finding the answer..."):
            response = rag_response(query)
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
