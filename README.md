# ecommerce-customer-support-chatbot
An AI-powered e-commerce customer support chatbot using a fine-tuned FLAN-T5 model, LoRA and Retrieval-Augmented Generation (RAG), with a Streamlit interface.
# E-commerce Customer Support Conversational AI

## Project Overview

This project focuses on developing an AI-powered conversational chatbot for e-commerce customer support. The system uses a Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) to answer customer queries using information from a WooCommerce customer-support knowledge base.

The objective is to help customers find relevant information about refunds, returns, order management, shipping and payments through a conversational interface.

## Objectives

- Develop an intelligent customer-support chatbot for e-commerce.
- Use a pretrained language model to generate conversational responses.
- Implement RAG to retrieve relevant information from documentation.
- Improve the relevance of responses by grounding them in retrieved knowledge.
- Develop a user-friendly chatbot interface using Streamlit.

## System Architecture

1. **Document Loading:** Load the WooCommerce customer-support PDF using PyPDFLoader.
2. **Text Chunking:** Split the document using RecursiveCharacterTextSplitter with a chunk size of 500 and overlap of 100.
3. **Embeddings:** Generate text embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
4. **Vector Search:** Store and retrieve document embeddings using FAISS.
5. **Response Generation:** Use the pretrained `google/flan-t5-base` model to generate responses using retrieved context.
6. **User Interface:** Provide an interactive chatbot using Streamlit.

## Technologies Used

- Python
- Google Colab
- Visual Studio Code
- PyTorch
- Hugging Face Transformers
- FLAN-T5
- Sentence Transformers
- LangChain
- FAISS
- Streamlit

## Repository Structure

- `app.py` – Streamlit chatbot application.
- `requirements.txt` – Project dependencies.
- `notebooks/` – Google Colab notebook and Python export.
- `knowledge_base/` – Customer-support documentation, subject to source redistribution permissions.
- `images/` – Application screenshots.

## Installation and Execution

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place the permitted WooCommerce knowledge-base PDF inside the `knowledge_base` folder.
5. Run the application:

```bash
streamlit run app.py
```

## Model and Retrieval Configuration

- Language model: `google/flan-t5-base`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector index: FAISS `IndexFlatL2`
- Chunk size: 500
- Chunk overlap: 100
- Retrieval: Top 3 document chunks

## Limitations

The chatbot relies on the information available in its knowledge base. It does not access live customer accounts, actual order records, payment gateways or shipment-tracking systems. Its responses should therefore be treated as documentation-based guidance rather than live transaction information.

## Research and Development

The Google Colab notebook contains the experimentation and development workflow, including the RAG implementation and language-model fine-tuning experiments. The Streamlit application provides the chatbot interface.

## Future Improvements

- Improve answer quality and retrieval relevance.
- Expand the customer-support knowledge base.
- Evaluate response quality and retrieval performance.
- Explore improved instruction-following language models.
- Integrate live e-commerce order and customer-support systems.
