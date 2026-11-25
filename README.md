# 🛍️ E-commerce Customer Support Conversational AI

An AI-powered conversational chatbot designed to assist customers with e-commerce-related queries using **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)**.

## 📌 Project Overview

This project focuses on developing an intelligent customer-support chatbot that provides relevant and context-aware responses to customer queries in the e-commerce domain.

Customers frequently require assistance with refunds, returns, order management, shipping and payment-related questions. Handling these queries manually can increase customer-support workloads and response times.

To address this challenge, this project implements a conversational AI system that retrieves relevant information from a customer-support knowledge base and generates responses using a pretrained language model.

The chatbot is deployed through an interactive Streamlit interface.

## 🎯 Project Objectives

- Develop an AI-powered conversational chatbot for e-commerce customer support.
- Implement Retrieval-Augmented Generation (RAG) to retrieve relevant information from documentation.
- Generate context-aware responses using a pretrained Large Language Model.
- Use semantic embeddings and vector similarity search for document retrieval.
- Build an interactive and user-friendly chatbot interface using Streamlit.
- Explore LoRA-based fine-tuning as part of the model experimentation process.

## 🏗️ System Architecture

The project follows a Retrieval-Augmented Generation pipeline.

1. **Document Loading:** Load customer-support documentation in PDF format using PyPDFLoader.
2. **Text Chunking:** Split documents into smaller text chunks using RecursiveCharacterTextSplitter.
3. **Embedding Generation:** Convert text chunks into numerical representations using Sentence Transformers.
4. **Vector Database:** Store embeddings using FAISS for efficient similarity search.
5. **Context Retrieval:** Retrieve the most relevant document chunks based on the customer's question.
6. **Response Generation:** Pass the retrieved context and customer question to FLAN-T5 to generate a response.
7. **User Interface:** Display the generated response through a Streamlit chatbot application.

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google Colab | Model experimentation and development |
| Visual Studio Code | Application development |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained language model |
| FLAN-T5 Base | Response generation |
| Sentence Transformers | Text embedding generation |
| FAISS | Vector similarity search |
| LangChain | PDF loading and text splitting |
| Streamlit | Chatbot user interface |
| PEFT / LoRA | Parameter-efficient fine-tuning experiments |

## 🤖 Models and Configuration

**Language Model**
- Model: `google/flan-t5-base`

**Embedding Model**
- Model: `sentence-transformers/all-MiniLM-L6-v2`

**Vector Search**
- FAISS IndexFlatL2

**Text Chunking**
- Chunk size: 500
- Chunk overlap: 100

**Retrieval Configuration**
- Top-K retrieval: 3 relevant chunks

## 📂 Project Structure

```text
ecommerce-customer-support-chatbot/
│
├── app.py
│
├── requirements.txt
│
├── .gitignore
│
├── Notebooks/
│   ├── E_Commerce_Support_LLM_Bot.ipynb
│   └── e_commerce_support_llm_bot.py
│
└── knowledge_base/
    └── Customer_Support_PDF.pdf
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Someshwar24/ecommerce-customer-support-chatbot.git
```

### 2. Navigate to the Project Directory

```bash
cd ecommerce-customer-support-chatbot
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The chatbot will open in your browser.

## 💬 Example Customer Queries

The chatbot is designed to handle questions related to:

- How can I request a refund for my order?
- What is the difference between automatic and manual refunds?
- How can I track my order?
- What should I do if my order is delayed?
- How can I manage order cancellations?
- What are the available payment options?

## 📊 Development and Experimentation

The Google Colab notebook contains the development and experimentation workflow, including:

- Dataset preparation and preprocessing.
- Pretrained language model experimentation.
- LoRA-based fine-tuning experiments.
- Document processing and text chunking.
- Embedding generation and FAISS indexing.
- RAG-based response generation.

The Streamlit application provides the interactive deployment interface for the RAG chatbot.

## ⚠️ Limitations

- Responses depend on the information available in the uploaded knowledge base.
- The chatbot does not access live customer accounts or order databases.
- It cannot independently process refunds or modify orders.
- It does not connect to live payment gateways or courier tracking systems.
- Response quality depends on document quality, retrieval relevance and language-model capabilities.

## 🚀 Future Enhancements

- Improve retrieval accuracy and response generation.
- Expand the knowledge base with additional e-commerce documentation.
- Integrate real-time order tracking and customer-account information.
- Improve chatbot evaluation using retrieval and response-quality metrics.
- Explore more advanced instruction-following language models.
- Deploy the chatbot to a cloud platform.

## 👨‍💻 Author

**Someshwar24**

GitHub: https://github.com/Someshwar24

## 📄 License

This repository contains project code and research materials. Refer to the respective dataset and documentation licenses before redistributing their contents.
