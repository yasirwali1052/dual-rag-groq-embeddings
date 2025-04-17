

# Dual-Paper Retrieval-Augmented Generation (RAG) with Groq LLaMA3 and Hugging Face

This project implements a Retrieval-Augmented Generation (RAG) model to answer queries from two research papers using Groq's LLaMA3 model. It compares two embedding techniques:

1. **Ollama’s `nomic-embed-text`** for embedding one paper.
2. **Hugging Face’s `all-MiniLM-L6-v2`** for embedding the other paper.

The app is built with **Streamlit** for the user interface and uses **Langchain** to connect the embeddings with Groq’s LLaMA3 model.

## Features:
- **Query research papers**: Input a query and get an accurate response based on the contents of the research papers.
- **Document similarity search**: Explore documents similar to the context of your query.
- **Two embedding techniques**: Compare Ollama embeddings with Hugging Face embeddings for better retrieval results.

---

## Requirements

Before running the project, ensure you have the following API keys and environment setup:

### 1. **Groq API Key**:
   - Go to [Groq API](https://groq.com/) and register for an API key.
   - Store it in your `.env` file as `GROQ_API_KEY`.

### 2. **Hugging Face Token**:
   - Sign up or log in at [Hugging Face](https://huggingface.co/).
   - Create a new token from your [settings](https://huggingface.co/settings/tokens).
   - Add the token in the `.env` file as `HF_TOKEN`.

---

## Setup

### 1. **Clone the Repository**:

```bash
git clone https://github.com/yasirwali1052/dual-rag-groq-embeddings.git
cd dual-rag-groq-embeddings
```

### 2. **Install Dependencies**:

Install all the necessary libraries using `pip`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```txt
streamlit
python-dotenv
openai
langchain
langchain-community
langchain-core
langchain-openai
langchain-groq
langchain-huggingface
faiss-cpu==1.7.4
pypdf
torch
transformers==4.28.1
tensorflow==2.11.0
```

---

## `.env` Setup

Create a `.env` file in the project root directory. The file should contain the following environment variables:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### **How to Generate API Keys**:

1. **Groq API Key**:
   - Go to [Groq API](https://groq.com/) and sign up.
   - Obtain your API key from your Groq account.

2. **Hugging Face Token**:
   - Log in to [Hugging Face](https://huggingface.co/).
   - Go to your [settings](https://huggingface.co/settings/tokens).
   - Create and copy the **User Access Token** and paste it in the `.env` file.

---

## Using the Application

### 1. **Start the Streamlit Application**:

To run the app locally, use the following command:

```bash
streamlit run app.py
```

This will start a Streamlit server at `http://localhost:8501` where you can interact with the RAG model.

### 2. **Embedding and Document Setup**:

The `create_vector_embedding()` function performs the following tasks:

- **Ollama Embedding** (for one paper): It uses Ollama’s `nomic-embed-text` for generating embeddings from the provided documents (e.g., research papers).
- **Hugging Face Embedding** (for the other paper): The Hugging Face `all-MiniLM-L6-v2` model is used to generate document embeddings.

The documents are split into chunks using `RecursiveCharacterTextSplitter` and stored in a FAISS vector store for efficient retrieval.

---

## Code Explanation

### **Embedding with Ollama**:

```python
from langchain_ollama import OllamaEmbeddings

st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

To use Ollama embeddings:
- Make sure you have an Ollama account and use the `nomic-embed-text` model.
- The embeddings are used to process and store the text from the research papers.

### **Embedding with Hugging Face**:

```python
from langchain_huggingface import HuggingFaceEmbeddings

st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

To use Hugging Face embeddings:
- You can use Hugging Face’s `all-MiniLM-L6-v2` model for document embedding.
- It’s particularly useful for smaller and efficient embeddings.

### **Retrieving Answers**:

Once embeddings are generated, we use a **retrieval chain** to fetch relevant documents based on the user's query. The chain uses the LLaMA3 model to generate a response from the documents.

```python
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

This connects the vector store with the query-answering chain, allowing the model to answer the user’s input based on document similarity.

---

## FAQ

### **1. Why am I getting errors related to missing API keys?**

Ensure your `.env` file is properly set up with your `GROQ_API_KEY` and `HF_TOKEN`. You must have both keys in place for the application to work properly.

### **2. How can I improve the performance of the model?**

- You can experiment with different chunk sizes and overlaps in the `RecursiveCharacterTextSplitter`.
- Try using different embeddings from Hugging Face or Ollama for better results.
- Consider optimizing the model's parameters in Groq’s LLaMA3 model for better response times.

### **3. How do I update the embeddings with new papers?**

Simply add more PDFs to the `research_papers` directory, and click on **"Document Embedding"** in the app to regenerate the embeddings. You can load up to 50 documents by default.

---

## Contributing

Feel free to fork the repository, submit issues, and create pull requests. Any contributions to improve the RAG model or the user interface are welcome!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Example Folder Structure:

```
dual-paper-ragg/
├── research_papers/
│   ├── Attention.pdf
│   ├── LLM.pdf
├── .env
├── app.py
├── requirements.txt
└── README.md
```

---

This `README.md` covers everything necessary to set up and run your project. Let me know if you need further adjustments!
