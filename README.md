

This repository contains both **Assignment 1 (RAG Prototype)** and **Assignment 2 (Evaluation Framework)** for the **AI Intern Hiring Task** at **Kalpit Pvt Ltd, UK**.

The project implements a **complete Retrieval-Augmented Generation (RAG) system** using:

- **LangChain (LCEL)**
- **ChromaDB** (local vector store)
- **HuggingFace sentence-transformers**
- **Ollama + Mistral 7B** (local LLM)
- Python 3.11+

The system loads Ambedkar’s writings, builds embeddings locally, retrieves relevant chunks, and generates answers via a local LLM.  
The evaluation framework computes **retrieval quality**, **answer quality**, **semantic similarity**, and **chunking performance** across a test dataset.

---

# 📂 Project Structure

```

├── main.py                  # Interactive RAG system (Assignment 1)
├── utils.py                 # Embedding, ingestion, retrieval functions
├── evaluation.py            # Full evaluation pipeline (Assignment 2)
├── corpus/                  # 6 Ambedkar documents for evaluation
│   ├── speech1.txt
│   ├── speech2.txt
│   ├── speech3.txt
│   ├── speech4.txt
│   ├── speech5.txt
│   └── speech6.txt
├── test_dataset.json        # 25 evaluation questions with ground truth
├── speech.txt               # Input speech for Assignment 1
├── test_results.json        # Auto-generated evaluation outputs
├── results_analysis.md      # Auto-generated analysis summary
├── requirements.txt         # Python dependencies
└── README.md                # You are here

````

---

# 🧩 **Assignment 1 — RAG Prototype**

## ✔ Features

- Loads **speech.txt** (Ambedkar excerpt)
- Splits text into chunks
- Creates local embeddings using **sentence-transformers/all-MiniLM-L6-v2**
- Stores vectors in **ChromaDB**
- Uses **LCEL pipeline** with `ChatOllama` for generation
- Performs retrieval using Chroma retriever
- Interactive CLI loop for Q&A  
- Type **exit** to quit

---

# ▶ **Run Assignment 1 (main.py)**

### 1️⃣ **Create Virtual Environment**

```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
````

### 2️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3️⃣ **Install Ollama + Mistral**

```bash
ollama pull mistral
ollama serve
```

### 4️⃣ **Run RAG System**

```bash
python main.py
```

### 📝 Example Session

```
ChromaDB not found. Running ingestion...
RAG system ready!

Your question: What is the remedy for caste?
=== ANSWER ===
...
Your question: exit
Exiting...
```

---

# 🧪 **Assignment 2 — Evaluation Framework**

You must evaluate the RAG system on:

### 📄 **Document Corpus**

6 files in `/corpus`

### 📝 **Test Dataset**

25 Q&A pairs in `test_dataset.json`

---

## 🔬 Evaluation Includes

### **1. Retrieval Metrics**

* Hit Rate
* Mean Reciprocal Rank (MRR)
* Precision@K

### **2. Answer Quality Metrics**

* ROUGE-L
* BLEU
* Cosine Similarity
* Answer Relevance
* Faithfulness

### **3. Semantic Metrics**

* Embedding similarity (SentenceTransformer)

### **4. Chunking Strategies Compared**

* **Small**: ~250 chars
* **Medium**: ~550 chars
* **Large**: ~900 chars

### **5. Outputs**

* **test_results.json**
* **results_analysis.md**

---

# ▶ **Run Assignment 2 (evaluation.py)**

Make sure Ollama is running:

```bash
ollama serve
```

Run evaluation:

```bash
python evaluation.py
```

This will:

* Build 3 Chroma vector stores
* Run **75 LLM Q&A calls** (25 questions × 3 chunk sizes)
* Compute all metrics
* Save:

```
test_results.json
results_analysis.md
```

---

# 💡 **How the Evaluation Pipeline Works**

1. Load all 6 documents
2. Build three vector stores (small/medium/large chunks)
3. For each chunking strategy:

   * Retrieve top-K documents
   * Generate answer using LCEL chain
   * Compute all metrics
   * Save results
4. Compare chunk sizes
5. Produce human-readable summary

---

# 🧠 **Technologies Used**

### 📚 LangChain (LCEL)

* `ChatPromptTemplate`
* `RunnablePassthrough`
* `ChatOllama`
* `StrOutputParser`

### 🔎 Vector Search

* **ChromaDB**

### 🔤 Embeddings

* `sentence-transformers/all-MiniLM-L6-v2`

### 🤖 Local LLM

* **Ollama Mistral 7B**

### 📊 Evaluation Tools

* ROUGE
* BLEU (NLTK)
* Cosine similarity
* SentenceTransformer embeddings
* scikit-learn

---

# 📈 **Results Files**

### **test_results.json**

Contains retrieved sources, generated answers, and all metric scores.

### **results_analysis.md**

Contains final summary, best chunk size, and evaluation insights.

---

# 🏁 **Final Notes**

* Fully offline RAG + evaluation
* No API keys required
* Uses LCEL (not RetrievalQA) as requested
* Metrics and evaluation strictly follow assignment specification

---

# ✍️ **Author**

Submission for **Kalpit Pvt Ltd – AI Intern Hiring Task**
Developed by: **Devam Singh**

