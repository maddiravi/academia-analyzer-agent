# 📘 Academia Analyzer — Multi-Agent RAG System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-green)
![Status](https://img.shields.io/badge/Status-MVP--Complete-brightgreen)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Tests](https://img.shields.io/badge/Validation-RAG--Backed-success)

**Academia Analyzer** is a multi-agent system built to autonomously process academic research papers and generate structured executive summaries using Retrieval-Augmented Generation and LangGraph orchestration.  
This project serves as the **Module 2 deliverable** for the Agentic AI certification, demonstrating applied skills in:

- Multi-Agent AI Coordination  
- Retrieval-Augmented Generation (RAG)  
- Operational Tooling and Orchestration  
- Applied NLP Extraction Pipelines  

---

## 🚀 What This System Solves

The exponential growth of academic literature has created **research fatigue**. Manually extracting:

- Hypotheses  
- Methodologies  
- Findings  

…from long-form research papers is slow, inconsistent, and often overwhelming.

**Academia Analyzer automates that entire workflow.**

---

## ⚙️ Architecture Overview

The workflow is executed using a controlled **LangGraph pipeline**, ensuring stability, deterministic role boundaries, and traceable state transitions.

### 🧠 Agent Roles

| Agent | Responsibility | Integrated Tools |
|-------|--------------|----------------|
| **DocumentIngestorAgent** | Reads and chunks files, builds semantic vector index | `PyPDF`, `TextLoader`, **FAISS** |
| **ThesisExtractorAgent** | Extracts hypothesis, research focus, technical keywords | NLP + `nltk` |
| **InsightSynthesizerAgent** | Generates final structured summary grounded by retrieval | OpenRouter LLM + Pydantic schema |

### 🔄 Workflow (LangGraph Flow)

```
User Upload → Ingest → Extract → Retrieve → Synthesize → Validated Summary Output
```

1. **Upload Input File** in UI  
2. **Ingestion:** Text split into 1500-token chunks, indexed using FAISS  
3. **Extraction:** Keyword + hypothesis extraction using NLP  
4. **Synthesis:** Final structured summary generated via RAG + JSON schema

---

## 🧪 Technical Validation & Configuration

Academia Analyzer ensures summarization accuracy by grounding generation with retrieval-based constraints.

### 🔍 RAG Configuration

| Parameter | Value | Reason |
|----------|-------|--------|
| Chunk Size | **1500 tokens** | Ensures full academic logic remains intact |
| Overlap | **250 tokens** | Maintains contextual continuity |
| Embedding Model | `all-MiniLM-L6-v2` | Efficient + strong semantic matching |
| LLM Model | `GPT-3.5-Turbo` (OpenRouter) | Reliable structured summarization |

### 🧾 Code Transparency

🗂 Source Code:  
👉 `https://github.com/maddiravi/academia-analyzer-agent`

All agents, schemas, and pipelines are documented and reproducible.

---

## 💻 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/maddiravi/academia-analyzer-agent
cd academia-analyzer-agent
pip install -r requirements.txt
```

### 2️⃣ Configure Environment

Create a `.env` file:

```env
OPENROUTER_API_KEY="your_key_here"
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📊 Using the System

Once running:

1. Drag & drop any research PDF/TXT file  
2. Execute the analysis pipeline  
3. View summary, extracted keywords, and structured outputs  

(Optional Screenshot Section)
> *(Add screenshot of generated summary page here)*

---

## 🚧 Limitations & Next Steps

### Known Limitations

- Currently operates on a **single document at a time**
- Does not yet extract citations or bibliography formatting
- Only supports text-based research formats

### Planned Enhancements

- 📑 Multi-Paper Comparison Mode  
- 🔖 Automatic Citation Extraction (APA / IEEE)  
- 🧬 Domain-Adaptive Summarization Tuning  

---

## 🔖 Recommended Tags

```
#AI #RAG #LangGraph #MultiAgentSystems #Python #ResearchAutomation #NLP #FAISS #LLM #AcademicTools #Streamlit #OpenSource #AIDevelopment
```

---

