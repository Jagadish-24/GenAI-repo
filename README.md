# Retrieval-Augmented Generation (RAG) System

A simple pipeline for building primitive document-based question-answering systems. This project demonstrates a complete end-to-end implementation of a RAG system with text processing (from a manually created pdf file), semantic embeddings (using nomic-embed-text model), and retrieval-augmented SLM (phi3:mini) inference.

---

## Project Overview

This repository contains a pipeline that transforms a PDF document which is in Question-Answer Format into a queryable knowledge base. The system uses semantic similarity and text processing techniques to retrieve relevant context and generate accurate, context-aware responses.

**Key Capabilities:**
- Document processing pipeline
- Text chunking with recursive strategies (Didnt use Langchain, since wanted to understand the core working)
- Semantic search using vector embeddings
- SLM-powered response generation with retrieval augmentation.

---

## Architecture & Pipeline

The system follows a 6-stage processing pipeline:

```
PDF Documents
    ↓
[0] PDF Processing → Extract raw text from PDFs
    ↓
[1] Text Cleaning → Remove headers, footers, OCR artifacts, whitespace normalization
    ↓
[2] Intelligent Chunking → Split text using recursive hierarchical strategies
    ↓
[3] Embedding Generation → Convert chunks to 768-dim semantic vectors
    ↓
[4] Vector Database → Store & index embeddings in ChromaDB
    ↓
[5] Retrieval & Generation → Semantic search + LLM inference
```

### Stage Details

| Stage | File | Purpose | Key Techniques |
|-------|------|---------|-----------------|
| **Extraction** | `0_pdf_processing.py` | Extract text from PDF documents | PyMuPDF, page-level processing |
| **Cleaning** | `1_clean.py` | Preprocess extracted text | Regex-based pattern removal, whitespace normalization |
| **Chunking** | `2_chunking.py` | Split text into processable units | Recursive character splitting, paragraph-based segmentation |
| **Embedding** | `3_embedding.py` | Generate semantic vectors | Nomic Embed Text (768-dim), Ollama integration, TF-IDF fallback (if ollama fails)|
| **Storage** | `4_vector_db.py` | Index embeddings for fast retrieval | ChromaDB persistent storage |
| **Retrieval** | `5_retreival.py` | Query processing & response generation | Semantic similarity search, relevance scoring, LLM inference |

---

## Technical Stack

### Core Technologies
- **Language**: Python 3.12
- **Vector Database**: ChromaDB (persistent local storage)
- **Embeddings**: Nomic AI's `nomic-embed-text` (137M parameters)
- **LLM**: Phi3:mini via Ollama
- **PDF Processing**: PyMuPDF
- **ML Libraries**: Scikit-learn (TF-IDF fallback)

### Key Features

#### Text Processing
- **Recursive Chunking**: Hierarchical splitting based on semantic boundaries (paragraphs → sentences → words)
- **Context Overlap**: 150-character overlap between chunks for contextual continuity
- **Pattern Removal**: Intelligent cleaning of page numbers, headers, footers, and OCR artifacts

#### Retrieval Mechanism
- **Semantic Search**: Vector similarity using cosine distance
- **Smart Re-ranking**: 
  - Boost scores for answer-like chunks
  - Penalize question-only chunks
  - Retrieve 2x top-k initially for re-ranking flexibility
- **Top-K Retrieval**: Configurable (default: 5 chunks per query)

#### Embedding Model
```
Model: nomic-embed-text
├── Context Length: 8,192 tokens
├── Vector Dimension: 768 (Matryoshka-compatible)
├── Parameters: ~137M
├── License: Apache 2.0 (commercial-friendly)
└── Performance: Optimized for speed and accuracy
```

#### LLM Model: Phi3:mini
```
Model: Phi3:mini
├── Size: ~3.8 billion parameters (lightweight) -> 2.2 GB smallest i could find
├── Performance: 80%+ of larger models on most tasks
├── Speed: Runs on CPU/consumer hardware (instant inference)
├── License: MIT (fully permissive, commercial-friendly)
├── Context Window: 128K tokens (excellent for RAG scenarios)
└── Quality: Trained on curated data (better than size suggests)
```

**Why Phi3:mini is ideal for RAG:**
- **Local Execution**: No API calls, no cloud dependency, complete privacy
- **Cost-Effective**: Zero per-token costs, one-time download using ollama
- **Speed**: Sub-second response times even on consumer hardware
- **Large Context**: 128K tokens allows more retrieved chunks without summarization
- **High Quality**: Consistently outperforms larger models in instruction-following tasks
- **Reproducible**: Deterministic inference, version-pinned for consistency
- **Production-Ready**: Used by enterprises for edge deployment and local AI

---

## Output Examples

The system generates structured retrieval results with high-quality context:

**Sample Query Results** (in `retrieval_examples/` and `sample_retrieved_chunks/`):
- `qa_what_is_decryption_20260815_203954.json`
- `qa_What_is_integrity__20260815_203615.json`
- `qa_what_is_LIN_20260815_203349.json`
- `retrieved_What_is_Confidentiality.json`

Each result includes:
- Query text
- Retrieved context chunks with relevance scores
- LLM-generated response
- Metadata and source references

---

## Quick Start

### Prerequisites
```bash
# Install dependencies
pip install pymupdf chromadb requests scikit-learn ollama

# Start Ollama server (separate terminal)
ollama serve

# Pull embedding model
ollama pull nomic-embed-text

# Pull LLM model
ollama pull phi3:mini
```

### Running the Pipeline

```bash
# 1. Process PDF and extract text
python 0_pdf_processing.py

# 2. Clean extracted text
python 1_clean.py

# 3. Create chunks
python 2_chunking.py

# 4. Generate embeddings
python 3_embedding.py

# 5. Build vector database
python 4_vector_db.py

# 6. Query and retrieve
python 5_retreival.py
```

### Data Flow
- Input: `nlp-book.pdf`
- Processing stages store intermediate results in `data/`:
  - `extracted.json` - Raw extracted text
  - `cleaned.json` - Preprocessed text
  - `chunked.json` - Text chunks with metadata
  - `embeddings.json` - Vector embeddings
  - `chroma_db/` - Persistent vector store
  - `retreival_results/` - Query results and responses

---

## Key Design Decisions

### Why Recursive Chunking?
- Preserves semantic boundaries (paragraphs → sentences → words)
- Avoids breaking meaning at arbitrary character limits
- Follows industry-standard approaches (LangChain RecursiveCharacterTextSplitter)

### Why Relevance Re-ranking?
- Raw similarity scores aren't always optimal
- Context: distinguishes answers from question-only chunks
- Boost/penalize logic improves retrieval quality

### Why Ollama + Local Models?
- No cloud dependency or API costs
- Complete privacy and reproducibility
- Full control over model versions
- Apache 2.0 license (commercial use allowed)

---

## Skills Demonstrated

- **NLP & Information Retrieval**: Semantic embeddings, vector similarity search, query processing
- **Data Pipeline Engineering**: Multi-stage ETL, format transformations, error handling
- **Python Development**: Clean architecture, modular code, type hints, configuration management
- **ML Operations**: Model integration, embedding storage, inference optimization
- **Database Design**: Vector database schema, persistence, query optimization
- **Text Processing**: Regex patterns, document parsing, preprocessing workflows
- **Software Engineering**: Version control, reproducibility, documentation

---

## Configuration & Customization

All main parameters are configurable at the top of each script:

- **Chunk size**: `chunk_size = 800` characters
- **Chunk overlap**: `chunk_overlap = 150` characters  
- **Top-K retrieval**: `top_k_retrievals = 5`
- **Embedding model**: `nomic-embed-text` (or alternative)
- **LLM model**: `phi3:mini`
- **Vector dimension**: 768

---

## Future Enhancements

- Multi-document support with source attribution
- Real-time indexing for growing document collections
- Hybrid search (combining BM25 + semantic search) --> P0 (TOP Priority)
- Web for deployment, basically wrap it with a streamlit/gradio interface --> P1 (Second Priority)
- Batch query processing

---

## License

Apache 2.0 - Suitable for commercial and personal use

---

**Built with Python | LLMs | Vector Databases | NLP**
