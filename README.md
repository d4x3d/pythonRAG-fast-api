# 🤖 Python RAG FastAPI Project

A Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain,Nomic,and groq LLM . Supports both web content and PDF document processing.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- git

### 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/d4x3d/pythonRAG-fast-api
cd  pythonRAG-fast-api
```

2. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # For Unix/MacOS
# OR
.venv\Scripts\activate  # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Copy `.env.example` to `.env`

## ⚙️ Configuration

You'll need API keys from:
- Groq
- Nomic
- LangSmith

## 🏃‍♂️ Running the Application

### Web Content Analysis
```bash
python rag.py
```

### PDF Document Analysis
1. Place your PDF files in the `pdf` directory
2. Run:
```bash
python pdfrag.py
```
The PDF analyzer will:
- Show available PDF files in the directory
- Let you select a file
- Process the document(chunck,embedd,store)
- Allow you to ask questions about its content(and allow you to query)

## 🔑 API Keys

Make sure to obtain API keys from:
- [Groq](https://console.groq.com)
- [Nomic](https://atlas.nomic.ai/)
- [LangSmith](https://smith.langchain.com)


## 📁 Project Structure

```
pythonRAG-fast-api/
├── pdf/                  # Directory for PDF files
├── rag.py               # Web content RAG implementation
├── pdfrag.py            # PDF processing RAG implementation
├── requirements.txt     # Project dependencies
└── .env                 # Environment variables
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
