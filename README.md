# 🌌 MCP AI Chat – SMART ORCHESTRATOR

A high-performance, multipurpose AI workstation that integrates **Text, Vision, Image Generation, and Data Science** tools into a single, sleek web interface. This project is designed to be free-to-use by combining premium local models with powerful public APIs.

---

## ✨ Key Features

### 1. 🤖 Smart Chat
- **Unlimited & Free**: Powered by Pollinations AI for conversational intelligence.
- **Context Aware**: Remembers your chat history and handles multiple sessions.
- **Rename & Manage**: Organize your workspace by naming your chat sessions.

### 2. 🎨 Creativity & Graphics
- **AI Image Generation**: Type what you want to see, and the AI will generate high-quality images.
- **Smart Prompting**: Automatically refines your simple ideas into detailed artistic prompts.


### 3. 📂 File Intelligence
- **PDF Extraction**: Reads and analyzes large PDF documents.
- **Code & Text**: Supports snippets and text file ingestion for analysis.

---

## 🛠️ Installation

### 1. Prerequisites
- **Python 3.10+** (Recommended)
- **GPU** (Optional, but recommended for faster Vision processing)

### 2. Setup
Clone the project and install the dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

To launch the orchestrator:
```bash
python api.py
```

- **URL**: [http://127.0.0.1:8001](http://127.0.0.1:8001)

> **Note**: On the first run, the system will download the local vision model (~500MB) if you use the image explanation feature. This happens only once.

---

## 🧠 Tech Stack
- **Framework**: FastAPI (Backend) / Vanilla CSS (Modern Dark UI)
- **Logic**: Python
- **Vision**: Transformers (BLIP Model)
- **Data**: Pandas / Matplotlib
- **Images**: PIL (Pillow)
- **APIs**: Pollinations (Text/Image Gen)

---

## 📜 License
This project is open-source and free for development.
