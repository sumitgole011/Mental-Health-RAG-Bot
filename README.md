# 🧠 Mental Health Buddy — AI-Powered RAG Chatbot

<p align="center">
  <strong>A conversational AI companion designed to provide empathetic responses using knowledge retrieval, large language models, and speech interaction.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge" alt="LangChain"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Database-2E8B57?style=for-the-badge" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Groq-LLM%20Integration-F55036?style=for-the-badge" alt="Groq"/>
</p>

## 📌 Overview

**Mental Health Buddy** is an AI-powered conversational application built using Python, Flask, LangChain, and Groq. It is designed to provide supportive, empathetic conversations while retrieving relevant information from a document-based knowledge source.

The application uses a **Retrieval-Augmented Generation (RAG)** pipeline to combine contextual information from local documents with language-model responses. It also maintains conversation history during a running session and can speak generated responses aloud using text-to-speech.

The project explores how AI, semantic retrieval, and natural language processing can be integrated into a conversational support application.

## ✨ Key Features

- 💬 **Conversational Interface:** Chat with an AI companion through a web interface.
- 💜 **Empathetic Responses:** Uses a customized prompt to encourage supportive and approachable communication.
- 📚 **Knowledge-Based Retrieval:** Loads local documents to provide additional context for responses.
- 🧠 **RAG Architecture:** Retrieves relevant document passages before generating answers.
- 🔎 **Semantic Search:** Uses ChromaDB with Hugging Face sentence embeddings.
- 🤖 **LLM Integration:** Uses Groq's hosted Gemma 2 9B Instruct model through LangChain.
- 🗂️ **Conversation History:** Maintains message history during the active application session.
- 🔊 **Text-to-Speech:** Speaks generated responses using `pyttsx3`.
- 🎙️ **Speech Interaction Prototype:** Includes a separate experimental script with speech recognition and tone selection.
- 🌐 **Flask Backend:** Exposes a chat endpoint that returns AI-generated responses as JSON.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Flask | Web framework and API routes |
| LangChain | RAG pipeline and model orchestration |
| Groq | Hosted language-model inference |
| Gemma 2 9B Instruct | Configured conversational model |
| ChromaDB | Vector storage and retrieval |
| Hugging Face Embeddings | Semantic representation of text |
| Sentence Transformers | Embedding model |
| pyttsx3 | Text-to-speech output |
| SpeechRecognition | Speech-input prototype |
| HTML and CSS | Chat interface |
| python-dotenv | Environment variable management |

## 🏗️ System Architecture

The main application follows this retrieval-augmented workflow:

1. **Document Loading:** Loads knowledge documents from a local directory.
2. **Text Preprocessing:** Splits documents into smaller chunks with overlapping context.
3. **Embedding Generation:** Converts chunks into vectors using the configured embedding model.
4. **Vector Indexing:** Stores document embeddings in ChromaDB.
5. **User Input:** Receives a message through the web interface.
6. **Context Retrieval:** Retrieves up to two relevant document chunks using maximum marginal relevance (MMR).
7. **Response Generation:** Sends the retrieved context, user message, and conversation history to the Groq-hosted model.
8. **Speech Output:** Uses text-to-speech to read the response aloud.
9. **API Response:** Returns the generated message to the frontend as JSON.

## 🔄 RAG Workflow

```text
Local Knowledge Documents
          |
          v
   Document Loading
          |
          v
    Text Chunking
          |
          v
 Hugging Face Embeddings
          |
          v
   ChromaDB Vector Store
          |
          v
      User Message
          |
          v
  Relevant Context Retrieval
          |
          v
 Context + Question + History
          |
          v
     Groq / Gemma Model
          |
          v
   Generated AI Response
          |
       +--+--+
       |     |
       v     v
   Web UI  Text-to-Speech
```

## 📂 Project Structure

```text
Mental-Health-RAG-Bot/
│
├── app.py
├── speaking_bot.py
├── requirements.txt
├── expreriment.ipynb
│
├── Doc/
│   ├── I_am_not_sick_excerpt.txt
│   └── The_little_book_of_mental_health_-_version_6.txt
│
├── templates/
│   └── index.html
│
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later recommended
- Git
- pip
- A Groq API key

### Step 1: Clone the repository

```bash
git clone https://github.com/sumitgole011/Mental-Health-RAG-Bot.git
cd Mental-Health-RAG-Bot
```

### Step 2: Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your actual Groq API key. Never commit API keys or other credentials to GitHub.

### Step 5: Verify the knowledge-base directory

The application currently loads documents from `Peoject_2/Doc`, while the repository contains a root-level `Doc/` directory.

Update the loader path in `app.py` to match the actual directory:

```python
loader = DirectoryLoader("Doc")
```

Ensure the required documents are present before running the application.

### Step 6: Run the chatbot

```bash
python app.py
```

Open the application in your browser:

[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## 🔌 API Documentation

### Chat Endpoint

**Route:** `/chat`

**Method:** `POST`

**Request Content-Type:** `application/json`

Example request:

```json
{
  "message": "I've been feeling stressed lately."
}
```

Example response format:

```json
{
  "response": "I'm sorry things have been stressful. Would you like to talk about what's been weighing on you?"
}
```

The response above is illustrative; actual responses are generated by the configured language model.

### Home Endpoint

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Renders the chat interface |
| POST | `/chat` | Processes a user message and returns a response |

## 🎙️ Experimental Speech Assistant

The repository also includes `speaking_bot.py`, an experimental voice-based assistant that incorporates:

- Speech input through the SpeechRecognition library
- Emotion or response-tone classification using a structured output schema
- Context retrieval from local documents
- Conversational response generation
- Text-to-speech playback

This script is separate from the Flask web application and may require additional dependencies and configuration before it runs successfully.

## 🧠 Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- Text embedding models
- Document chunking and contextual retrieval
- LangChain runnable pipelines
- Large Language Model integration
- Flask web application development
- Conversation history management
- Text-to-speech integration
- Speech recognition and structured output parsing

## 🔮 Future Enhancements

- 💬 Improve multi-turn conversation memory.
- 🎤 Add integrated
