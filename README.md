# LangChain-YouTube-Video-RAG-Assistant

https://github.com/user-attachments/assets/d5e1aa4f-c9c6-4b8d-af58-0df3e5f902bb

A full-stack AI application that allows users to chat directly with any YouTube video. By extracting the video's transcript and leveraging Retrieval-Augmented Generation (RAG), this assistant answers questions, summarizes content, and explains concepts in real-time.

## ✨ Key Features
* **URL to Chat:** Simply paste a YouTube link to generate a dedicated knowledge base.
* **Multilingual Transcript Support:** Select between English, Hindi, or other available transcripts.
* **Real-Time Streaming:** Answers stream token-by-token for a fast, ChatGPT-like experience.
* **Conversation Memory:** The AI remembers previous questions within the same session for contextual learning.
* **Beautiful UI:** Built with Tailwind CSS, featuring automatic Markdown parsing for code blocks and bullet points.
* **Export Chat:** Save your conversation history as a `.txt` file with a single click.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, LangChain, OpenAI (GPT-4o-mini & Text Embeddings), FAISS (Vector Database), YouTube Transcript API.
* **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript, Marked.js (Markdown rendering).

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8+
* An OpenAI API Key
