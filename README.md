# 🎥 YouTube Video Transcript Summarizer

A Python tool that downloads YouTube video transcripts, processes and summarizes them using the **Ollama LLM**, and generates structured, study-friendly notes using **Gemini (Google Generative AI)**.

---

## 🚀 Features

- 📥 **Downloads auto-generated captions** from any YouTube video.
- 🧹 **Cleans & deduplicates** the `.vtt` subtitle file.
- 🧩 **Chunks** the transcript and summarizes each part using `ollama` (locally).
- 📝 **Formats** the summarized output into organized Markdown notes using **Gemini via LangChain**.
- 💾 **Saves** the final notes as `final_notes.md`.

---

## ⚙️ Requirements

- Python 3.9+
- [`yt-dlp`](https://github.com/yt-dlp/yt-dlp)
- [`ollama`](https://ollama.com/) (installed and model pulled, e.g., `deepseek-r1`)
- Gemini API Key (via [Google AI Studio](https://aistudio.google.com/app/apikey))

Install Python dependencies:
```bash
pip install yt-dlp langchain-google-genai python-dotenv
```

# 📦 Installation & Setup

### Clone the repository:

```bash
git clone https://github.com/atishaydeveloper/Youtube_Videos_to_Notes.git
cd youtube-transcript-summarizer
```
## 🤖 Models Used

- **Ollama**: Local LLM (e.g., DeepSeek, Mistral, etc.) for summarization
- **Gemini**: Google’s powerful generative model for final notes formatting

---

## 🧠 Use Cases

- Quick learning from long educational videos
- Making structured revision notes
- Distilling online courses or tech talks
- Preparing summaries of tutorials

