import re
import os
import subprocess
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Markdown, display

# === CONFIGURATION ===
video_url = "https://www.youtube.com/watch?v=1HuD2ryeOLg"
lang = "en"
chunk_size = 3000
model = "deepseek-r1"

# === STEP 1: DOWNLOAD CAPTIONS ===
def download_captions(video_url, lang="en"):
    subprocess.run([
        "yt-dlp",
        "--write-auto-subs",
        "--sub-lang", lang,
        "--skip-download",
        video_url
    ])

# === STEP 2: CLEAN AND CONVERT VTT TO TXT ===
def clean_line(line):
    line = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", line)
    line = re.sub(r"</?c>", "", line)
    return line.strip()

def convert_vtt_to_txt(video_url, lang="en"):
    video_id = video_url.split("v=")[-1].split("&")[0]
    vid = f"[{video_id}]"
    vtt_file = next((f for f in os.listdir() if f.endswith(f"{vid}.{lang}.vtt")), None)

    if not vtt_file:
        print("❌ VTT file not found.")
        return None

    txt_file = vtt_file.replace(".vtt", ".txt")
    seen_lines = set()
    output_lines = []

    with open(vtt_file, "r", encoding="utf-8", errors="ignore") as vtt:
        for line in vtt:
            line = clean_line(line)
            if not line or '-->' in line or line.startswith(("WEBVTT", "Kind:", "Language:")):
                continue
            if line not in seen_lines:
                seen_lines.add(line)
                output_lines.append(line)

    with open(txt_file, "w", encoding="utf-8") as txt:
        txt.write(' '.join(output_lines))

    print(f"✅ Transcript saved to: {txt_file}")
    return txt_file

# === STEP 3: QUERY OLLAMA ===
def query_ollama(prompt, model="deepseek-r1"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

# === STEP 4: SUMMARIZE CHUNKS ===
def summarize_chunks(text, chunk_size=3000, model="deepseek-r1"):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []

    for i, chunk in enumerate(chunks):
        prompt = f"Summarize this educational video transcript:\n\n{chunk}"
        summary = query_ollama(prompt, model)
        print(f"✅ Chunk {i+1} summarized.")
        summaries.append(summary)

    with open("video_notes.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(summaries))
    return "video_notes.txt"

# === STEP 5: STRUCTURE NOTES WITH GEMINI ===
def generate_markdown_notes(summary_file):
    with open(summary_file, "r", encoding="utf-8") as f:
        text = f.read()

    load_dotenv()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    prompt = f"""
You are an expert student who knows how to make effective, organized, and clear notes that help in understanding and retaining important information quickly. Your task is to rewrite the given concise summary from an educational video into well-structured, readable, and efficient study notes in Markdown format.

Instructions:
- **Focus on key takeaways**
- **Organize under headings**
- **Ensure clarity and conciseness**
- **Highlight important points**
- **Use bullet points and markdown structure**

Input Text:

{text}

Output format:
## Overview

## Key Concepts

## Main Steps/Processes

## Key Insights/Takeaways
"""

    result = llm.invoke(prompt)
    display(Markdown(result.content))
    with open("final_notes.md", "w", encoding="utf-8") as f:
        f.write(result.content)
    print("✅ Final notes saved to final_notes.md")

# === PIPELINE ===
if __name__ == "__main__":
    download_captions(video_url, lang)
    txt_file = convert_vtt_to_txt(video_url, lang)

    if txt_file:
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        summary_file = summarize_chunks(text, chunk_size, model)
        generate_markdown_notes(summary_file)
