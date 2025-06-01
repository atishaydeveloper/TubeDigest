import streamlit as st
import re
import os
import subprocess
import json
import glob
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile # For managing temporary files

# === CONFIGURATION (Defaults) ===
DEFAULT_LANG = "en"
DEFAULT_CHUNK_SIZE = 4000
AVAILABLE_OLLAMA_MODELS = ["phi3", "llama3", "mistral", "gemma:2b", "deepseek-coder","deepseek-r1"]

# === HELPER FUNCTIONS ===
def get_video_id_from_url(url):
    """Extracts video ID from various YouTube URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@st.cache_data
def get_video_metadata(video_url):
    """Fetches video title and uploader using yt-dlp."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--print-json", "-s", "--skip-download", video_url],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        json_str = result.stdout.strip()
        if not json_str.startswith('{'):
            json_str = json_str.split('\n')[0]
        data = json.loads(json_str)
        return {
            "title": data.get("title", "N/A"),
            "uploader": data.get("uploader", "N/A"),
            "duration_string": data.get("duration_string", "N/A"),
            "original_url": data.get("webpage_url", video_url)
        }
    except subprocess.CalledProcessError as e:
        st.error(f"Error fetching video metadata: {e.stderr}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding video metadata JSON. Output was: {result.stdout[:500]}...")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching metadata: {str(e)}")
        return None

# === STEP 1: DOWNLOAD CAPTIONS ===
@st.cache_data(show_spinner=False)
def download_captions(video_url, video_id, lang="en", download_dir="."):
    output_template = os.path.join(download_dir, f"{video_id}.%(ext)s")
    try:
        process = subprocess.run([
            "yt-dlp",
            "--write-auto-subs", # Try auto-subs first
            "--sub-lang", lang,
            "--skip-download",
            "--restrict-filenames",
            "-o", output_template,
            video_url
        ], capture_output=True, text=True, check=False, encoding='utf-8') # check=False to handle cases where only manual subs exist

        # If auto-subs download failed or didn't exist, try manual subs
        # A bit simplistic check, yt-dlp exit codes can be more nuanced
        # We primarily check if a file was created.
        
        vtt_file_found = False
        # Try expected name for auto-subs first
        expected_auto_vtt_file = os.path.join(download_dir, f"{video_id}.{lang}.vtt")
        if os.path.exists(expected_auto_vtt_file):
            vtt_file_found = True
            vtt_file_path = expected_auto_vtt_file
        else: # Glob for other possibilities (like .asr.vtt for auto)
            glob_pattern_auto = os.path.join(download_dir, f"{video_id}*.{lang}.vtt")
            vtt_files_auto = glob.glob(glob_pattern_auto)
            if vtt_files_auto:
                vtt_file_path = vtt_files_auto[0]
                vtt_file_found = True

        if not vtt_file_found:
            st.info(f"Auto-captions for '{lang}' not found or download failed. Trying to find manual captions...")
            # Attempt to download any available manual subtitles for the language
            process_manual = subprocess.run([
                "yt-dlp",
                "--write-subs", # For manual subs
                "--sub-lang", lang,
                "--skip-download",
                "--restrict-filenames",
                "-o", output_template,
                video_url
            ], capture_output=True, text=True, check=False, encoding='utf-8')

            # Check again for VTT files (manual subs often have simpler names)
            expected_manual_vtt_file = os.path.join(download_dir, f"{video_id}.{lang}.vtt") # Same pattern often
            if os.path.exists(expected_manual_vtt_file):
                vtt_file_path = expected_manual_vtt_file
                vtt_file_found = True
            else:
                glob_pattern_manual = os.path.join(download_dir, f"{video_id}*.{lang}.vtt")
                vtt_files_manual = glob.glob(glob_pattern_manual)
                if vtt_files_manual:
                    vtt_file_path = vtt_files_manual[0]
                    vtt_file_found = True

        if vtt_file_found:
            return vtt_file_path
        else:
            st.warning(f"Could not find any VTT captions for '{video_id}' in language '{lang}'.")
            st.warning(f"yt-dlp (auto-subs) output: STDOUT: {process.stdout[:300]}... STDERR: {process.stderr[:300]}...")
            if 'process_manual' in locals(): # if manual attempt was made
                st.warning(f"yt-dlp (manual-subs) output: STDOUT: {process_manual.stdout[:300]}... STDERR: {process_manual.stderr[:300]}...")
            return None
            
    except subprocess.CalledProcessError as e: # This might not be hit if check=False
        st.error(f"Error during caption download process: {e.stderr}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during caption download: {str(e)}")
        return None

# === STEP 2: CLEAN AND CONVERT VTT TO TXT ===
def clean_line(line):
    line = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", line)
    line = re.sub(r"</?c.*?>", "", line)
    line = re.sub(r"<[^>]+>", "", line)
    return line.strip()

@st.cache_data(show_spinner=False)
def convert_vtt_to_txt(vtt_filepath, output_dir="."):
    if not vtt_filepath or not os.path.exists(vtt_filepath):
        st.error("❌ VTT file not found for conversion.")
        return None
    base_name = os.path.splitext(os.path.basename(vtt_filepath))[0]
    txt_file = os.path.join(output_dir, f"{base_name}.txt")
    seen_lines = set()
    output_lines = []
    try:
        with open(vtt_filepath, "r", encoding="utf-8", errors="ignore") as vtt:
            for line in vtt:
                cleaned = clean_line(line)
                if not cleaned or '-->' in cleaned or cleaned.startswith(("WEBVTT", "Kind:", "Language:")):
                    continue
                if cleaned not in seen_lines:
                    seen_lines.add(cleaned)
                    output_lines.append(cleaned)
    except Exception as e:
        st.error(f"Error reading or processing VTT file {vtt_filepath}: {str(e)}")
        return None
    try:
        with open(txt_file, "w", encoding="utf-8") as txt:
            txt.write(' '.join(output_lines))
        return txt_file
    except Exception as e:
        st.error(f"Error writing TXT file {txt_file}: {str(e)}")
        return None

# === STEP 3: QUERY OLLAMA ===
@st.cache_data(show_spinner=False)
def query_ollama(prompt, model="phi3"):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt, text=True, capture_output=True, check=True, encoding='utf-8'
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        st.error(f"Error querying Ollama with model {model}: {e.stderr}")
        if "model not found" in e.stderr.lower():
            st.error(f"Ollama model '{model}' not found. Please pull it first: `ollama pull {model}`")
        return f"Error: Ollama query failed. {e.stderr}"
    except FileNotFoundError:
        st.error("Ollama command not found. Is Ollama installed and in your PATH?")
        return "Error: Ollama not found."
    except Exception as e:
        st.error(f"An unexpected error occurred while querying Ollama: {str(e)}")
        return f"Error: {str(e)}"

# === STEP 4: SUMMARIZE CHUNKS ===
def summarize_chunks(text, ollama_model, chunk_size=DEFAULT_CHUNK_SIZE, output_dir="."):
    if not text:
        st.warning("No text to summarize.")
        return None
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    summary_progress_bar = st.progress(0, text="Summarizing chunks...")
    for i, chunk in enumerate(chunks):
        # --- MODIFIED OLLAMA PROMPT ---
        prompt = (
            "You are a helpful assistant tasked with summarizing video transcript excerpts. "
            "Your goal is to extract the most important information, key topics, significant statements, decisions, or events mentioned. "
            "The summary should be concise and factual, focusing on the core content of this specific excerpt. "
            "Avoid conversational filler, introductions, or conclusions that are not part of the main substance. "
            "Just provide the summary of this excerpt:\n\n"
            f"{chunk}"
        )
        summary = query_ollama(prompt, ollama_model)
        if "Error:" in summary:
            st.error(f"Failed to summarize chunk {i+1}. Ollama error: {summary}")
            summaries.append(f"[Error summarizing chunk {i+1}: {summary}]")
        else:
            summaries.append(summary)
        summary_progress_bar.progress((i + 1) / len(chunks), text=f"Summarizing chunk {i+1}/{len(chunks)}...")
    summary_progress_bar.empty()
    combined_summary_file = os.path.join(output_dir, "video_notes_ollama_summary.txt")
    try:
        with open(combined_summary_file, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(summaries))
        return combined_summary_file
    except Exception as e:
        st.error(f"Error writing combined summary file: {str(e)}")
        return None

# === STEP 5: STRUCTURE NOTES WITH GEMINI ===
@st.cache_data(show_spinner=False)
def generate_markdown_notes_gemini(summary_text, gemini_api_key_to_use):
    if not summary_text:
        st.warning("No summary text provided to Gemini.")
        return None
    current_env_key = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = gemini_api_key_to_use
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2) # Slightly lower temp for factuality
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {str(e)}")
        if current_env_key: os.environ["GOOGLE_API_KEY"] = current_env_key
        elif "GOOGLE_API_KEY" in os.environ: del os.environ["GOOGLE_API_KEY"]
        return f"Error: Could not initialize Gemini. {str(e)}"
    
    # --- MODIFIED GEMINI PROMPT ---
    prompt = f"""
You are a highly skilled information synthesizer. Your task is to process the provided text, which is a collection of summaries from a video transcript, and transform it into comprehensive, well-structured notes in Markdown format. The goal is to create an overview that captures all important details, key discussions, events, and takeaways from the video, making it easy for someone to understand the video's content without watching it.

Video Content Summary:
{summary_text}

Instructions for Formatting the Notes:
1.  **Overall Summary:** Start with a concise `## Overall Video Summary` section that gives a bird's-eye view of the video's main subject and purpose.
2.  **Key Topics/Segments:** Create a `## Key Topics/Segments Discussed` section. Under this, use subheadings (e.g., `### Topic/Segment Title`) for each distinct part or major theme of the video. Under each subheading, use bullet points to list the main points, arguments, or information presented in that segment.
3.  **Significant Details & Information:** Include a `## Noteworthy Details & Information` section. Use bullet points to list specific facts, figures, important statements, examples, or any other crucial pieces of information mentioned throughout the video that don't fit neatly into the topics/segments or deserve special mention.
4.  **Main Conclusions/Outcomes (if applicable):** If the video presents conclusions, results, decisions, or calls to action, summarize them under `## Main Conclusions/Outcomes`. If not applicable, this section can be omitted.
5.  **Formatting:**
    *   Use Markdown for all formatting (headings, subheadings, bold text, bullet points).
    *   Ensure clarity, conciseness, and logical flow.
    *   Highlight key terms or names using **bold text** where appropriate.
    *   If the input text mentions errors (e.g., "[Error summarizing chunk...]"), briefly acknowledge that some information might be incomplete and proceed to structure the available content as best as possible.

Please generate the structured Markdown notes based on the video content summary provided.
"""
    try:
        result = llm.invoke(prompt)
        final_notes_content = result.content
    except Exception as e:
        st.error(f"Error invoking Gemini API: {str(e)}")
        final_notes_content = f"Error: Gemini API call failed. {str(e)}"
    finally:
        if current_env_key: os.environ["GOOGLE_API_KEY"] = current_env_key
        elif "GOOGLE_API_KEY" in os.environ and os.environ["GOOGLE_API_KEY"] == gemini_api_key_to_use:
            del os.environ["GOOGLE_API_KEY"]
    return final_notes_content

# === STREAMLIT UI ===
st.set_page_config(page_title="YouTube Video Notes Generator", layout="wide")
st.title("📝 YouTube Video to Structured Notes")
st.markdown("Extract transcript, summarize with Ollama, and structure notes with Gemini to get a comprehensive overview of any YouTube video.")

st.sidebar.header("⚙️ Configuration")
youtube_url = st.sidebar.text_input("Enter YouTube Video URL:", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")
selected_lang = st.sidebar.text_input("Caption Language (e.g., en, es, fr):", value=DEFAULT_LANG)
selected_ollama_model = st.sidebar.selectbox("Select Ollama Model for Summarization:", AVAILABLE_OLLAMA_MODELS, index=AVAILABLE_OLLAMA_MODELS.index("phi3") if "phi3" in AVAILABLE_OLLAMA_MODELS else 0)

load_dotenv()
env_gemini_key = os.getenv("GOOGLE_API_KEY", "")
gemini_api_key = st.sidebar.text_input(
    "Enter your Google Gemini API Key:",
    type="password",
    value="Add key",
    help="Get yours from Google AI Studio. Can be pre-filled from .env (GOOGLE_API_KEY)."
)

if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
temp_dir = st.session_state.temp_dir

if st.sidebar.button("🚀 Generate Notes", use_container_width=True):
    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
    elif not gemini_api_key:
        st.warning("Please enter your Google Gemini API Key.")
    else:
        video_id = get_video_id_from_url(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL or could not extract Video ID.")
        else:
            st.info(f"Processing Video ID: {video_id}")
            
            metadata = get_video_metadata(youtube_url)
            if metadata:
                st.subheader(f"🎬 Video: {metadata['title']}")
                st.caption(f"🎙️ Uploader: {metadata['uploader']} | ⏱️ Duration: {metadata['duration_string']}")
                st.markdown(f"🔗 [Watch on YouTube]({metadata['original_url']})")
            else:
                st.warning("Could not retrieve video metadata. Proceeding with transcript processing...")

            final_notes_md_content = None
            ollama_summary_content = ""

            with st.status("Processing steps...", expanded=True) as processing_status:
                processing_status.update(label="Step 1: Downloading captions...", state="running")
                vtt_file_path = download_captions(youtube_url, video_id, selected_lang, download_dir=temp_dir)
                if not vtt_file_path:
                    st.error("Failed to download captions. Cannot proceed.")
                    processing_status.update(label="Caption download failed.", state="error", expanded=False)
                    st.stop()
                processing_status.update(label="Step 1: Captions downloaded. ✅", state="complete")

                processing_status.update(label="Step 2: Converting VTT to TXT...", state="running")
                txt_file_path = convert_vtt_to_txt(vtt_file_path, output_dir=temp_dir)
                if not txt_file_path:
                    st.error("Failed to convert VTT to TXT. Cannot proceed.")
                    processing_status.update(label="VTT conversion failed.", state="error", expanded=False)
                    st.stop()
                st.success(f"✅ Transcript extracted: {os.path.basename(txt_file_path)}")
                processing_status.update(label="Step 2: VTT converted to TXT. ✅", state="complete")

                transcript_text = ""
                try:
                    with open(txt_file_path, "r", encoding="utf-8") as f:
                        transcript_text = f.read()
                    if not transcript_text.strip():
                        st.warning("The extracted transcript is empty. Notes might be minimal or incorrect.")
                except Exception as e:
                    st.error(f"Error reading transcript file: {str(e)}")
                    processing_status.update(label="Transcript reading failed.", state="error", expanded=False)
                    st.stop()

                if transcript_text.strip():
                    processing_status.update(label=f"Step 3: Summarizing with Ollama ({selected_ollama_model})...", state="running")
                    ollama_summary_file = summarize_chunks(transcript_text, selected_ollama_model, chunk_size=DEFAULT_CHUNK_SIZE, output_dir=temp_dir)
                    if not ollama_summary_file:
                        st.error("Failed to summarize chunks with Ollama. Attempting to proceed with raw transcript for Gemini.")
                        processing_status.update(label="Ollama summarization failed.", state="error")
                        ollama_summary_content = "Ollama summarization failed. Using raw transcript for final notes."
                    else:
                        st.success(f"✅ Ollama summary generated: {os.path.basename(ollama_summary_file)}")
                        processing_status.update(label="Step 3: Ollama summarization complete. ✅", state="complete")
                        try:
                            with open(ollama_summary_file, "r", encoding="utf-8") as f:
                                ollama_summary_content = f.read()
                        except Exception as e:
                            st.error(f"Error reading Ollama summary file: {str(e)}")
                            ollama_summary_content = f"Error reading summary file. Using raw transcript. ({str(e)})"
                else:
                    ollama_summary_content = "Transcript was empty. No content to summarize with Ollama."
                    st.warning(ollama_summary_content)
                    processing_status.update(label="Step 3: Skipped Ollama (empty transcript).", state="complete")

                processing_status.update(label="Step 4: Generating structured notes with Gemini...", state="running")
                input_for_gemini = ollama_summary_content if ollama_summary_content.strip() and "Ollama summarization failed" not in ollama_summary_content else transcript_text
                
                if not input_for_gemini.strip():
                    final_notes_md_content = "Both transcript and Ollama summary are effectively empty. Cannot generate notes."
                    st.error(final_notes_md_content)
                else:
                    if "Ollama summarization failed" in ollama_summary_content or not ollama_summary_content.strip():
                         st.info("Ollama summary was empty or failed. Using raw transcript for Gemini.")
                    final_notes_md_content = generate_markdown_notes_gemini(input_for_gemini, gemini_api_key)

                if final_notes_md_content and "Error:" not in final_notes_md_content:
                    st.success("✅ Final notes generated by Gemini!")
                    processing_status.update(label="Step 4: Gemini notes generated. ✅", state="complete")
                elif final_notes_md_content:
                    st.error("Failed to generate notes with Gemini.")
                    processing_status.update(label="Gemini notes generation failed.", state="error", expanded=False)
                else:
                    st.error("Gemini did not return any content.")
                    processing_status.update(label="Gemini notes generation failed.", state="error", expanded=False)

            if final_notes_md_content and "Error:" not in final_notes_md_content:
                st.subheader("📑 Generated Video Overview & Notes")
                st.markdown(final_notes_md_content)
                final_notes_filename = f"{video_id}_video_notes.md"
                final_notes_filepath = os.path.join(temp_dir, final_notes_filename)
                try:
                    with open(final_notes_filepath, "w", encoding="utf-8") as f:
                        f.write(final_notes_md_content)
                    with open(final_notes_filepath, "rb") as fp:
                        st.download_button(
                            label="📥 Download Notes (.md)", data=fp, file_name=final_notes_filename,
                            mime="text/markdown", use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error saving final notes for download: {str(e)}")
            elif final_notes_md_content:
                 st.error(f"Could not display notes due to an error: {final_notes_md_content}")
            else:
                 st.error("No final notes were generated to display.")

st.sidebar.markdown("---")
st.sidebar.markdown("Get a structured overview of any YouTube video.")