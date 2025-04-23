import streamlit as st
import yt_dlp
from pydub import AudioSegment
import whisper
import os
import google.generativeai as genai
import re
import torch
from urllib.parse import parse_qs, urlparse
import tempfile
import ffmpeg

# --- Configuration ---
st.set_page_config(page_title="AI Creative Analysis", layout="wide")

# Custom CSS for GoMarble style
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }
    .main-container {
        display: flex;
        gap: 30px;
    }
    .video-container {
        width: 400px;
        background: #f5f5f5;
        border-radius: 8px;
        padding: 10px;
    }
    .analysis-container {
        flex-grow: 1;
        padding: 20px;
    }
    .main-score {
        text-align: center;
        margin: 20px 0;
    }
    .score-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        border: 5px solid;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        font-size: 40px;
        font-weight: bold;
    }
    .score-label {
        font-size: 16px;
        margin-top: 10px;
        color: #333;
    }
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-score {
        color: #666;
        font-size: 18px;
    }
    .what-works {
        background-color: #f0f9f4;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .what-not-works {
        background-color: #fff1f0;
        padding: 20px;
        border-radius: 8px;
    }
    .header-buttons {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
    }
    .share-button, .try-button {
        padding: 8px 16px;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: white;
        cursor: pointer;
    }
    .expert-button {
        padding: 8px 16px;
        border-radius: 4px;
        background: #004d40;
        color: white;
        border: none;
        cursor: pointer;
    }
    .copy-button {
        padding: 4px 8px;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: white;
        cursor: pointer;
        font-size: 14px;
    }
    .sidebar-option {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .item-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
        gap: 8px;
    }
    .item-icon {
        min-width: 20px;
    }
    .item-content {
        flex-grow: 1;
    }
    .actionable {
        color: #1a73e8;
        margin-top: 5px;
    }
    .divider {
        height: 1px;
        background: #eee;
        margin: 20px 0;
    }
    iframe {
        width: 100%;
        aspect-ratio: 9/16;
        border: none;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Gemini API Configuration ---
try:
    GEMINI_API_KEY = "AIzaSyCm72WohnOlHU18K68bNft1aFKAnw9Cw-A"
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# --- Helper Functions ---

def save_uploaded_file(uploaded_file):
    """Saves the uploaded file to a temporary location and returns the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def extract_audio_from_video(video_path):
    """Extracts audio from a video file and saves it as MP3."""
    output_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    try:
        # Ensure any existing file with the same name is removed (though NamedTemporaryFile should handle uniqueness)
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)

        (   ffmpeg
            .input(video_path)
            .output(output_audio_path, format='mp3', acodec='libmp3lame', audio_bitrate='192k')
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        if os.path.exists(output_audio_path):
             return output_audio_path
        else:
            st.error("Audio extraction failed: Output file not found.")
            return None

    except ffmpeg.Error as e:
        st.error(f"Error extracting audio using ffmpeg: {e.stderr.decode()}")
        st.warning("Please ensure FFmpeg is installed and accessible in your system's PATH.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during audio extraction: {e}")
        return None

def download_youtube_audio_yt_dlp(url):
    """Downloads audio from a YouTube URL and saves it as MP3."""
    output_file = "audio.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noplaylist': True,
        'nocheckcertificate': True,
    }
    try:
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except OSError as e:
                st.warning(f"Could not remove previous audio file: {e}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if os.path.exists(output_file):
            return output_file
        else:
            for f in os.listdir('.'):
                if f.startswith("audio.") and f.endswith(('.mp3', '.m4a', '.webm', '.ogg')):
                    os.rename(f, output_file)
                    return output_file
            st.error("Audio download finished, but the expected output file 'audio.mp3' was not found.")
            return None
    except yt_dlp.utils.DownloadError as e:
        if 'ffprobe and ffmpeg not found' in str(e):
            st.error("Error: FFmpeg not found. Please install FFmpeg and add it to your system's PATH.")
            st.error("Download FFmpeg from: https://ffmpeg.org/download.html or https://www.gyan.dev/ffmpeg/builds/")
        else:
            st.error(f"Error downloading video: {e}. Please check the URL and network connection.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during download: {e}")
        return None

@st.cache_resource
def load_whisper_model(model_size="large-v3"):
    """Loads the Whisper model with improved error handling."""
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using {device.upper()} for transcription")
        
        model = whisper.load_model(model_size, device=device)
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model '{model_size}': {e}")
        st.warning("Ensure you have PyTorch installed. You can install it via: pip install torch torchvision torchaudio")
        return None

@st.cache_data
def transcribe_audio(filepath, _whisper_model, language=None):
    """Transcribes the audio file using Whisper with improved settings."""
    if not filepath or not os.path.exists(filepath):
        st.error(f"Audio file not found at path: {filepath}")
        return None
    if not _whisper_model:
        st.error("Whisper model not loaded.")
        return None
    try:
        # Use improved transcription settings
        result = _whisper_model.transcribe(
            filepath,
            fp16=False,  # More stable on CPU
            language=language,  # Use specified language if provided
            verbose=True,  # Show progress
            task="transcribe",  # Force transcription mode
            temperature=0.0,  # More deterministic output
            best_of=5,  # Better quality
            beam_size=5  # Better quality
        )
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

def parse_gemini_analysis(text):
    """Parses the Gemini response text to extract score, works, and does not work."""
    score = None
    works = []
    does_not_work = []

    try:
        # Extract Score
        score_match = re.search(r"Score:\s*(\d+)\s*/\s*100", text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))

        # Extract Works
        works_match = re.search(r"Works:(.*?)(DoesNotWork:|What doesn't work:|$)", text, re.IGNORECASE | re.DOTALL)
        if works_match:
            works_text = works_match.group(1).strip()
            works = [line.strip('- ').strip() for line in works_text.split('\n') if line.strip() and line.strip().startswith('-')]

        # Extract Does Not Work
        does_not_work_match = re.search(r"(?:DoesNotWork:|What doesn't work:)(.*?)$", text, re.IGNORECASE | re.DOTALL)
        if does_not_work_match:
            does_not_work_text = does_not_work_match.group(1).strip()
            does_not_work = [line.strip('- ').strip() for line in does_not_work_text.split('\n') if line.strip() and line.strip().startswith('-')]

    except Exception as e:
        st.error(f"Error parsing Gemini analysis: {e}")
        return score, text.split('\n'), []

    if score is None and not works and not does_not_work:
        st.warning("Could not parse the Gemini response structure. Displaying raw output.")
        return None, text.split('\n'), []

    return score, works, does_not_work

def analyze_with_gemini(transcript):
    """Analyzes the transcript using the Gemini API."""
    prompt = f"""
You are an AI Creative Analyst specializing in evaluating video ad transcripts for effectiveness, particularly for a wellness audience. Analyze the following transcript:

"{transcript}"

Provide your analysis in the following format, strictly adhering to this structure:

Score: [Overall score out of 100]
Works:
- [Point 1 about what works]
- [Point 2 about what works]
...
DoesNotWork:
- [Point 1 about what doesn't work]
- [Point 2 about what doesn't work]
...

Focus your analysis on:
- Clarity of message
- Target audience resonance (assume wellness focus)
- Benefit articulation
- Call to action effectiveness (if present)
- Overall engagement potential based *only* on the text.
Ensure the output starts exactly with "Score:", followed by "Works:", and then "DoesNotWork:".
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

def get_video_id(url):
    """Extract video ID from various YouTube URL formats."""
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    
    parsed_url = urlparse(url)
    if 'youtube.com' in parsed_url.netloc:
        if 'shorts' in parsed_url.path:
            return parsed_url.path.split('/')[-1]
        return parse_qs(parsed_url.query).get('v', [None])[0]
    return None

def analyze_content(transcript, industry="Wellness", target_age=12):
    """Analyzes the content using Gemini API with industry and age context."""
    prompt = f"""
Analyze this video advertisement transcript considering the target audience is the **{industry} industry** and the target age group is around **{target_age} years old**. Provide a detailed analysis in exactly this format, generating scores based on the context:

Overall Score: [Generate score]/100

Hook (Score: [Generate score]/50):
What works:
- [List specific elements that work well for the {industry} industry and age {target_age}, be detailed]
- [Include at least 3-4 detailed points]

What does not work:
- [List specific issues with actionable improvements tailored for {industry} and age {target_age}]
- [Include at least 3-4 detailed points]
🎯 Actionable: [Specific improvement suggestion tailored to {industry} and age {target_age}]

Script (Score: [Generate score]/50):
What works:
- [List effective script elements for {industry} audience, age {target_age}, be detailed]
- [Include at least 3-4 detailed points]

What does not work:
- [List script issues with actionable improvements for {industry}, age {target_age}]
- [Include at least 3-4 detailed points]
🎯 Actionable: [Specific improvement suggestion for {industry} messaging, age {target_age}]

Visuals (Score: [Generate score]/50):
What works:
- [List effective visual elements for {industry}, age {target_age}, be detailed]
- [Include at least 3-4 detailed points]

What does not work:
- [List visual issues with actionable improvements for {industry}, age {target_age}]
- [Include at least 3-4 detailed points]
🎯 Actionable: [Specific improvement suggestion for {industry} visuals, age {target_age}]

**Important Instructions:**
- Generate scores for Overall, Hook, Script, and Visuals based on the transcript, target industry ({industry}), and target age ({target_age}).
- Ensure each section has detailed, industry-specific analysis with at least 3-4 points.
- Each "does not work" item must have an actionable improvement prefixed with "🎯 Actionable:".

Transcript: {transcript}
"""
    try:
        response = gemini_model.generate_content(prompt)
        # Add a check for potential safety blocks or empty responses
        if not response.parts:
             st.error("Analysis failed: Received an empty response from the AI. This might be due to safety filters or an issue with the request.")
             return None
        return response.text
    except Exception as e:
        # Catch potential exceptions like blocked prompts
        st.error(f"Error in analysis: {e}")
        # Log the error for debugging if needed
        # print(f"Gemini API Error: {e}") 
        return None

def parse_analysis(text):
    """Parse the analysis text into structured data, including dynamic scores.
    Handles potential formatting variations in the Gemini response.
    """
    overall_match = re.search(r"Overall Score:\s*(\d+)", text, re.IGNORECASE)
    overall_score = int(overall_match.group(1)) if overall_match else 0
    
    sections = {
        'Hook': {},
        'Script': {},
        'Visuals': {}
    }
    
    text_with_end_marker = text + "\nEND_OF_ANALYSIS"
    section_names_pattern = "|".join(sections.keys()) + "|END_OF_ANALYSIS"

    for section_name in sections:
        section_start_match = re.search(f"^({section_name})\s*\(Score:\s*(\d+)/50\)", text_with_end_marker, re.MULTILINE | re.IGNORECASE)
        if not section_start_match:
            st.warning(f"Could not find section header or score for: {section_name}")
            # Try finding the header without score as a fallback
            section_start_match_fallback = re.search(f"^{section_name}", text_with_end_marker, re.MULTILINE | re.IGNORECASE)
            if not section_start_match_fallback:
                continue # Skip section if header not found at all
            current_section_start_index = section_start_match_fallback.start()
            extracted_score = 0 # Default score if not found in header
            header_length = len(section_name)
        else:
            # Successfully extracted score from the header
            current_section_start_index = section_start_match.start()
            extracted_score = int(section_start_match.group(2))
            header_length = section_start_match.end() - section_start_match.start()

        # Find the start of the *next* section header or the end marker
        next_section_match = re.search(
            f"^(?:{section_names_pattern})", 
            text_with_end_marker[current_section_start_index + header_length:], 
            re.MULTILINE | re.IGNORECASE
        )
        
        # Extract the text belonging only to the current section (after the header)
        if next_section_match:
            current_section_content_text = text_with_end_marker[
                current_section_start_index + header_length : current_section_start_index + header_length + next_section_match.start()
            ].strip()
        else:
            current_section_content_text = text_with_end_marker[
                 current_section_start_index + header_length : -len("\nEND_OF_ANALYSIS")
            ].strip()

        # Parse "What works" within the current section's content text
        works_match = re.search(r"What works:(.*?)(?:What does not work:|$)", current_section_content_text, re.IGNORECASE | re.DOTALL)
        works_list = []
        if works_match:
            works_text = works_match.group(1).strip()
            works_list = [line.strip('- ').strip() for line in works_text.split('\n') if line.strip().startswith('-')]

        # Parse "What does not work" within the current section's content text
        not_works_match = re.search(r"What does not work:(.*?)$", current_section_content_text, re.IGNORECASE | re.DOTALL)
        not_works_list = []
        if not_works_match:
            not_works_text = not_works_match.group(1).strip()
            issues_raw = re.split(r'\n\s*-\s+', '\n- ' + not_works_text)
            for issue_block in issues_raw:
                 if not issue_block.strip(): continue
                 cleaned_block = issue_block.strip().lstrip('- ').strip()
                 if not cleaned_block: continue
                 actionable_match = re.search(r'🎯\s*Actionable:(.*)', cleaned_block, re.IGNORECASE | re.DOTALL)
                 if actionable_match:
                     issue_text = cleaned_block[:actionable_match.start()].strip()
                     actionable_text = actionable_match.group(1).strip()
                 else:
                     issue_text = cleaned_block
                     actionable_text = ''
                 if issue_text:
                    not_works_list.append({'issue': issue_text, 'actionable': actionable_text})

        # Update the sections dictionary with parsed data and the extracted score
        sections[section_name].update({
            'score': extracted_score,
            'works': works_list,
            'not_works': not_works_list
        })
    
    return overall_score, sections

# --- Streamlit App UI ---
st.title("📊 AI Creative Analysis")
st.markdown("Enter a YouTube URL **or** upload a video file to analyze its content.")

# Sidebar controls
with st.sidebar:
    st.header("Video Source & Options")

    # Video Source Selection
    st.subheader("Source")
    source_option = st.radio("Choose video source:", ("YouTube URL", "Upload Video File"), key="source_type")

    youtube_link = None
    uploaded_file_obj = None
    video_path_for_preview = None # To store URL or local path for preview

    if source_option == "YouTube URL":
        youtube_link = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...", key="youtube_url_input_sidebar")
        if youtube_link:
            video_id = get_video_id(youtube_link)
            if video_id:
                 video_path_for_preview = f"https://www.youtube.com/watch?v={video_id}"
            else:
                st.warning("Invalid YouTube URL format.")
    elif source_option == "Upload Video File":
        uploaded_file_obj = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi", "mkv", "webm"], # Common video formats
            key="video_upload"
        )
        if uploaded_file_obj is not None:
            # Save the uploaded file temporarily for preview
            # Note: Saving is necessary for st.video with local files
            temp_video_path = save_uploaded_file(uploaded_file_obj)
            if temp_video_path:
                video_path_for_preview = temp_video_path
            else:
                st.error("Failed to process uploaded video file.")

    # Advanced Options
    st.header("Advanced options")
    st.subheader("Industry")
    industry = st.selectbox(
        "Select Industry",
        ["Fashion", "Fitness", "Beauty", "Healthcare", "Wellness", "Baby and Children", "Food", "Tech", "Travel", "Education", "Finance", "Real Estate", "Automotive", "Entertainment", "Other"],
        key="industry"
    )
    st.subheader("Target Audience")
    target_age = st.number_input("Age", min_value=1, max_value=100, value=25, key="target_age_input")

    # Video Preview Area (conditionally displayed)
    if video_path_for_preview:
        st.subheader("Video Preview")
        try:
            st.video(video_path_for_preview)
        except Exception as e:
            st.error(f"Could not display video preview: {e}")
            # Clean up temp file if preview fails for local file
            if source_option == "Upload Video File" and video_path_for_preview and os.path.exists(video_path_for_preview):
                 try: os.remove(video_path_for_preview) 
                 except: pass # Ignore cleanup error

# --- Main Analysis Area --- 

analysis_triggered = st.button("Analyze Video", key="analyze_button_main", type="primary")

# Determine the source path for analysis (URL or saved uploaded file path)
source_for_analysis = None
temporary_files = [] # Keep track of temporary files to delete

if source_option == "YouTube URL" and youtube_link:
    if get_video_id(youtube_link):
        source_for_analysis = youtube_link
    else:
        # Error already shown in sidebar
        pass 
elif source_option == "Upload Video File" and uploaded_file_obj:
    # Save the uploaded file for processing (separate from preview potentially)
    temp_video_path_analysis = save_uploaded_file(uploaded_file_obj)
    if temp_video_path_analysis:
        source_for_analysis = temp_video_path_analysis
        temporary_files.append(temp_video_path_analysis) # Add video file to cleanup list
    else:
         st.error("Failed to save uploaded video file for analysis.")

if analysis_triggered and source_for_analysis:
    
    # --- Analysis Steps --- (Moved inside the button click conditional) --- #
    with st.spinner(f"Analyzing video for {industry} industry (Age: {target_age})..."):
        audio_file = None
        try:
            # Step 1: Get Audio File (Download or Extract)
            st.write("**Step 1: Preparing Audio...**")
            progress_bar_audio = st.progress(0, text="Starting audio preparation...")
            
            if source_option == "YouTube URL":
                progress_bar_audio.progress(10, text="Downloading audio from YouTube...")
                audio_file = download_youtube_audio_yt_dlp(source_for_analysis)
                if audio_file:
                    temporary_files.append(audio_file) # Add downloaded audio to cleanup list
                    progress_bar_audio.progress(50, text="Audio downloaded.")
                else:
                    st.error("Failed to download audio from YouTube.")
                    st.stop()
            elif source_option == "Upload Video File":
                progress_bar_audio.progress(10, text="Extracting audio from uploaded video...")
                audio_file = extract_audio_from_video(source_for_analysis)
                if audio_file:
                    temporary_files.append(audio_file) # Add extracted audio to cleanup list
                    progress_bar_audio.progress(50, text="Audio extracted.")
                else:
                    st.error("Failed to extract audio from uploaded video.")
                    st.stop()
            
            progress_bar_audio.progress(100, text="Audio ready for transcription.")
            st.success("✅ Audio Prepared")

            # Step 2: Load Whisper Model (Cached)
            st.write("**Step 2: Loading Transcription Model...**")
            whisper_model = load_whisper_model()
            if not whisper_model:
                st.error("Failed to load transcription model.")
                st.stop()
            st.success("✅ Transcription Model Loaded")

            # Step 3: Transcribe Audio
            st.write("**Step 3: Transcribing Audio...**")
            progress_bar_transcribe = st.progress(0, text="Starting transcription...")
            # Simulating progress for transcription (Whisper doesn't have built-in progress callbacks easily)
            # In a real app, you might update this based on observed time or segments
            progress_bar_transcribe.progress(50, text="Transcription in progress...")
            transcript = transcribe_audio(audio_file, whisper_model)
            if not transcript:
                st.error("Transcription failed.")
                st.stop()
            progress_bar_transcribe.progress(100, text="Transcription complete.")
            st.success("✅ Transcription Complete")
            with st.expander("View Transcript"):
                st.text_area("Transcript:", transcript, height=150)

            # Step 4: Analyze with Gemini
            st.write("**Step 4: Analyzing Content with AI...**")
            analysis = analyze_content(transcript, industry, target_age)
            if not analysis:
                st.error("AI analysis failed. Please check logs or API key.")
                st.stop()
            st.success("✅ Analysis Complete")

            # --- Display Analysis Results --- 
            st.markdown("--- ## Analysis Results ---")
            overall_score, sections = parse_analysis(analysis)

            # Header Buttons (Show only after successful analysis)
            st.markdown("""
                <div class="header-buttons">
                    <button class="share-button">Share</button>
                    <button class="try-button">Try Another Ad</button>
                    <button class="expert-button">Connect with Ads Expert</button>
                </div>
            """, unsafe_allow_html=True)

            # Display overall score
            score_color = "green" if overall_score >= 70 else ("orange" if overall_score >= 40 else "red")
            st.markdown(f"""
                <div class="main-score">
                    <div class="score-circle" style="border-color: {score_color}">
                        {overall_score}
                    </div>
                    <div class="score-label">Your overall score is {overall_score}/100</div>
                </div>
            """, unsafe_allow_html=True)

            # Display sections with full content and dynamic score
            for section_name, section_data in sections.items():
                display_score = section_data.get('score', 0)
                with st.expander(f"{section_name} ({display_score}/50)", expanded=True):
                    # What works section
                    if section_data.get('works'):
                        st.markdown('<div class="what-works">', unsafe_allow_html=True)
                        st.markdown("#### ✅ What works")
                        for item in section_data['works']:
                            st.markdown(f"• {item}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # What doesn't work section - Modified display
                    if section_data.get('not_works'):
                        st.markdown('<div class="what-not-works">', unsafe_allow_html=True)
                        st.markdown("#### ❌ What does not work")
                        for item in section_data['not_works']:
                            if item['actionable']:
                                combined_text = f"{item['issue']} (🎯 Actionable: {item['actionable']})"
                            else:
                                combined_text = item['issue']
                            st.markdown(f"• {combined_text}")
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An unexpected error occurred during the analysis process: {e}")
            # Optionally log the full traceback
            # import traceback
            # st.error(traceback.format_exc())
        finally:
            # Clean up all temporary files created during this run
            # Also clean up preview file if it exists and is temporary
            if source_option == "Upload Video File" and video_path_for_preview and os.path.exists(video_path_for_preview):
                if video_path_for_preview not in temporary_files:
                     temporary_files.append(video_path_for_preview)
            
            st.write("Cleaning up temporary files...")
            for temp_file_path in temporary_files:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        # st.write(f"Removed: {temp_file_path}") # Optional debug message
                    except Exception as e:
                        st.warning(f"Could not remove temporary file {temp_file_path}: {e}")
elif analysis_triggered and not source_for_analysis:
    st.warning("Please provide a valid YouTube URL or upload a video file before analyzing.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Note: Analysis is based on audio transcription and may take several minutes for longer videos.
    For best results, use videos with clear audio and select the appropriate language.
</div>
""", unsafe_allow_html=True) 