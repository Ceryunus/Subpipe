# SubPipe: Automated Video Subtitling Pipeline

SubPipe is a Python-based pipeline that automates the process of
extracting audio from videos, transcribing speech, translating text, and
embedding subtitles. It leverages Whisper's automatic language detection
for transcription and supports translation to 196 languages using
NLLB-200, delivering high-accuracy, multilingual subtitles with a
flexible and modular design.

## 🌟 Key Features

- **End-to-End Automation**: Process videos to subtitled output with a
  single command.
- **Smart Multilingual Support**: Automatically detects source language
  (99 languages via Whisper) and translates to any target language
  specified in `config.json`.
- **Powerful AI Models**: This project supports a wide range of models. 
Check the `config.json` file for all supported models and options.
  - **Whisper Large-v3**: High-accuracy speech-to-text transcription
    with auto-detection.
  - **Meta NLLB-200**: Fast and precise text translation for 196
    languages.
- **Flexible Subtitle Embedding**:
  - **Soft Subtitles**: Adds subtitles as a toggleable track.
  - **Burned Subtitles**: Permanently embeds subtitles with customizable
    position (bottom, top, center).
- **Customizable Output**: Supports `.srt`, `.json`, and `.txt` formats
  for original and translated subtitles.
- **Advanced Configuration**:
  - Control subtitle appearance (font, size, color, position).
  - Hardware acceleration (`hwaccel`) for faster processing.
  - Limit subtitle length (`max_line_length`) and lines (`max_lines`)
    for readability.
  - Log rotation with configurable `max_file_size`.
- **Memory Efficient**: Clears GPU memory after each step for optimized
  resource usage.
- **Robust Error Handling**: Detailed logging and specific error
  messages for easier debugging.
- **Optional FFmpeg Audio Extraction**: Faster audio extraction with
  FFmpeg and hardware acceleration support.

⚠️ **Important Note**: This project requires an NVIDIA GPU for efficient
transcription and translation. FFmpeg must be installed for subtitle
embedding and optional audio extraction.

## 📂 Project Structure

    Subpipe/
    ├── pipeline.py              # Orchestrates the full pipeline
    ├── mp3extractor.py         # Extracts audio (MoviePy or FFmpeg)
    ├── transcribe_only.py      # Transcribes audio using Whisper
    ├── translate_only.py       # Translates text using NLLB-200
    ├── embed_subtitles.py      # Embeds subtitles using FFmpeg
    ├── config.json             # Configuration for paths, models, and settings
    ├── requirements.txt        # Project dependencies
    ├── data/
    │   ├── video/             # Input videos
    │   ├── audio/             # Extracted audio files
    │   ├── subs_original/     # Original transcriptions (.srt, .json)
    │   ├── subs_translated/   # Translated subtitles (.srt, .txt)
    │   ├── logs/              # Runtime logs with rotation
    │   └── video_with_subs/   # Output videos with subtitles
    └──

## 🚀 Installation

1.  **Set Up the Environment**:

    ``` bash
    conda create -n subpipe python=3.10 -y
    conda activate subpipe
    ```

2.  **Clone Repo**:

    ``` bash
    git clone https://github.com/Ceryunus/Subpipe.git
    cd Subpipe
    ```

3.  **Install Dependencies**:

    ``` bash
    pip install -r requirements.txt
    ```

4.  **Install PyTorch for CUDA**:

    Before running, install the appropriate PyTorch and CUDA version for
    your system. Visit:

    ``` bash
    https://pytorch.org/get-started/locally
    ```

5.  **Install FFmpeg**:

    Ensure FFmpeg is installed and accessible in your system's PATH. For
    example:

    - **Ubuntu**: `sudo apt-get install ffmpeg`
    - **Windows**: Download from [FFmpeg
      website](https://ffmpeg.org/download.html) and add to PATH.
    - For easy installation way, use
      [chocolatey](https://chocolatey.org/install)
    - **macOS**: `brew install ffmpeg`

    Whisper Large-v3 and NLLB-200 models are downloaded automatically
    from Hugging Face on first run.

## ▶️ Usage

Run `pipeline.py` to process videos with flexible options.

### 1. Run the Full Pipeline

- Edit the `target_lang` from the `config.json`
- Processes the first `.mp4` file in `data/video/`:

``` bash
python pipeline.py --mode burned
```

- `--mode burned`: Embeds subtitles directly onto the video.
- `--mode soft`: Adds toggleable subtitles.

### 2. Process a Specific Video

``` bash
python pipeline.py --video "path/to/video.mp4" --mode burned
```

- `--video`: Specifies the video file path (enclose in quotes).

### 3. Run Specific Steps (optional)

- **Extract Audio Only**:

  ``` bash
  python pipeline.py --steps extract
  ```

- **Transcription Only** (after audio extraction):

  ``` bash
  python pipeline.py --steps transcribe
  ```

- **Translation Only** (after transcription):

  ``` bash
  python pipeline.py --steps translate
  ```

- **Subtitle Embedding Only** (after translation):

  ``` bash
  python pipeline.py --steps subtitles --mode burned
  ```

### 4. Example Workflow

1.  Copy an `.mp4` video to `data/video/` (replace `sample.mp4`).

2.  Run:

    ``` bash
    python pipeline.py --mode burned
    ```

3.  Find outputs:

    - Subtitled video: `data/video_with_subs/`
    - Original subtitles: `data/subs_original/` (`.srt`, `.json`)
    - Translated subtitles: `data/subs_translated/` (`.srt`, `.txt`)
    - Logs: `data/logs/` (rotated based on `max_file_size`)

## 📊 Workflow

1.  **Video to Audio**: Extracts audio using MoviePy or FFmpeg
    (`data/audio/`).
2.  **Audio to Transcription**: Uses Whisper to transcribe audio
    (`.srt`, `.json`).
3.  **Transcription to Translation**: Translates text with NLLB-200,
    respecting `max_line_length` and `max_lines`.
4.  **Translation to Video**: Embeds subtitles (soft or burned) with
    customizable position and hardware acceleration.

## ⚙️ Configuration (`config.json`)

Customize the pipeline via `config.json`:

``` json
{
  "paths": {
    "video_dir": "data/video",
    "audio_dir": "data/audio",
    "subs_original_dir": "data/subs_original",
    "subs_translated_dir": "data/subs_translated",
    "logs_dir": "data/logs",
    "video_with_subs_dir": "data/video_with_subs",
    "video_file": "sample.mp4",
    "audio_file": "sample.mp3",
    "subs_original_file": "original.srt",
    "subs_original_file_json": "segments.json",
    "subs_translated_file_srt": "translated.srt",
    "subs_translated_file_txt": "translated.txt",
    "output_soft": "sample_output_soft.mp4",
    "output_burned": "sample_output_burned.mp4"
  },
  "ffmpeg": {
    "font_name": "Arial",
    "font_size": 24,
    "primary_colour": "&H00FFFFFF",
    "outline_colour": "&H00000000",
    "border_style": 3,
    "outline": 1,
    "shadow": 0,
    "preset": "medium",
    "crf": 23,
    "pix_fmt": "yuv420p",
    "subtitle_position": "bottom",
    "_comment_subtitle_position": "Position of burned subtitles: 'bottom', 'top', or 'center'.",
    "hwaccel": "auto",
    "_comment_hwaccel": "Hardware acceleration for FFmpeg: 'auto', 'cuda', or 'none'."
  },
  "models": {
    "source_lang": "auto",
    "_comment_source_lang": "Source language for transcription. Set to 'auto' for auto-detection or use NLLB format (e.g., 'eng_Latn').",
    "whisper_model": "medium",
    "translation_model": "facebook/nllb-200-1.3B",
    "target_lang": "tur_Latn",
    "language_code_map": {
      "en": "eng_Latn",
      "tr": "tur_Latn",
      ...
    }
  },
  "logging": {
    "level": "DEBUG",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "max_file_size": "10MB",
    "_comment_max_file_size": "Maximum log file size before rotation (e.g., '10MB')."
  },
  "subtitles": {
    "mode": "burned",
    "max_line_length": 60,
    "_comment_max_line_length": "Maximum characters per subtitle line for readability.",
    "max_lines": 2,
    "_comment_max_lines": "Maximum number of subtitle lines displayed at once.",
    "use_regex_splitter": false,
    "_comment_use_regex_splitter": "Use regex-based sentence splitter instead of NLTK for multilingual support."
  },
  "audio_extraction": {
    "use_ffmpeg": false,
    "_comment_use_ffmpeg": "Use FFmpeg instead of MoviePy for audio extraction (faster with hwaccel)."
  }
}
```

### Key Configuration Options

- **Paths**: Customize input/output directories and file names.
- **FFmpeg**: Adjust font, size, color, subtitle position (`bottom`,
  `top`, `center`), and hardware acceleration (`hwaccel`).
- **Models**: Set Whisper model (`tiny`, `base`, `medium`, etc.) and
  NLLB target language (`tur_Latn`, `eng_Latn`, etc.).
- **Subtitles**: Control subtitle mode (`soft` or `burned`), max line
  length, and max lines.
- **Logging**: Set log level (`DEBUG`, `INFO`) and max file size for
  rotation.
- **Audio Extraction**: Toggle between MoviePy and FFmpeg for faster
  processing.
- **Whisper Model Param**: Can be change whisper model parameters 
  in config.json


## 📜 License

This project is free for personal and educational use.
