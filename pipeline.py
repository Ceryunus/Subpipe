import os
import json
import logging
import logging.handlers
import argparse
from pathlib import Path
import torch
from mp3extractor import extract_audio_from_video
from transcribe_only import transcribe
from translate_only import load_segments, translate
from embed_subtitles import add_soft_subs, add_burned_subs

# ENVIRONMENT & CONFIG

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

paths = config['paths']
models = config['models']
subtitles = config['subtitles']
logging_config = config['logging']

# Log setup

os.makedirs(paths['logs_dir'], exist_ok=True)
log_file = os.path.join(paths['logs_dir'], 'pipeline.log')
handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=int(logging_config['max_file_size'].replace('MB', '')) * 1024 * 1024,
    backupCount=5
)
logging.basicConfig(
    handlers=[handler],
    level=getattr(logging, logging_config['level']),
    format=logging_config['format']
)


# HELPERS

def reload_config():
    global paths, models, subtitles, logging_config

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    paths = config['paths']
    models = config['models']
    subtitles = config['subtitles']
    logging_config = config['logging']

    # Close hadler
    for h in logging.getLogger().handlers[:]:
        h.close()
        logging.getLogger().removeHandler(h)

    
    max_file_size_mb = logging_config['max_file_size'].replace('MB', '').strip()
    max_bytes = int(max_file_size_mb) * 1024 * 1024

    os.makedirs(paths['logs_dir'], exist_ok=True)
    log_file = os.path.join(paths['logs_dir'], 'pipeline.log')

    new_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=5
    )
    logging.basicConfig(
        handlers=[new_handler],
        level=getattr(logging, logging_config['level']),
        format=logging_config['format']
    )
    logging.info("reload_config(): Reload config and settings.")

def cleanup_gpu() -> None:
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("GPU cache cleared")
    except Exception as e:
        logging.warning(f"GPU cleanup skipped: {e}")

def select_video_file(video_arg: str | None) -> str:
    if video_arg:
        video_path = Path(video_arg).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return video_path.as_posix()
    video_files = list(Path(paths["video_dir"]).glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No MP4 files found in {paths['video_dir']}")
    return video_files[0].resolve().as_posix()

# PIPELINE STEPS

def step_extract_audio(video_file, output_audio):
    logging.info("Step 1: Extracting audio...")
    extract_audio_from_video(video_file, str(output_audio))
    logging.info("Step 1 completed.")

def step_transcribe(output_audio, subs_original_json, subs_original_file):
    logging.info("Step 2: Transcribing audio...")
    transcribe(output_audio, subs_original_json, subs_original_file)
    cleanup_gpu()
    logging.info("Step 2 completed.")

def step_translate(subs_original_json, subs_translated_srt, subs_translated_txt):
    logging.info("Step 3: Translating transcription...")
    segments, source_lang = load_segments(subs_original_json)
    translate(
        segments,
        source_lang=source_lang,
        target_lang=models['target_lang'],
        output_srt=subs_translated_srt,
        output_txt=str(subs_translated_txt),
    )
    cleanup_gpu()
    logging.info("Step 3 completed.")

def step_subtitles(video_file, subs_translated_srt, output_soft, output_burned, mode):
    logging.info(f"Step 4: Adding subtitles ({mode})...")
    if mode == 'soft':
        add_soft_subs(video_file, subs_translated_srt, str(output_soft))
    else:
        add_burned_subs(video_file, subs_translated_srt, str(output_burned))
    logging.info("Step 4 completed.")

# MAIN PIPELINE FUNCTION (with callback)

def run_pipeline(steps, mode=None, video_arg=None, on_step_complete=print):
    try:
        reload_config()
        video_file = select_video_file(video_arg)
        output_audio = Path(paths['audio_dir']) / paths['audio_file']
        subs_original_json = Path(paths['subs_original_dir']) / paths['subs_original_file_json']
        subs_original_file = Path(paths['subs_original_dir']) / paths['subs_original_file']
        subs_translated_srt = (Path(paths['subs_translated_dir']) / paths['subs_translated_file_srt']).as_posix()
        subs_translated_txt = Path(paths['subs_translated_dir']) / paths['subs_translated_file_txt']
        output_soft = Path(paths['video_with_subs_dir']) / paths['output_soft']
        output_burned = Path(paths['video_with_subs_dir']) / paths['output_burned']
        os.makedirs(Path(paths['audio_dir']), exist_ok=True)

        final_mode = mode or subtitles.get('mode', 'burned')
        if "extract" in steps:
            step_extract_audio(video_file, output_audio)
            on_step_complete("extract")
            

        if "transcribe" in steps:
            step_transcribe(output_audio, subs_original_json, subs_original_file)
            on_step_complete("transcribe")

        if "translate" in steps:
            step_translate(subs_original_json, subs_translated_srt, subs_translated_txt)
            on_step_complete("translate")

        if "subtitles" in steps:
            step_subtitles(video_file, subs_translated_srt, output_soft, output_burned, final_mode)
            on_step_complete("subtitles")

        logging.info("Pipeline completed successfully.")
        on_step_complete("done")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        on_step_complete("error", str(e))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subtitle processing pipeline from video to subtitled video.",
        epilog="""Examples:
            python pipeline.py --video "FilePath"     # Run full pipeline for custom video.
            python pipeline.py                        # Run full pipeline (for : data/video/sample.mp4 | mode from config)
            python pipeline.py --steps transcribe     # Only transcribe
            python pipeline.py --steps subtitles --mode soft   # Add soft subtitles
            python pipeline.py --steps subtitles --mode burned # Add burned subtitles
            python pipeline.py --steps extract transcribe  # Extract + Transcribe""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["extract", "transcribe", "translate", "subtitles"],
        help="Pipeline steps to run. Options: extract, transcribe, translate, subtitles"
    )
    parser.add_argument(
        "--mode",
        choices=["soft", "burned"],
        help="Subtitle mode: soft or burned (overrides config.json)"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Custom video file path (optional)"
    )
    args = parser.parse_args()
    run_pipeline(args.steps, mode=args.mode, video_arg=args.video)
