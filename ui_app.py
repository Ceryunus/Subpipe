import streamlit as st
import streamlit.runtime.scriptrunner as st_runner
import os
import shutil
import json
import threading
import time
from pathlib import Path
from typing import List
import queue
import torch
from pipeline import run_pipeline


# CONSTANTS & HELPERS

DATA_ROOT = Path("data")
OUTPUT_DIRS = [
    DATA_ROOT / "audio",
    DATA_ROOT / "subs_original",
    DATA_ROOT / "subs_translated",
    DATA_ROOT / "video_with_subs",
    DATA_ROOT / "logs",
]
WHISPER_MODELS = {"tiny":"tiny","base":"base","small":"small","medium":"medium (Recommended)","large-v3":"large-v3"}
NLLB_MODELS = {
    "facebook/nllb-200-distilled-600M": "600M (Fast)",
    "facebook/nllb-200-1.3B": "1.3B (Accurate - Recommended)",
    "facebook/nllb-200-3.3B": "3.3B (Most Accurate)",
}
LANGUAGE_MAP = {"en":"eng_Latn","tr":"tur_Latn","es":"spa_Latn","fr":"fra_Latn","de":"deu_Latn","it":"ita_Latn","pt":"por_Latn","nl":"nld_Latn","pl":"pol_Latn","ro":"ron_Latn","sv":"swe_Latn","da":"dan_Latn","no":"nno_Latn","nb":"nob_Latn","fi":"fin_Latn","cs":"ces_Latn","sk":"slk_Latn","hu":"hun_Latn","hr":"hrv_Latn","sr":"srp_Cyrl","bg":"bul_Cyrl","uk":"ukr_Cyrl","ru":"rus_Cyrl","el":"ell_Grek","ja":"jpn_Jpan","ko":"kor_Hang","zh":"zho_Hans","zh-tw":"zho_Hant","th":"tha_Thai","vi":"vie_Latn","hi":"hin_Deva","bn":"ben_Beng","ta":"tam_Taml","te":"tel_Telu","ml":"mal_Mlym","kn":"kan_Knda","gu":"guj_Gujr","pa":"pan_Guru","ur":"urd_Arab","fa":"fas_Arab","he":"heb_Hebr","id":"ind_Latn","ms":"zsm_Latn","ar":"ara_Arab","af":"afr_Latn","sq":"als_Latn","az":"aze_Latn","eu":"eus_Latn","be":"bel_Cyrl","ca":"cat_Latn","et":"est_Latn","gl":"glg_Latn","is":"isl_Latn","lv":"lvs_Latn","lt":"lit_Latn","mk":"mkd_Cyrl","mt":"mlt_Latn","sl":"slv_Latn","sw":"swh_Latn","cy":"cym_Latn"}
STEP_ORDER = ["extract", "transcribe", "translate", "subtitles"]

def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def check_cuda() -> bool:
    return torch.cuda.is_available()

def get_gpu_name() -> str:
    return torch.cuda.get_device_name(0) if check_cuda() else "None"

def check_disk_space(min_gb: int = 5) -> bool:
    total, used, free = shutil.disk_usage(DATA_ROOT)
    return free / (1024**3) >= min_gb

def clear_output_dirs() -> None:
    for d in OUTPUT_DIRS:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            continue
        for root, _, files in os.walk(d):
            for f in files:
                fp = Path(root) / f
                if fp.name == "pipeline.log":
                    continue
                try:
                    fp.unlink()
                except (PermissionError, OSError):
                    continue
        for root, dirs, _ in os.walk(d, topdown=False):
            for name in dirs:
                try:
                    (Path(root) / name).rmdir()
                except OSError:
                    pass

def load_config() -> dict:
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg: dict) -> None:
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# PAGE SETTINGS & STYLES 

os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/config.toml", "w") as f:
    f.write("[server]\nmaxUploadSize = 2048\nfileWatcherType = 'none'\n")

st.set_page_config(page_title="Video Subtitle Pipeline", page_icon="🎬", layout="wide")
# custom CSS for better visuals
st.markdown(
    """
    <style>
    .main { padding: 0.5rem 1rem; }
    .card {
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
    }
    .stButton>button {
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
    }
    .step-bubble {
        display:inline-block;
        min-width:36px;
        height:36px;
        line-height:36px;
        text-align:center;
        border-radius:18px;
        margin-right:8px;
        background:#2b6cb0;
        color:white;
        font-weight:700;
    }
    .muted { color: #777; font-size:0.95em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎞️ Automatic Video Subtitle Pipeline")
st.write("Upload MP4 → Extract Audio → Transcribe → Translate → Add Subtitles")

# LAYOUT: left controls, right preview

col_left, col_right = st.columns([1.1, 1.9])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Controls")
    st.caption("Sistem bilgileri ve ayarlar")

    st.markdown(f"**FFmpeg:** {'✅' if check_ffmpeg() else '❌ Not found'}")
    st.markdown(f"**GPU:** `{get_gpu_name()}`")
    st.markdown(f"**Cuda:** `{'✅ Avaible' if check_cuda() else '❌ Not found'}`")
    st.markdown(f"**Disk:** {'✅ Enough' if check_disk_space() else '⚠️ Low'}")

    st.divider()
    st.markdown("### 1. Upload")
    uploaded = st.file_uploader("MP4 file", type=["mp4"], label_visibility="collapsed")
    video_path: Path | None = None
    if uploaded:
        video_dir = DATA_ROOT / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "sample.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("Video uploaded and saved.")

    st.markdown("### 2. Models & Language")
    whisper_model = st.selectbox("Whisper model", list(WHISPER_MODELS.keys()), index=3, format_func=lambda x: WHISPER_MODELS[x])
    nllb_model = st.selectbox("NLLB model", list(NLLB_MODELS.keys()), index=1, format_func=lambda x: NLLB_MODELS[x])
    target_lang_name = st.selectbox("Target language", list(LANGUAGE_MAP.keys()))
    target_lang = LANGUAGE_MAP[target_lang_name]

    st.markdown("### 3. Pipeline")
    subtitle_mode = st.radio("Subtitle type", ["burned", "soft"], index=0, horizontal=True)
    full_pipeline = st.checkbox("Run all steps (recommended)", value=True)
    if not full_pipeline:
        steps: List[str] = []
        if st.checkbox("Extract audio"): steps.append("extract")
        if st.checkbox("Transcribe"): steps.append("transcribe")
        if st.checkbox("Translate"): steps.append("translate")
        if st.checkbox("Add subtitles"): steps.append("subtitles")
    else:
        steps = STEP_ORDER.copy()

    st.divider()
    run_btn = st.button("▶️ Start pipeline", type="primary", disabled=not uploaded or st.session_state.get("pipeline_running", False))
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # Stepper
    st.subheader("📈 Progress")
    step_cols = st.columns(4)
    for i, step in enumerate(STEP_ORDER):
        with step_cols[i]:
            completed = st.session_state.get("progress", 0.0) >= (i+1)/len(STEP_ORDER)
            bubble = f"<div class='step-bubble' style='background:{'#2b6cb0' if completed else '#CBD5E0'}'>{i+1}</div>"
            st.markdown(bubble, unsafe_allow_html=True)
            st.caption(step.capitalize())

    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    status_placeholder = st.empty()

    # Video + Tabs (Original SRT / Translated SRT) — same panel
    st.divider()
    st.subheader("🎬 Preview & Subtitles")
    preview_cols = st.columns([1, 1])
    with preview_cols[0]:
        # video display if ready
        soft_vid = DATA_ROOT / "video_with_subs" / "sample_output_soft.mp4"
        burned_vid = DATA_ROOT / "video_with_subs" / "sample_output_burned.mp4"
        final_vid = soft_vid if subtitle_mode == "soft" and soft_vid.exists() else burned_vid if burned_vid.exists() else None
        if final_vid and final_vid.exists():
            st.video(str(final_vid))
            with open(final_vid, "rb") as f:
                st.download_button("Download final video", f, f"video_{'soft' if final_vid==soft_vid else 'burned'}.mp4")
        else:
            # show uploaded original video if exists
            if video_path and video_path.exists():
                st.video(str(video_path))
            else:
                st.info("Upload a video to preview here.")

    # Tabs for SRTs right beside video
    with preview_cols[1]:
        tabs = st.tabs(["Original SRT", f"{target_lang_name.upper()} SRT", "Audio / Files"])
        orig_srt = DATA_ROOT / "subs_original" / "original.srt"
        trans_srt = DATA_ROOT / "subs_translated" / "translated.srt"

        with tabs[0]:
            st.markdown("**Original (auto) SRT**")
            if orig_srt.exists():
                with open(orig_srt, "r", encoding="utf-8") as f:
                    st.text_area("Original SRT", f.read(), height=400, key="orig_srt_area")
                with open(orig_srt, "rb") as f:
                    st.download_button("Download original.srt", f, "original.srt", key="dl_orig")
            else:
                st.write("No original SRT yet.")

        with tabs[1]:
            st.markdown(f"**{target_lang_name.upper()} translated SRT**")
            if trans_srt.exists():
                with open(trans_srt, "r", encoding="utf-8") as f:
                    st.text_area("Translated SRT", f.read(), height=400, key="trans_srt_area")
                with open(trans_srt, "rb") as f:
                    st.download_button("Download translated.srt", f, "translated.srt", key="dl_trans")
            else:
                st.write("No translated SRT yet.")

        with tabs[2]:
            st.markdown("**Audio & Helpers**")
            audio_path = DATA_ROOT / "audio" / "sample.mp3"
            if audio_path.exists():
                st.audio(str(audio_path))
                with open(audio_path, "rb") as f:
                    st.download_button("Download MP3", f, "audio.mp3", key="dl_audio_small")
            else:
                st.write("No audio extracted yet.")
    st.markdown("</div>", unsafe_allow_html=True)


# SESSION STATE 

defaults = {
    "log": "",
    "progress": 0.0,
    "pipeline_running": False,
    "pipeline_finished": False,
    "status_msg": "",
    "event_queue": queue.Queue(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
event_queue = st.session_state.event_queue

# PIPELINE LAUNCH

if run_btn and video_path and not st.session_state.pipeline_running:
    clear_output_dirs()
    st.session_state.log = ""
    st.session_state.progress = 0.0
    st.session_state.pipeline_finished = False
    st.session_state.status_msg = ""
    cfg = load_config()
    cfg["models"]["whisper_model"] = whisper_model
    cfg["models"]["translation_model"] = nllb_model
    cfg["models"]["target_lang"] = target_lang
    cfg["subtitles"]["mode"] = subtitle_mode
    save_config(cfg)
    st.session_state.pipeline_running = True

    def on_step_complete(step, msg=None):
        event_queue.put((step, msg))

    def worker():
        ctx = st_runner.get_script_run_ctx()
        st_runner.add_script_run_ctx(threading.current_thread(), ctx)
        try:
            run_pipeline(steps, mode=subtitle_mode, video_arg=str(video_path), on_step_complete=on_step_complete)
        except Exception as e:
            event_queue.put(("error", str(e)))
        finally:
            event_queue.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()
    st.rerun()


# EVENT LOOP

def process_events():
    q = st.session_state.event_queue
    rerun_needed = False
    while not q.empty():
        step, msg = q.get_nowait()
        if step in STEP_ORDER:
            idx = STEP_ORDER.index(step) + 1
            st.session_state.progress = idx / len(STEP_ORDER)
            st.session_state.log += f"{STEP_ORDER[idx-1]} step completed.\n"
        elif step == "done":
            st.session_state.pipeline_running = False
            st.session_state.pipeline_finished = True
            st.session_state.status_msg = "Pipeline completed successfully!"
            rerun_needed = True
        elif step == "error":
            st.session_state.pipeline_running = False
            st.session_state.pipeline_finished = True
            st.session_state.status_msg = f"Error: {msg}"
            rerun_needed = True
    if rerun_needed and not st.session_state.get("rerun_done", False):
        st.session_state.rerun_done = True
        st.rerun()

process_events()


# OUTPUTS

soft_vid = DATA_ROOT / "video_with_subs" / "sample_output_soft.mp4"
burned_vid = DATA_ROOT / "video_with_subs" / "sample_output_burned.mp4"
final_vid = soft_vid if subtitle_mode == "soft" and soft_vid.exists() else burned_vid if burned_vid.exists() else None

# Status / progress / logs
if st.session_state.pipeline_running:
    progress_placeholder.progress(st.session_state.progress, text=f"Progress: {int(st.session_state.progress*100)}%")
    log_placeholder.code(st.session_state.log or "Running…", language="text")
    status_placeholder.warning("Pipeline is running...")
    time.sleep(1.0)
    st.rerun()
elif st.session_state.pipeline_finished:
    progress_placeholder.progress(1.0, text="Completed – 100%")
    log_placeholder.code(st.session_state.log, language="text")
    if "Error" in st.session_state.status_msg:
        status_placeholder.error(st.session_state.status_msg)
    else:
        status_placeholder.success(st.session_state.status_msg)
else:
    progress_placeholder.progress(0.0)
    log_placeholder.code("No log yet.", language="text")
    status_placeholder.info("Upload a video and start.")
