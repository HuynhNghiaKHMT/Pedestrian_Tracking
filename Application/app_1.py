# app.py
# ======================================================
# 🚶‍♂️ Pedestrian Tracking Streamlit App
# YOLOX + ReID + AFLink integration
# ======================================================

# --- Imports library ---
import streamlit as st
import os
import sys
import tempfile
import configparser
import subprocess
import warnings
from pathlib import Path
import shutil

# --- Tắt cảnh báo ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Cấu hình đường dẫn gốc ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Import các module trong project ---
try:
    from Tracktrack.YOLOX.run_detect import detect
    from Tracktrack.FastReID.ext_feats import reid
    from Tracktrack.Tracker.run import tracker
    from Utils.split_video import video_preprocess
    from Utils.create_output import create_output_folder, create_video
except ImportError as e:
    st.error(f"Lỗi Import: {e}")
    def detect(): st.info("Dummy Detect running")
    def reid(): st.info("Dummy ReID running")
    def tracker(): st.info("Dummy Tracker running")
    def video_preprocess(): st.info("Dummy Preprocess running")
    def create_output_folder(): st.info("Dummy Create Output running")
    def create_video(): st.info("Dummy Create Video running")


# --- Config & Setup ---
CONFIG_FILE = 'env.ini'
INPUT_PATH_BASE = PROJECT_ROOT / "Input"
OUTPUT_PATH_BASE = PROJECT_ROOT / "Outputs"


def write_env_config(uploaded_video_path, seq_name, detection_params, tracking_params):
    """
    Cập nhật env.ini: chỉ thay đổi các giá trị Detection & Tracking,
    giữ nguyên các section khác.
    """
    config = configparser.ConfigParser()
    config.optionxform = str  # giữ nguyên chữ hoa - chữ thường của key

    # Đọc file env.ini gốc (nếu có)
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE, encoding='utf-8')
    else:
        st.warning("⚠️ Không tìm thấy env.ini gốc, tạo file mới.")
    
    # Đảm bảo các section tồn tại
    if 'Detection' not in config:
        config['Detection'] = {}
    if 'Tracking' not in config:
        config['Tracking'] = {}
    if 'Input' not in config:
        config['Input'] = {}
    if 'Path' not in config:
        config['Path'] = {}
    if 'General' not in config:
        config['General'] = {}
    if 'Model' not in config:
        config['Model'] = {}

    # Cập nhật lại những giá trị cần thay đổi
    config['Input']['input_video'] = str(uploaded_video_path)
    config['Path']['input_path'] = 'Input'
    config['Path']['output_path'] = 'Outputs'
    config['General']['mode'] = 'test'
    config['Model']['data2model'] = 'mot17'

    # --- Detection ---
    config['Detection']['conf'] = str(detection_params['conf'])
    config['Detection']['nms_1'] = str(detection_params['nms_1'])
    config['Detection']['nms_2'] = str(detection_params['nms_2'])

    # --- Tracking ---
    config['Tracking']['penalty_p'] = str(tracking_params['penalty_p'])
    config['Tracking']['penalty_q'] = str(tracking_params['penalty_q'])
    config['Tracking']['tai_thr'] = str(tracking_params['tai_thr'])

    # --- Ghi đè lại file (giữ nguyên phần còn lại) ---
    with open(CONFIG_FILE, 'w', encoding='utf-8') as configfile:
        config.write(configfile)


def check_existing_processed_videos(seq_name: str):
    """Kiểm tra xem video đã được xử lý trước đó chưa."""
    video_dir = OUTPUT_PATH_BASE / seq_name / "videos"
    if not video_dir.exists():
        return False
    return len(list(video_dir.glob(f"{seq_name}_*.mp4"))) > 0


def convert_all_videos_to_h264(video_dir: Path, seq_name: str, status_placeholder):
    """Chuyển tất cả video sang định dạng H.264 (ghi đè file gốc)."""
    if not video_dir.exists():
        status_placeholder.error(f"❌ Không tìm thấy thư mục video: {video_dir}")
        return False

    mp4_files = list(video_dir.glob(f"{seq_name}_*.mp4"))
    if not mp4_files:
        status_placeholder.error("❌ Không tìm thấy video .mp4 nào để chuyển đổi.")
        return False


    converted = []

    for video_path in mp4_files:
        temp_path = video_path.with_name(video_path.stem + "_temp.mp4")
        command = [
            "ffmpeg", "-i", str(video_path),
            "-vcodec", "libx264", "-acodec", "aac", "-y",
            str(temp_path)
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            os.replace(temp_path, video_path)
            converted.append(video_path.name)
        except subprocess.CalledProcessError as e:
            status_placeholder.error(f"❌ FFmpeg lỗi với {video_path.name}: {e.stderr[:100]}")
        except FileNotFoundError:
            status_placeholder.error("❌ FFmpeg chưa cài hoặc chưa thêm PATH.")
            return False

    if converted:
        status_placeholder.success(f"✅ Đã chuyển: {', '.join(converted)}")
        return True
    return False


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Pedestrian Tracking")

st.title("🚶‍♂️ Pedestrian Tracking in Video")
st.write("Ứng dụng sử dụng YOLOX, ReID và AFLink để theo dõi người đi bộ trong video.")

# --- Session State ---
st.session_state.setdefault('uploaded_file_path', None)
st.session_state.setdefault('video_name', None)
st.session_state.setdefault('video_processed', False)
st.session_state.setdefault('is_running', False)
st.session_state.setdefault('current_seq_name', None)
st.session_state.setdefault('selected_video_type', None)

# --- Sidebar Upload ---
st.sidebar.header("📁 Upload Video")
uploaded_file = st.sidebar.file_uploader("Chọn video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_dir = Path(tempfile.gettempdir())
    temp_video_path = temp_dir / uploaded_file.name
    seq_name = Path(uploaded_file.name).stem

    if st.session_state.uploaded_file_path is None or temp_video_path != Path(st.session_state.uploaded_file_path):
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.uploaded_file_path = str(temp_video_path)
        st.session_state.video_name = uploaded_file.name
        st.session_state.current_seq_name = seq_name

        if check_existing_processed_videos(seq_name):
            st.session_state.video_processed = True
            st.sidebar.success(f"🟢 Video '{seq_name}' đã được xử lý trước đó.")
        else:
            st.session_state.video_processed = False
            st.sidebar.info("🕓 Video mới, cần xử lý pipeline.")
        st.rerun()

# --- Sidebar: Detection Parameters ---
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Parameters")

conf = st.sidebar.slider("Confidence threshold (conf)", 0.0, 1.0, 0.1, 0.05)
nms_1 = st.sidebar.slider("NMS 1 threshold (nms_1)", 0.0, 1.0, 0.8, 0.05)
nms_2 = st.sidebar.slider("NMS 2 threshold (nms_2)", nms_1 + 0.05, 1.0, 0.95, 0.05)

# --- Sidebar: Tracking Parameters ---
st.sidebar.markdown("---")
st.sidebar.subheader("Tracking Parameters")

penalty_p = st.sidebar.slider("Penalty p", 0.0, 1.0, 0.2, 0.05)
penalty_q = st.sidebar.slider("Penalty q", 0.0, 1.0, 0.4, 0.05)
tai_thr = st.sidebar.slider("TAI Threshold", 0.0, 1.0, 0.55, 0.05)

# --- Sidebar: Video Output Selector ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Video Output Viewer")

seq_name = st.session_state.get('current_seq_name')
if seq_name:
    video_dir = OUTPUT_PATH_BASE / seq_name / "videos"
    video_options = {
        "Detection": f"{seq_name}_detection.mp4",
        "Trajectory": f"{seq_name}_trajectory.mp4",
        "Tracking": f"{seq_name}_tracking.mp4",
        "Combine": f"{seq_name}_combine.mp4",
    }

    selected_video_type = st.sidebar.selectbox(
        "Chọn loại video:",
        list(video_options.keys()),
        index=2,
        help="Chọn video kết quả: detection, tracking, predict hoặc final."
    )
    st.session_state['selected_video_type'] = selected_video_type
else:
    st.sidebar.info("Vui lòng tải video lên trước và tiến hành xử lý.")

st.sidebar.markdown("---")
start_button = st.sidebar.button("▶️ Start Tracking")

# --- Hiển thị video ---
if st.session_state.uploaded_file_path and not st.session_state.is_running:
    if st.session_state.video_processed:
        seq_name = st.session_state.current_seq_name
        video_dir = OUTPUT_PATH_BASE / seq_name / "videos"

        if st.session_state.get('selected_video_type'):
            video_options = {
                "Detection": f"{seq_name}_detection.mp4",
                "Trajectory": f"{seq_name}_trajectory.mp4",
                "Tracking": f"{seq_name}_tracking.mp4",
                "Combine": f"{seq_name}_combine.mp4",
            }

            selected_video_name = video_options[st.session_state['selected_video_type']]
            selected_video_path = video_dir / selected_video_name

            if selected_video_path.exists():
                st.video(str(selected_video_path))
                st.info(f"📁 {selected_video_path}")
            else:
                st.warning(f"❌ Không tìm thấy: {selected_video_path.name}")
    else:
        st.info("📹 Video mới — hãy nhấn **Start Tracking** để bắt đầu.")
elif st.session_state.is_running:
    st.warning("⏳ Đang xử lý, vui lòng chờ...")


# --- Pipeline ---
def run_full_pipeline(seq_name):
    st.session_state.is_running = True
    progress = st.progress(0)
    status = st.empty()

    # Ghi lại config trước khi chạy
    detection_params = {'conf': conf, 'nms_1': nms_1, 'nms_2': nms_2}
    tracking_params = {'penalty_p': penalty_p, 'penalty_q': penalty_q, 'tai_thr': tai_thr}
    write_env_config(st.session_state.uploaded_file_path, seq_name, detection_params, tracking_params)

    try:
        status.info("Bước 1/6: Video preprocessing...")
        video_preprocess(); create_output_folder(); progress.progress(10)

        status.info("Bước 2/6: Detection (YOLOX)...")
        detect(); progress.progress(30)

        status.info("Bước 3/6: Extract features (ReID)...")
        reid(); progress.progress(60)

        status.info("Bước 4/6: Tracking & Post-processing...")
        tracker(); progress.progress(80)

        status.info("Bước 5/6: Generate output video...")
        create_video(); progress.progress(90)

        status.info("Bước 5/6: Convert video to H.264...")
        video_dir = OUTPUT_PATH_BASE / seq_name / "videos"
        convert_all_videos_to_h264(video_dir, seq_name, status)

        st.session_state.video_processed = True
        progress.progress(100)
        st.success(f"✅ Hoàn tất xử lý video: {seq_name}")
    except Exception as e:
        st.error(f"❌ Pipeline lỗi: {e}")
    finally:
        st.session_state.is_running = False
        progress.empty()
        st.rerun()


# --- Trigger ---
if start_button and st.session_state.uploaded_file_path and not st.session_state.is_running:
    run_full_pipeline(st.session_state.current_seq_name)
elif start_button and st.session_state.is_running:
    st.warning("⚠️ Pipeline đang chạy, vui lòng chờ.")
elif start_button and not st.session_state.uploaded_file_path:
    st.error("Vui lòng tải video trước khi chạy pipeline.")
