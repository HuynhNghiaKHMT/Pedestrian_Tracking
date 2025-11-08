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
import json  # Import thư viện json để lưu trữ tham số
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
    # st.error(f"Lỗi Import: {e}")
    def detect(): st.info("Dummy Detect running")
    def reid(): st.info("Dummy ReID running")
    def tracker(): st.info("Dummy Tracker running")
    def video_preprocess(): st.info("Dummy Preprocess running")
    def create_output_folder(): st.info("Dummy Create Output running")
    def create_video(): st.info("Dummy Create Video running")


# --- Config & Setup ---
CONFIG_FILE = 'env.ini'
PARAMS_CACHE_FILE = Path(tempfile.gettempdir()) / "tracking_params_cache.json"
INPUT_PATH_BASE = PROJECT_ROOT / "Input"
OUTPUT_PATH_BASE = PROJECT_ROOT / "Outputs"
OUTPUT_PATH_VIDEO = INPUT_PATH_BASE / "videos"


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
    for section in ['Detection', 'Tracking', 'Input', 'Path', 'General', 'Model']:
        if section not in config:
            config[section] = {}

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
    """Kiểm tra xem video đã được xử lý (có các file video output) trước đó chưa."""
    video_dir = OUTPUT_PATH_BASE / seq_name / "videos"
    if not video_dir.exists():
        return False
    # Giả sử chỉ cần check file tracking là đủ
    return (video_dir / f"{seq_name}_tracking.mp4").exists()


def get_saved_params(seq_name: str):
    """Đọc các tham số đã lưu cho video này từ file cache."""
    if not PARAMS_CACHE_FILE.exists():
        return None
    try:
        with open(PARAMS_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            return cache.get(seq_name)
    except json.JSONDecodeError:
        return None

def save_current_params(seq_name: str, detection_params, tracking_params):
    """Lưu trữ tham số hiện tại vào file cache."""
    cache = {}
    if PARAMS_CACHE_FILE.exists():
        try:
            with open(PARAMS_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            pass # Bỏ qua nếu file bị lỗi
    
    current_params = {
        'detection': detection_params,
        'tracking': tracking_params
    }
    cache[seq_name] = current_params
    
    with open(PARAMS_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)


def convert_all_videos_to_h264(video_dir: Path, seq_name: str, status_placeholder):
    """Chuyển tất cả video sang định dạng H.264 (ghi đè file gốc)."""
    if not video_dir.exists():
        status_placeholder.error(f"❌ Không tìm thấy thư mục video: {video_dir}")
        return False

    mp4_files = list(video_dir.glob(f"{seq_name}_*.mp4"))
    if not mp4_files:
        status_placeholder.warning("⚠️ Không tìm thấy video .mp4 nào để chuyển đổi.")
        return True # Vẫn coi là thành công

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
        # status_placeholder.success(f"✅ Đã chuyển: {', '.join(converted)}")
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
st.session_state.setdefault('selected_video_type', 'Tracking')

# --- Sidebar Upload ---
st.sidebar.header("📁 Upload Video")
uploaded_file = st.sidebar.file_uploader("Chọn video", type=["mp4", "avi", "mov"])

# Khởi tạo các biến tham số với giá trị mặc định (để tránh lỗi ReferenceError nếu không có video)
conf, nms_1, nms_2, penalty_p, penalty_q, tai_thr = 0.1, 0.8, 0.95, 0.2, 0.4, 0.55
start_button = False


if uploaded_file is not None:
    seq_name = Path(uploaded_file.name).stem
    
    # 🌟 THAY ĐỔI Ở ĐÂY: Dùng OUTPUT_PATH_VIDEO
    # Đảm bảo thư mục Input tồn tại
    OUTPUT_PATH_VIDEO.mkdir(parents=True, exist_ok=True) 
    
    # Tạo đường dẫn file trong thư mục Input
    uploaded_video_path_in_input = OUTPUT_PATH_VIDEO / uploaded_file.name
    seq_name = Path(uploaded_file.name).stem

    # Xử lý khi video mới được upload
    if st.session_state.uploaded_file_path is None or uploaded_video_path_in_input != Path(st.session_state.uploaded_file_path):
        with open(uploaded_video_path_in_input, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.uploaded_file_path = str(uploaded_video_path_in_input)
        st.session_state.video_name = uploaded_file.name
        st.session_state.current_seq_name = seq_name
        st.session_state.video_processed = check_existing_processed_videos(seq_name)

        if st.session_state.video_processed:
            st.sidebar.success(f"🟢 Video '{seq_name}' đã được xử lý trước đó.")
        else:
            st.sidebar.info("🕓 Video mới, cần xử lý pipeline.")
        # Dùng st.rerun() để cập nhật giao diện sau khi upload
        st.rerun()

    # --- HIỂN THỊ CÁC THAM SỐ VÀ NÚT START SAU KHI UPLOAD ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Parameters")

    conf = st.sidebar.slider("Confidence threshold (conf)", 0.0, 1.0, 0.1, 0.05, help="")
    # Đảm bảo nms_2 luôn lớn hơn nms_1
    nms_1 = st.sidebar.slider("NMS 1 threshold (nms_1)", 0.0, 1.0, 0.8, 0.05, help="")
    nms_2 = st.sidebar.slider("NMS 2 threshold (nms_2)", nms_1 + 0.05, 1.0, 0.95, 0.05, help="")
    
    # --- Sidebar: Tracking Parameters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tracking Parameters")

    penalty_p = st.sidebar.slider("Penalty p", 0.0, 1.0, 0.2, 0.05, help="")
    penalty_q = st.sidebar.slider("Penalty q", 0.0, 1.0, 0.4, 0.05, help="")
    tai_thr = st.sidebar.slider("TAI Threshold", 0.0, 1.0, 0.55, 0.05, help="")
    
    # --- Sidebar: Video Output Selector ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎬 Video Output Viewer")

    seq_name = st.session_state.get('current_seq_name')
    video_options = {
        "Detection": f"{seq_name}_detection.mp4",
        "Trajectory": f"{seq_name}_trajectory.mp4",
        "Tracking": f"{seq_name}_tracking.mp4",
        "Combine": f"{seq_name}_combine.mp4",
    }

    selected_video_type = st.sidebar.selectbox(
        "Chọn loại video:",
        list(video_options.keys()),
        index=list(video_options.keys()).index(st.session_state['selected_video_type']),
        help="Chọn video kết quả: detection, tracking, predict hoặc final."
    )
    st.session_state['selected_video_type'] = selected_video_type

    st.sidebar.markdown("---")
    
    # Logic kiểm tra cần chạy lại không
    current_detection_params = {'conf': conf, 'nms_1': nms_1, 'nms_2': nms_2}
    current_tracking_params = {'penalty_p': penalty_p, 'penalty_q': penalty_q, 'tai_thr': tai_thr}
    
    saved_params = get_saved_params(seq_name)
    
    if not st.session_state.video_processed:
        # Chưa xử lý lần nào, chắc chắn cần chạy
        run_required = True
        status_msg = "Chưa xử lý lần nào. Nhấn **Start Tracking**."
    elif saved_params and saved_params.get('detection') == current_detection_params and saved_params.get('tracking') == current_tracking_params:
        # Đã xử lý và tham số không thay đổi
        run_required = False
        status_msg = "Tham số **không đổi**. Kết quả đã sẵn sàng."
    else:
        # Đã xử lý nhưng tham số thay đổi hoặc không tìm thấy tham số cũ
        run_required = True
        status_msg = "⚠️ Nếu thay đổi tham số thì cần xử lý lại."

    if run_required:
        start_button = st.sidebar.button("▶️ Start Tracking", type="primary")
        st.sidebar.info(status_msg)
    else:
        # Giữ nút Start ở trạng thái 'đã hoàn thành' nếu không cần chạy lại
        st.sidebar.success(status_msg)
        start_button = st.sidebar.button("▶️ Run Tracking Again", help="Buộc chạy lại pipeline dù tham số không đổi.")
        if start_button:
            # Nếu người dùng bấm Run Tracking Again, thì set run_required = True để chạy pipeline
            pass

else:
    st.sidebar.info("Vui lòng tải video lên để hiển thị các tham số.")


# --- Hiển thị video ---
if st.session_state.uploaded_file_path and not st.session_state.is_running:
    seq_name = st.session_state.current_seq_name
    video_dir = OUTPUT_PATH_BASE / seq_name / "videos"

    if st.session_state.video_processed:
        # Video đã xử lý, hiển thị kết quả
        selected_video_name = video_options[st.session_state['selected_video_type']]
        selected_video_path = video_dir / selected_video_name

        if selected_video_path.exists():
            st.video(str(selected_video_path))
            st.info(f"📁 Đang xem: **{selected_video_path.name}**")
        else:
            st.warning(f"❌ Không tìm thấy video kết quả: **{selected_video_path.name}**. Vui lòng chạy lại pipeline.")
    else:
        # Video đã upload nhưng chưa xử lý
        st.info("📹 Video mới — hãy điều chỉnh tham số và nhấn **Start Tracking** để bắt đầu.")

elif st.session_state.is_running:
    st.warning("⏳ Đang xử lý, vui lòng chờ...")


# --- Pipeline ---
def run_full_pipeline(seq_name):
    st.session_state.is_running = True
    progress = st.progress(0)
    status = st.empty()

    # Lấy tham số hiện tại từ Slider
    detection_params = {'conf': conf, 'nms_1': nms_1, 'nms_2': nms_2}
    tracking_params = {'penalty_p': penalty_p, 'penalty_q': penalty_q, 'tai_thr': tai_thr}
    
    # 1. Ghi lại config
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
        
        status.info("Bước 6/6: Convert video to H.264 (Để tương thích Streamlit)...")
        video_dir = OUTPUT_PATH_BASE / seq_name / "videos"
        convert_all_videos_to_h264(video_dir, seq_name, status)
        
        # 2. Lưu tham số sau khi chạy thành công
        save_current_params(seq_name, detection_params, tracking_params)
        
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
    # Nếu nút 'Start Tracking' hoặc 'Run Tracking Again' được nhấn
    run_full_pipeline(st.session_state.current_seq_name)
elif start_button and st.session_state.is_running:
    st.warning("⚠️ Pipeline đang chạy, vui lòng chờ.")
elif start_button and not st.session_state.uploaded_file_path:
    st.error("Vui lòng tải video trước khi chạy pipeline.")