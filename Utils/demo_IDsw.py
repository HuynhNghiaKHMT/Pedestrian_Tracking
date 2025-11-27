import cv2
import os
import pandas as pd
import numpy as np
from typing import Set, Dict, Tuple, Optional, List

# ==============================================================================
# 1. CẤU HÌNH CHUNG & ĐƯỜNG DẪN (ĐÃ CẬP NHẬT ID MẪU)
# ==============================================================================
id_video = "clip7"
path_root = r"D:\F.NCKH\EXP" 
img_folder = rf"D:\HKVII\CS420\Pedestrian_Tracking\Input\image_seq\test\{id_video}\img1"

# Đường dẫn TUYỆT ĐỐI của file dữ liệu
GT_FILE_PATH = f"D:\\HKVII\\CS420\\Pedestrian_Tracking\\Outputs\\{id_video}\\3. track\\seq_0.8\\{id_video}.txt"
PREDICT_FILE_PATH = f"D:\\HKVII\\CS420\\Pedestrian_Tracking\\Outputs\\{id_video}\\3. track\\seq_0.8\\{id_video}.txt"

# Cập nhật ID MẪU để phù hợp với dữ liệu 1-12 bạn cung cấp
gt_id: Set[int] = {} # GT ID Mới (Đã thay thế 19, 20, 21, 23)
predict_id: Set[int] = {27, 68, 14, 63, 109, 81, 112} # Predict ID Mới (Đã thay thế 3, 19, 33, 40)

# Sắp xếp và chuyển đổi ID thành chuỗi để dùng trong tên file
predict_id_list_str = '_'.join(map(str, sorted(list(predict_id))))
gt_id_list_str = '_'.join(map(str, sorted(list(gt_id))))

# Cập nhật tên file video (ĐƯỜNG DẪN TUYỆT ĐỐI)
predict_video_name = rf"MOT17-{id_video}_predict_MultiID_{predict_id_list_str}.mp4"
gt_video_name = rf"MOT17-{id_video}_gt_MultiID_{gt_id_list_str}.mp4"
final_video_name = rf"MOT17-{id_video}_compare_gt{gt_id_list_str}_pred{predict_id_list_str}.mp4"
fps = 30

GT_VIDEO_PATH = os.path.join(path_root, gt_video_name)
PREDICT_VIDEO_PATH = os.path.join(path_root, predict_video_name)
FINAL_VIDEO_PATH = os.path.join(path_root, final_video_name)

# === Đọc ảnh (dùng để lấy kích thước) ===
images: List[str] = []
height, width, fourcc = 1080, 1920, 0
try:
    images = sorted([img for img in os.listdir(img_folder) if img.endswith((".jpg", ".png"))],
                    key=lambda x: int(os.path.splitext(x)[0]))
    if not images:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong thư mục: {img_folder}")
    frame_test = cv2.imread(os.path.join(img_folder, images[0]))
    height, width, _ = frame_test.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
except Exception as e:
    print(f"🔴 LỖI KHỞI TẠO: Không thể đọc ảnh hoặc thư mục ảnh không tồn tại. Lỗi: {e}")

# === THIẾT LẬP MÀU ===
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (255, 128, 0), (128, 0, 255), (0, 128, 255),
    (128, 255, 0), (255, 0, 128), (0, 255, 128)
]
def get_color(idx): return colors[idx % len(colors)]

id_color_map: Dict[int, Tuple[int, int, int]] = {}
gt_list = sorted(list(gt_id))
pred_list = sorted(list(predict_id))

# Logic ghép cặp màu
min_len = min(len(gt_list), len(pred_list))
for i in range(min_len):
    color = get_color(i)
    id_color_map[gt_list[i]] = color
    id_color_map[pred_list[i]] = color

for i in range(min_len, len(gt_list)):
    id_color_map[gt_list[i]] = get_color(i)

for i in range(min_len, len(pred_list)):
    id_color_map[pred_list[i]] = get_color(i)

# === CÁC THAM SỐ TIÊU ĐỀ ===
TITLE_POS = (40, height - 30)
TITLE_COLOR = (255, 255, 255)
TITLE_FONT = cv2.FONT_HERSHEY_DUPLEX
TITLE_SCALE = 1.2
TITLE_THICKNESS = 3


# ==============================================================================
# 1️⃣ VIDEO GROUND TRUTH
# ==============================================================================
def export_gt_video(df_gt: pd.DataFrame) -> Optional[str]:
    if df_gt.empty or not images:
        print("Lỗi: Dữ liệu GT rỗng hoặc không có ảnh. Bỏ qua GT video.")
        return None

    print(f"\n=== Xuất video Ground Truth (Vẽ ID: {gt_id_list_str}) ===")
    
    # Lọc dữ liệu CHỈ theo ID (đã loại bỏ điều kiện df['score'] == 1)
    df_gt_filtered = df_gt[df_gt["id"].isin(gt_id)]

    if df_gt_filtered.empty:
        print("🔴 CẢNH BÁO: Không có detection nào khớp với GT ID đã chọn trong toàn bộ sequence.")
        return None
    
    try:
        video_gt = cv2.VideoWriter(GT_VIDEO_PATH, fourcc, fps, (width, height))
    except Exception as e:
        print(f"LỖI: Không thể khởi tạo VideoWriter cho GT. Lỗi: {e}")
        return None

    trackers_gt = {id_val: [] for id_val in gt_id}
    TITLE_GT = f"MOT17-{id_video} | GROUND TRUTH | IDs: {gt_id_list_str}"

    for idx, img_name in enumerate(images, start=1):
        frame = cv2.imread(os.path.join(img_folder, img_name))
        
        if frame is None:
            continue

        detections_in_frame = df_gt_filtered[df_gt_filtered["frame"] == idx]

        current_ids_present = set()

        for _, det in detections_in_frame.iterrows():
            curr_id = int(det["id"])
            current_ids_present.add(curr_id)
            color = id_color_map.get(curr_id, (255, 255, 255))

            x, y, w, h = det[["x", "y", "w", "h"]]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"GT:{curr_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cx, cy = int(x + w/2), int(y + h)
            trackers_gt[curr_id].append((cx, cy))

        for id_val, track in trackers_gt.items():
            if len(track) >= 1:
                color = id_color_map.get(id_val, (255, 255, 255))
                for i in range(1, len(track)):
                    cv2.line(frame, track[i-1], track[i], color, 2)
                if id_val in current_ids_present:
                    cv2.circle(frame, track[-1], 4, color, -1)

        cv2.putText(frame, TITLE_GT, TITLE_POS, TITLE_FONT, TITLE_SCALE, TITLE_COLOR, TITLE_THICKNESS)
        video_gt.write(frame)

    video_gt.release()
    print(f"✅ GT video saved: {GT_VIDEO_PATH}")
    return GT_VIDEO_PATH


# ==============================================================================
# 2️⃣ VIDEO PREDICT
# ==============================================================================
def export_predict_video(df_pred: pd.DataFrame) -> Optional[str]:
    if df_pred.empty or not images:
        print("Lỗi: Dữ liệu Predict rỗng hoặc không có ảnh. Bỏ qua Predict video.")
        return None

    print(f"\n=== Xuất video Predict (Vẽ ID: {predict_id_list_str}) ===")
    
    df_pred_filtered = df_pred[df_pred["id"].isin(predict_id)]

    if df_pred_filtered.empty:
        print("🔴 CẢNH BÁO: Không có detection nào khớp với Predict ID đã chọn trong toàn bộ sequence.")
        return None

    try:
        video_pred = cv2.VideoWriter(PREDICT_VIDEO_PATH, fourcc, fps, (width, height))
    except Exception as e:
        print(f"LỖI: Không thể khởi tạo VideoWriter cho Predict. Lỗi: {e}")
        return None

    trackers_pred = {id_val: [] for id_val in predict_id}
    TITLE_PRED = f"MOT17-{id_video} | PREDICTED TRACKS | IDs: {predict_id_list_str}"

    for idx, img_name in enumerate(images, start=1):
        frame = cv2.imread(os.path.join(img_folder, img_name))
        
        if frame is None:
            continue

        detections_in_frame = df_pred_filtered[df_pred_filtered["frame"] == idx]

        current_ids_present = set()

        for _, det in detections_in_frame.iterrows():
            curr_id = int(det["id"])
            current_ids_present.add(curr_id)
            color = id_color_map.get(curr_id, (255, 255, 255))

            x, y, w, h = det[["x", "y", "w", "h"]]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Pred:{curr_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cx, cy = int(x + w/2), int(y + h)
            trackers_pred[curr_id].append((cx, cy))

        for id_val, track in trackers_pred.items():
            if len(track) >= 1:
                color = id_color_map.get(id_val, (255, 255, 255))
                for i in range(1, len(track)):
                    cv2.line(frame, track[i-1], track[i], color, 2)
                if id_val in current_ids_present:
                    cv2.circle(frame, track[-1], 4, color, -1)

        cv2.putText(frame, TITLE_PRED, TITLE_POS, TITLE_FONT, TITLE_SCALE, TITLE_COLOR, TITLE_THICKNESS)
        video_pred.write(frame)

    video_pred.release()
    print(f"✅ Predict video saved: {PREDICT_VIDEO_PATH}")
    return PREDICT_VIDEO_PATH


# ==============================================================================
# 3️⃣ GHÉP VIDEO
# ==============================================================================
def combine_videos(gt_path: str, pred_path: str, final_path: str):
    print("\n=== Tạo video Final (GT + Predict) ===")

    cap_gt = cv2.VideoCapture(gt_path)
    cap_pred = cv2.VideoCapture(pred_path)

    if not cap_gt.isOpened():
        print(f"LỖI: Không thể mở video GT tại: {gt_path}")
        return
    if not cap_pred.isOpened():
        print(f"LỖI: Không thể mở video Predict tại: {pred_path}")
        return

    width_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    final_width = width_gt * 2
    final_height = height_gt

    final_video = cv2.VideoWriter(final_path, fourcc, fps, (final_width, final_height))

    if not final_video.isOpened():
        print(f"LỖI: Không thể khởi tạo VideoWriter cho video cuối cùng tại: {final_path}")
        cap_gt.release()
        cap_pred.release()
        return

    while True:
        ret_gt, frame_gt = cap_gt.read()
        ret_pred, frame_pred = cap_pred.read()

        if not (ret_gt and ret_pred):
            break

        if frame_gt.shape[:2] == frame_pred.shape[:2]:
            combined = cv2.hconcat([frame_gt, frame_pred])
            final_video.write(combined)
        else:
            print("WARNING: Kích thước frame không khớp, dừng ghép video.")
            break

    cap_gt.release()
    cap_pred.release()
    final_video.release()
    print(f"✅ Final video saved: {final_path}")


# ==============================================================================
# 7. KHỐI THỰC THI CHÍNH
# ==============================================================================
if __name__ == "__main__":

    df_gt_loaded = None
    df_pred_loaded = None

    # --- 1. Đọc file và Ép kiểu DType ---
    try:
        df_gt_loaded = pd.read_csv(GT_FILE_PATH, header=None).iloc[:, :9]
        df_gt_loaded.columns = ["frame", "id", "x", "y", "w", "h", "score", "class", "visibility"]

        df_gt_loaded['frame'] = df_gt_loaded['frame'].astype(int)
        df_gt_loaded['id'] = df_gt_loaded['id'].astype(int)

        available_gt_ids = df_gt_loaded['id'].unique()
        missing_gt_ids = gt_id - set(available_gt_ids)
        if missing_gt_ids:
            print(f"⚠️ CHẨN ĐOÁN GT: Các ID sau KHÔNG có trong file tracking: {missing_gt_ids}")
            print(f"   Các ID có sẵn trong file (Tối đa 10 ID đầu): {sorted(list(available_gt_ids))[:10]}...")
        
    except FileNotFoundError:
        print(f"🔴 Lỗi: Không tìm thấy file GT tại: {GT_FILE_PATH}. Bỏ qua GT video.")
    except Exception as e:
        print(f"🔴 Lỗi khi đọc hoặc xử lý file GT: {e}")

    try:
        df_pred_loaded = pd.read_csv(PREDICT_FILE_PATH, header=None).iloc[:, :9]
        df_pred_loaded.columns = ["frame", "id", "x", "y", "w", "h", "score", "class", "visibility"]

        df_pred_loaded['frame'] = df_pred_loaded['frame'].astype(int)
        df_pred_loaded['id'] = df_pred_loaded['id'].astype(int)

        available_pred_ids = df_pred_loaded['id'].unique()
        missing_pred_ids = predict_id - set(available_pred_ids)
        if missing_pred_ids:
            print(f"⚠️ CHẨN ĐOÁN PREDICT: Các ID sau KHÔNG có trong file tracking: {missing_pred_ids}")
            print(f"   Các ID có sẵn trong file (Tối đa 10 ID đầu): {sorted(list(available_pred_ids))[:10]}...")
             
    except FileNotFoundError:
        print(f"🔴 Lỗi: Không tìm thấy file Predict tại: {PREDICT_FILE_PATH}. Bỏ qua Predict video.")
    except Exception as e:
        print(f"🔴 Lỗi khi đọc hoặc xử lý file Predict: {e}")
    
    # --- 2. Thực thi Xuất video ---
    gt_video_output_path = None
    if df_gt_loaded is not None and images:
        gt_video_output_path = export_gt_video(df_gt_loaded)

    pred_video_output_path = None
    if df_pred_loaded is not None and images:
        pred_video_output_path = export_predict_video(df_pred_loaded)

    # --- 3. Ghép video ---
    if gt_video_output_path and pred_video_output_path:
        combine_videos(gt_video_output_path, pred_video_output_path, FINAL_VIDEO_PATH)
    else:
        print("\nKhông thể tạo video so sánh vì thiếu một hoặc cả hai video GT/Predict.")
