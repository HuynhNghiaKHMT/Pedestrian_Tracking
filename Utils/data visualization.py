import cv2
import os
import pandas as pd
import numpy as np
from typing import Optional, Callable

# ==============================================================================
# 1. Cấu hình Đường dẫn (GIỮ NGUYÊN NHƯ YÊU CẦU CỦA BẠN)
# Vui lòng đảm bảo các đường dẫn này là chính xác trên hệ thống của bạn trước khi chạy.
# ==============================================================================
IMAGE_FOLDER = r"D:\F.NCKH\TrackTrack\MOT17\MOT17\train\MOT17-02-FRCNN\img1"
DET_FILE = r"D:\F.NCKH\EXP\det_sorted.txt"
GT_FILE = r"D:\F.NCKH\TrackTrack\MOT17\MOT17\train\MOT17-02-FRCNN\gt\gt.txt"

# Đường dẫn output cho 3 video
OUTPUT_VIDEO_DET = r"D:\F.NCKH\EXP\MOT17-02_with_det.mp4" # Từ demo.py
OUTPUT_VIDEO_GT_FLAG0 = r"D:\F.NCKH\EXP\MOT17-02_with_flag0_classes.mp4" # Từ demo_gt_with_flag0.py
OUTPUT_VIDEO_GT_FLAG1 = r"D:\F.NCKH\EXP\MOT17-02_with_flag1_visibility.mp4" # Từ demo_gt_with_flag1.py
OUTPUT_VIDEO_GT_ALL = r"D:\F.NCKH\EXP\MOT17-02_GT_ALL.mp4" # Video vẽ cả 2 flag (theo yêu cầu)
# ==============================================================================

# Khai báo màu BGR cho OpenCV
COLOR_GREEN = (0, 255, 0)   # Detection
COLOR_BLUE = (255, 0, 0)    # Ground Truth Flag 0 (Xanh Biển)
COLOR_RED = (0, 0, 255)     # Ground Truth Flag 1 (Đỏ)


def draw_bounding_box(frame: np.ndarray, x: float, y: float, w: float, h: float, color: tuple, label: str):
    """
    Hàm helper để vẽ bounding box và nhãn lên khung hình.
    """
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ==============================================================================
# Hàm 1: Vẽ Detection (Màu Xanh Lá Cây)
# ==============================================================================
def draw_detections(frame: np.ndarray, det_df: pd.DataFrame, frame_idx: int):
    """
    Vẽ tất cả detection trong frame hiện tại với bounding box màu xanh lá cây và confidence score.
    """
    detections = det_df[det_df["frame"] == frame_idx]

    for _, row in detections.iterrows():
        x, y, w, h, score = row["x"], row["y"], row["w"], row["h"], row["score"]
        label = f"Score: {score:.2f}"
        
        draw_bounding_box(frame, x, y, w, h, COLOR_GREEN, label)


# ==============================================================================
# Hàm 2: Vẽ Ground Truth (Flag 0: Xanh Biển, Flag 1: Đỏ + Visibility)
# ==============================================================================
def draw_ground_truth(frame: np.ndarray, gt_df: pd.DataFrame, frame_idx: int, flag_filter: Optional[int] = None):
    """
    Vẽ các Ground Truth có trong frame hiện tại theo quy tắc Flag.
    :param flag_filter: Lọc theo flag (0 hoặc 1). Nếu None, vẽ cả hai.
    """
    if flag_filter is not None:
        detections = gt_df[(gt_df["frame"] == frame_idx) & (gt_df["flag"] == flag_filter)]
    else:
        # Nếu flag_filter là None, vẽ tất cả detection trong frame
        detections = gt_df[gt_df["frame"] == frame_idx]

    for _, row in detections.iterrows():
        obj_id = int(row["id"])
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        flag = int(row["flag"])
        
        # Quyết định màu và nhãn dựa trên giá trị flag
        if flag == 0:
            # Màu Xanh Biển, hiển thị ID và Class
            obj_class = int(row["class"])
            label = f"ID {obj_id} | C{obj_class}"
            color = COLOR_BLUE
        elif flag == 1:
            # Màu Đỏ, hiển thị ID và Visibility
            visibility = float(row["visibility"])
            label = f"ID {obj_id} | Vis: {visibility:.2f}"
            color = COLOR_RED
        else:
            # Bỏ qua các flag khác (ví dụ: flag=2, 3...)
            continue
            
        draw_bounding_box(frame, x, y, w, h, color, label)


# ==============================================================================
# Hàm chung để xử lý và tạo Video
# ==============================================================================
def process_video(
    image_folder: str, 
    data_df: pd.DataFrame, 
    output_video_path: str, 
    drawing_func: Callable[[np.ndarray, pd.DataFrame, int], None]
):
    """
    Tạo video từ chuỗi khung hình bằng cách áp dụng hàm vẽ bounding box tùy chỉnh.
    """
    try:
        # Chuẩn bị danh sách ảnh
        images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))]
        
        # Sắp xếp theo số frame (cần thiết vì tên file thường là số thứ tự)
        # Sử dụng lambda để đảm bảo sắp xếp đúng số thứ tự, không phải thứ tự chữ cái
        images.sort(key=lambda x: int(os.path.splitext(x)[0]))
        
        if not images:
            print(f"Lỗi: Không tìm thấy ảnh trong thư mục: {image_folder}")
            return

        # Lấy kích thước frame
        frame0 = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, _ = frame0.shape

        # Khởi tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not video.isOpened():
             print(f"Lỗi: Không thể mở VideoWriter cho đường dẫn: {output_video_path}")
             return

        print(f"Bắt đầu tạo video: {os.path.basename(output_video_path)}...")
        
        for idx, image_name in enumerate(images, start=1):
            frame = cv2.imread(os.path.join(image_folder, image_name))
            if frame is None:
                continue

            # Gọi hàm vẽ (draw_detections hoặc draw_ground_truth)
            drawing_func(frame, data_df, idx)
            
            video.write(frame)

        video.release()
        print(f"✅ Video đã được lưu thành công tại: {output_video_path}")
        
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình xử lý video: {e}")


# ==============================================================================
# 6. Khối thực thi chính (Chỉ cần gọi hàm)
# ==============================================================================
if __name__ == "__main__":
    
    # ------------------------------------------------------------------
    # 6.1. Đọc dữ liệu Detection và Ground Truth
    # Giữ nguyên cách đọc file (pd.read_csv) theo yêu cầu
    # ------------------------------------------------------------------
    df_det = None
    df_gt = None
    
    try:
        df_det = pd.read_csv(DET_FILE, header=None)
        df_det.columns = ["frame", "id", "x", "y", "w", "h", "score"]
    except FileNotFoundError:
        print(f"🔴 Lỗi: Không tìm thấy file Detection tại: {DET_FILE}. Bỏ qua video DET.")
    except Exception as e:
        print(f"🔴 Lỗi khi đọc file Detection: {e}")

    try:
        df_gt = pd.read_csv(GT_FILE, header=None)
        df_gt.columns = ["frame", "id", "x", "y", "w", "h", "flag", "class", "visibility"]
    except FileNotFoundError:
        print(f"🔴 Lỗi: Không tìm thấy file Ground Truth tại: {GT_FILE}. Bỏ qua video GT.")
    except Exception as e:
        print(f"🔴 Lỗi khi đọc file Ground Truth: {e}")


    # ------------------------------------------------------------------
    # 6.2. Tạo Video (Gọi hàm)
    # ------------------------------------------------------------------

    # 1. Video Detection (Xanh Lá Cây)
    if df_det is not None:
        process_video(
            image_folder=IMAGE_FOLDER, 
            data_df=df_det, 
            output_video_path=OUTPUT_VIDEO_DET, 
            drawing_func=lambda frame, data, idx: draw_detections(frame, data, idx)
        )

    # 2. Video Ground Truth (Flag 0 - Xanh Biển)
    if df_gt is not None:
        process_video(
            image_folder=IMAGE_FOLDER, 
            data_df=df_gt, 
            output_video_path=OUTPUT_VIDEO_GT_FLAG0, 
            drawing_func=lambda frame, data, idx: draw_ground_truth(frame, data, idx, flag_filter=0)
        )

        # 3. Video Ground Truth (Flag 1 - Đỏ + Visibility)
        process_video(
            image_folder=IMAGE_FOLDER, 
            data_df=df_gt, 
            output_video_path=OUTPUT_VIDEO_GT_FLAG1, 
            drawing_func=lambda frame, data, idx: draw_ground_truth(frame, data, idx, flag_filter=1)
        )
        
        # 4. Video Ground Truth (Cả 2 Flag: Flag 0 Xanh Biển, Flag 1 Đỏ)
        process_video(
            image_folder=IMAGE_FOLDER, 
            data_df=df_gt, 
            output_video_path=OUTPUT_VIDEO_GT_ALL, 
            drawing_func=lambda frame, data, idx: draw_ground_truth(frame, data, idx, flag_filter=None)
        )