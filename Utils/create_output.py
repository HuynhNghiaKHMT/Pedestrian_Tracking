import os
import cv2
import sys
import configparser
import pickle
import pandas as pd

def create_output_video(output_path, video_root, seq_name, img_folder_path, mode):
    # === Định nghĩa Hàm Màu Toàn Cục ===
    # Định nghĩa màu ngẫu nhiên cho từng ID
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (100, 100, 200)]
    def get_color_simple(obj_id):
        """Trả về màu (BGR) dựa trên ID đối tượng để đảm bảo cùng ID có cùng màu."""
        return colors[obj_id % len(colors)]
    # ====================================


    # === Cấu hình đường dẫn ===
    ## Đường dẫn folder
    img_folder = fr"{img_folder_path}/{mode}/{seq_name}/img1"
    path_root = fr"{output_path}/{seq_name}"
    det_file_rel = fr"1. det/seq_0.8.pickle" 
    predict_file_rel = fr"3. track/seq_0.8/{seq_name}.txt" 

    ## Đường dẫn video
    original_video_name = fr"{seq_name}_original.mp4" 
    detection_video_name = fr"{seq_name}_detection.mp4"
    tracking_video_name = fr"{seq_name}_tracking.mp4"
    trajectory_video_name = fr"{seq_name}_trajectory.mp4"
    combine_video_name = fr"{seq_name}_combine.mp4"
    fps = 30 # tùy theo từng video mà đặt fps cho phù hợp

    # === Đường dẫn tệp tin đầu vào (Input Files) ===
    detection_file_path = os.path.join(path_root, det_file_rel)
    tracking_file_path = os.path.join(path_root, predict_file_rel)

    # === Đường dẫn Video đầu ra (Output Video Paths) ===
    video_root = video_root
    original_video_path = os.path.join(video_root, f"{original_video_name}")
    detection_video_path = os.path.join(video_root, f"{detection_video_name}")
    tracking_video_path = os.path.join(video_root, f"{tracking_video_name}")
    trajectory_video_path = os.path.join(video_root, f"{trajectory_video_name}")
    combine_video_path = os.path.join(video_root, f"{combine_video_name}")


    # Lấy danh sách ảnh và sắp xếp (Bạn cần đảm bảo biến 'images' được định nghĩa)
    try:
        images = [img for img in os.listdir(img_folder) if img.endswith((".jpg", ".png"))]
        images.sort() 
        if not images:
            print("Lỗi: Không tìm thấy ảnh nào trong thư mục img_folder.")
            # exit()
        frame_test = cv2.imread(os.path.join(img_folder, images[0]))
        height, width, layers = frame_test.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec mp4
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")
        # exit()
    # ... (Phần in đường dẫn)
    print(f"Path file detection.txt: {detection_file_path}")
    print(f"Path file tracking.txt: {tracking_file_path}")
    print(f"Path video original: {original_video_path}")
    print(f"Path video detection: {detection_video_path}")
    print(f"Path video tracking: {tracking_video_path}")
    print(f"Path video trajectory: {trajectory_video_path}")
    print(f"Path video combine: {combine_video_path}")

    # -------------------------
    ## 1. Original Video (Không thay đổi)
    # -------------------------

    video = cv2.VideoWriter(original_video_path, fourcc, fps, (width, height))

    # Thêm từng ảnh vào video
    for image_name in images:
        img = cv2.imread(os.path.join(img_folder, image_name)) 
        if img is not None:
            video.write(img)

    video.release()
    print(f"Video saved at: {original_video_path}")


        # -------------------------
    ## 2. Det Video (Có kiểm tra và debug)
    # -------------------------
    try:
        with open(detection_file_path, 'rb') as f:
            detections = pickle.load(f)

        # 🧩 Debug: hiển thị các key trong pickle
        # print("\n📁 Các key có trong file pickle:", list(detections.keys()))

        # Nếu seq_name không tồn tại trong pickle thì dùng key đầu tiên
        if seq_name not in detections:
            # print(f"⚠️ '{seq_name}' không có trong pickle, dùng key đầu tiên thay thế.")
            seq_name = list(detections.keys())[0]

        # 🧠 Kiểm tra kiểu dữ liệu
        if not isinstance(detections[seq_name], dict):
            # print(f"⚠️ Dữ liệu detections[{seq_name}] không phải dict, bỏ qua phần vẽ detection.")
            detections = {seq_name: {}}

        total_frames = len(detections[seq_name])
        # print(f"✅ Tổng số frame có detection: {total_frames}\n")

        video = cv2.VideoWriter(detection_video_path, fourcc, fps, (width, height))

        for idx, image_name in enumerate(images, start=1):
            frame = cv2.imread(os.path.join(img_folder, image_name))

            # Lấy detection cho frame hiện tại (nếu có)
            detection = detections[seq_name].get(idx, None)

            if detection is None:
                # print(f"⚠️ Frame {idx:04d} không có detection.")
                video.write(frame)
                continue

            # 🧩 Debug: số lượng bbox trong frame
            # print(f"🖼️ Frame {idx:04d}: {len(detection)} bounding boxes.")

            for box in detection:
                try:
                    x1, y1, x2, y2, score, cls = box
                    bbox_color = (0, 255, 0)
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                    label = f"{score:.2f}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
                except Exception as e:
                    print(f"❌ Lỗi khi đọc bbox ở frame {idx}: {e}")
                    continue

            video.write(frame)

        video.release()
        print(f"Video saved at: {detection_video_path}")

    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file DET tại {detection_file_path}")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý DET video: {e}")



    # -------------------------
    ## 3. Predict Video (Áp dụng màu theo ID)
    # -------------------------

    try:
        df_pred = pd.read_csv(tracking_file_path, header=None).iloc[:, :9]
        df_pred.columns = ["frame", "id", "x", "y", "w", "h", "flag", "class", "visibility"]
        df_pred['flag'] = pd.to_numeric(df_pred['flag'], errors='coerce').fillna(0).astype(int)
        video = cv2.VideoWriter(tracking_video_path, fourcc, fps, (width, height))

        for idx, image_name in enumerate(images, start=1):
            frame = cv2.imread(os.path.join(img_folder, image_name))
            detections = df_pred[(df_pred["frame"] == idx)]
            for _, row in detections.iterrows():
                obj_id = int(row["id"])
                x, y, w, h = row["x"], row["y"], row["w"], row["h"]
                visibility = float(row["visibility"])

                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                bbox_color = get_color_simple(obj_id) # <--- SỬ DỤNG MÀU THEO ID

                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

                # Hiển thị ID + visibility
                label = f"{obj_id}"
                cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2) # <--- SỬ DỤNG MÀU THEO ID

            video.write(frame)

        video.release()
        print(f"Video saved at: {tracking_video_path}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file PREDICT tại {tracking_file_path}")
    except Exception as e:
        print(f"Lỗi khi xử lý PREDICT video: {e}")


    # -------------------------
    ## 5. Track Video (Không thay đổi, đã dùng màu theo ID)
    # -------------------------

    try:
        video = cv2.VideoWriter(trajectory_video_path, fourcc, fps, (width, height))
        track_history = {} 
        TRACK_HISTORY_FRAMES = 50 
        
        for idx, image_name in enumerate(images, start=1):
            frame = cv2.imread(os.path.join(img_folder, image_name))
            frame_track = frame.copy() 

            detections = df_pred[(df_pred["frame"] == idx)]
            current_frame_ids = set(detections["id"].unique())
            
            for _, row in detections.iterrows():
                obj_id = int(row["id"])
                x, y, w, h = row["x"], row["y"], row["w"], row["h"]
                
                x_br, y_br = int(x + w), int(y + h) 
                x_bc, y_bc = int(x + w//2), int(y + h) 
                current_point = (x_bc, y_bc)
                
                if obj_id not in track_history:
                    track_history[obj_id] = []
                
                track_history[obj_id].append((idx, current_point)) 
                
                if len(track_history[obj_id]) > TRACK_HISTORY_FRAMES:
                    track_history[obj_id].pop(0)
                    
                # Vẽ bounding box và ID (Đã dùng màu theo ID)
                x1, y1 = int(x), int(y)
                color = get_color_simple(obj_id)
                cv2.rectangle(frame_track, (x1, y1), (x_br, y_br), color, 1)
                
                # label = f"ID {obj_id}"
                # cv2.putText(frame_track, label, (x1, max(0, y1 - 10)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


            # Logic Xóa quỹ đạo của các đối tượng đã biến mất
            for obj_id in list(track_history.keys()): 
                if obj_id not in current_frame_ids:
                    del track_history[obj_id]
                    continue 

                # Vẽ Quỹ đạo (Đã dùng màu theo ID)
                points_with_frame = track_history[obj_id]
                color = get_color_simple(obj_id)
                points = [p[1] for p in points_with_frame] 

                for i in range(1, len(points)):
                    cv2.line(frame_track, points[i-1], points[i], color, 2) 
                    
                    if i == len(points) - 1:
                        cv2.circle(frame_track, points[i], 4, color, -1)

            video.write(frame_track)

        video.release()
        print(f"Video saved at: {trajectory_video_path}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file PREDICT tại {tracking_file_path} để vẽ quỹ đạo.")
    except Exception as e:
        print(f"Lỗi khi xử lý Track Video: {e}")
        
    # -------------------------
    ## 6. Final Video (Ghép 2x2 - Không thay đổi)
    # -------------------------

    try:
        # 1. Khởi tạo Video Readers
        cap_ori = cv2.VideoCapture(original_video_path)
        cap_det = cv2.VideoCapture(detection_video_path)
        cap_pred = cv2.VideoCapture(tracking_video_path)
        cap_track = cv2.VideoCapture(trajectory_video_path)

        # 2. Kiểm tra các video đã được mở chưa
        if not (cap_ori.isOpened() and cap_det.isOpened() and cap_pred.isOpened() and cap_track.isOpened()):
            print("Lỗi: Không thể mở một hoặc nhiều video để ghép. Vui lòng kiểm tra lại đường dẫn.")
            # Dọn dẹp tài nguyên
            cap_ori.release()
            cap_det.release()
            cap_pred.release()
            cap_track.release()
            # exit()

        # Lấy thông số khung hình từ video gốc
        width = int(cap_ori.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_ori.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 3. Tính toán kích thước đầu ra (2x2)
        final_width = width * 2
        final_height = height * 2

        # 4. Khởi tạo Video Writer cho video cuối cùng
        video_final = cv2.VideoWriter(combine_video_path, fourcc, fps, (final_width, final_height))

        # 5. Đọc và ghép từng khung hình
        frame_count = 0
        while True:
            ret_ori, frame_ori = cap_ori.read()
            ret_det, frame_det = cap_det.read()
            ret_pred, frame_pred = cap_pred.read()
            ret_track, frame_track = cap_track.read()

            if not (ret_ori and ret_det and ret_pred and ret_track):
                break

            # Sắp xếp theo bố cục 2x2
            top_row = cv2.hconcat([frame_ori, frame_det])
            bottom_row = cv2.hconcat([frame_track, frame_pred])
            final_frame = cv2.vconcat([top_row, bottom_row]) 

            # Ghi khung hình đã ghép vào video cuối cùng
            video_final.write(final_frame)
            frame_count += 1
            
        # 6. Dọn dẹp và đóng tài nguyên
        cap_ori.release()
        cap_det.release()
        cap_pred.release()
        cap_track.release()
        video_final.release()

        print(f"Video saved at: {combine_video_path}")

    except Exception as e:
        print(f"\n❌ Lỗi khi tạo Final Video: {e}")


def create_video():
    
    # Config
    config_env = configparser.ConfigParser()
    config_env.read("env.ini")
    

    # Get video name
    input_path = config_env.get("Path", "input_path") # Duong dan input
    output_path = config_env.get("Path", "output_path") # Duong dan output
    input_video = config_env.get("Input", "input_video")
    seq_name = os.path.splitext(os.path.basename(input_video))[0]

# Đường dẫn đến folder video
    video_output_path = f"{output_path}/{seq_name}/videos" # VD: ../Outputs/video2/videos
    img_folder_path = f"{input_path}/image_seq"
    mode = config_env.get("General", "mode")

    create_output_video(output_path, video_output_path, seq_name, img_folder_path, mode)


def create_output_folder():
    # Config
    config_env = configparser.ConfigParser()
    config_env.read("env.ini")

    output_path = config_env.get("Path", "output_path")
    input_video = config_env.get("Input", "input_video")
    seq_name = os.path.splitext(os.path.basename(input_video))[0]

    # Create det, feat, track, video folder
    folders = ["1. det", "2. det_feat", "3. track", "videos"]
    
    for folder in folders:
        seq_path = os.path.join(output_path, seq_name)
        folder_path = os.path.join(seq_path, folder)

        os.makedirs(folder_path, exist_ok=True)
        print("Tạo thành công ", folder_path)
