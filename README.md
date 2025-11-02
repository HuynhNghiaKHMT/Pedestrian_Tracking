# Đồ án môn học 🏆 <br>
## Tên đề tài: Theo dõi người đi bộ trong Video (Pedestrian Tracking in Videos) 🚶 📹
## 👨‍🏫 Giảng viên hướng dẫn
| Giảng viên        | Email                 |
|-------------------|-------------------|
| Mai Tiến Dũng | dungmt@uit.edu.vn |
## 📬 Thông tin thành viên nhóm

| Họ và Tên         | MSSV     | Email                 |GitHub                                      |
|-------------------|----------|------------------------|--------------------------------------------|
| Huỳnh Trung Nghĩa | 22520945 | 22520945@gm.uit.edu.vn | [HuynhNghiaKHMT](https://github.com/HuynhNghiaKHMT) |
| Huỳnh Chí Nhân | 22520996 | 22520996@gm.uit.edu.vn | [nhanhuynh123](https://github.com/nhanhuynh123) |
| Nguyễn Hồng Phát | 22521076 | 22521076@gm.uit.edu.vn | [hongphat13](https://github.com/hongphat13) |


# 🔤 Nội dung mã nguồn  
## 📦 Công nghệ và thư viện sử dụng
## 📂 Cấu trúc thư mục
```bash
demo
├── demo/
    └── demo_Tracktrack.py/
├── Input/
    └── videos/
        └── <input-video>.mp4
├── Outputs/
    └── <input-video>/
        ├── 1. det/
        ├── 2. det_feat/
        ├── 3. track/
        └── videos/
├── Tracktrack/
    ├── YOLOX/
    ├── FastReID/
    └── Tracker/
├── Utils/
├── env.ini
└── requirements.txt
```
## 🚀 Cài đặt và sử dụng
Để chạy dự án, hãy làm theo các bước sau:

### 1. Clone Repository

```bash
git clone https://github.com/HuynhNghiaKHMT/Pedestrian_Tracking.git
cd Pedestrian_Tracking
```

### 2. Tạo môi trường ảo
Việc tạo môi trường ảo sẽ giúp bạn dễ dàng quản lí các phiên bản thư viện, giúp dễ cài đặt và sửa chữa, tránh lỗi phiên bản. <br>
Khởi tạo môi trường ảo, khuyến khích dùng python=3.11.x. <br>

### 3. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```


## 📝 Đánh giá mô hình trên toàn bộ Dataset

| Dataset | Mode | Model | HOTA↑ | AssA | MOTA↑  | IDF1↑ | IDsw↓ | Frag↓ |
|--------------|--------|--------|-------|-------|------|------|------|------|
|  |  | **YOLOX_X**  | 69.1% | 72.7 |	79.7% | 85.0% | 40.0 | 87.0 |
| ***MOT17*** | *val* | **YOLOv5_X** |  |  |  |  |  |  |
|  |  | **YOLOv12_X** | 58.7% | 62.9 | 59.2% | 70.4% | 89.0 | 219.0 |
|  |  | **YOLOX_X**  | 67.1% | 68.1 | 81.6% | 83.0%	| 822 | 1341.0 |
| ***MOT17*** | *test* | **YOLOv5_X** |  |  |  |  |  |  |
|  |  | **YOLOv12_X** | 48.6% | 52.7 | 51% | 61% | 1014 | 2199 |


## ✉️ Nội dung file **env.ini**

<h3>[Path]</h3>
<p>root_path = .. (đường dẫn chính, cố định)</p>
<p>input_path = ../Input (đường dẫn chính đến thư mục Input, cố định)</p>
<p>output_path = ../Outputs (đường dẫn chính đến thư mục Output, cố định)</p>

<h3>[General]</h3>
<p>mode = test (cố định)</p>

<h3>[Input]</h3>
<p>input_video = ../Input/videos/video2.mp4 (đường dẫn đến video input)</p>

<h3>[Model]</h3>
<p>data2model= mot17 (dùng cho lựa chọn mô hình detect, reid (mot17: thông thường, mot20: cảnh đông đúc))</p>
<p>model= yolox (mô hình sử dụng, [yoloxx, yolov5x, yolov12x])</p>

<h3>[YOLOX]</h3>
<p>exp_path = ../Tracktrack/YOLOX/exps (Đường dẫn folder exp, cố định)</p>
<p>json_path = ../Tracktrack/YOLOX/json (Đường dẫn folder json cho yolox detect, cố định)</p>
<p>weight_path = ../Tracktrack/YOLOX/weights (Đường dẫn folder weight chứa trọng số cho yolox, cố định)</p>

<h3>[ReID]</h3>
<p>weight_path = ../Tracktrack/FastReID/weights (Đường dẫn folder weight chứa trọng số cho model reid, cố định)</p>
<p>config_path = ../Tracktrack/FastReID/configs (Đường dẫn folder configs, cố định)</p>

<h3>[Track]</h3>
<p>af_link = ../Tracktrack/Tracker/AFLink/AFLink_epoch20.pth (Đường dẫn trọng số model AFLink, cố định)</p>

## 🏃 Demo


1. Thêm video vào thư mục ../Input/videos

2. Sửa đường dẫn của input_video trong env.ini:

```
[Input]
input_video = ../Input/videos/<tên-video>.mp4
```

3. Chạy chương trình tại thư mục gốc:
```bash
python demo_Tracktrack.py
```

4. Video kết quả được lưu tại ../Output/<tên-video>/videos

