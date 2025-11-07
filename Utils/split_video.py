import os
import cv2
import json
import math
import configparser
import argparse
import sys


# Hàm tạo folder ảnh
def extract_frames_to_mot17_format(input_video, input_path, mode, seq_name,  img_dir_name='img1'):
    """
    Extract frames from a video and save them in MOT17-like format:
    - Create directory: img_folder_path/seq_name/img_dir_name/ with img000001.jpg, img000002.jpg, etc.
    - Create seqinfo.ini in img_folder_path/ with sequence metadata.
    """
    # Create image_seq path
    image_seq_path = os.path.join(input_path, "image_seq")
    mode_folder = os.path.join(image_seq_path, mode)
    # Create output directories
    seq_path = os.path.join(mode_folder, seq_name) # Vd: image_seq/test/video2
    img_path = os.path.join(seq_path, img_dir_name)
    os.makedirs(img_path, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = math.ceil(width / 32) * 32 # Chia het cho 32
    height = math.ceil(height / 32) * 32 # Chia het cho 32
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: FPS={fps}, Resolution={width}x{height}, Frames={total_frames}")
    
    # Extract frames
    frame_id = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame as JPG with zero-padded name
        frame_filename = f"{frame_id:06d}.jpg"
        frame_path = os.path.join(img_path, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_id += 1
    
    cap.release()
    
    # Create seqinfo.ini
    seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')  # image_seq/test/video2/seqinfo.ini
    with open(seqinfo_path, 'w') as f:
        f.write(f"""[Sequence]
name = {seq_name}
imgDir = {img_dir_name}
rameRate = {fps}
seqLength = {frame_id-1}
imWidth = {width}
imHeight = {height}
imgExt = .jpg

""")
    
    print(f"Extraction complete! Frames saved to: {img_path}")
    print(f"Seqinfo.ini saved to: {seqinfo_path}")


# Hàm tạo file json
def create_anno_json(json_path, input_path, seq_name, mode):

    seqinfo_path = f"{input_path}/image_seq/{mode}/{seq_name}/seqinfo.ini"
    config = configparser.ConfigParser()
    config.read(seqinfo_path)

    
    # Đọc thông tin từ config
    width = int(config["Sequence"]["imWidth"])
    height = int(config["Sequence"]["imHeight"])
    length = int(config["Sequence"]["seqLength"])

    
    # Đường dẫn lưu file
    out_path = f'{json_path}/{seq_name}.json'

    # Tạo thư mục nếu nó chưa tồn tại. 
    output_dir = os.path.dirname(out_path) 
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    
    # Nội dung file
    ann_cnt = 0
    tid_curr = 0
    image_cnt = 0
    tid_last = -1 
    video_cnt = 1 # video sequence number.
    out = {'info': {"description": "MOT20 Dataset in COCO format",
                    "url": "https://motchallenge.net/",
                    "version": "1.0",
                    "year": 2025,
                    "contributor": "Your Name",
                    "date_created": "2025/09/29"},
                    'images': [], 'annotations': [], 'videos': [],
                   'categories': [{'id': 1, 'name': 'pedestrian'}]}
    out['videos'].append({'id': video_cnt, 'file_name': seq_name})

    
    # Tạo nội dung
    num_images = length
    image_range = [0, num_images-1]
    for i in range(num_images):
        if i < image_range[0] or i > image_range[1]:
            continue
        else:
            image_info = {'file_name': 'image_seq/{}/{}/img1/{:06d}.jpg'.format(mode, seq_name, i + 1),  # image name VD: ../Input/image_seq/test/video2/img1/000001.jpg
                          'id': image_cnt + i + 1,  # image number in the entire training set.
                          'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                          'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                          'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                          'video_id': video_cnt,
                          'height': height, 'width': width}
            out['images'].append(image_info)


    # Ghi kết quả
    json.dump(out, open(out_path, 'w'))
    
    # In ra log
    print("Tạo thành công file json tại ", out_path)


# Hàm tạo file experiment
def make_exp_file(json_path, seq_name, exp_path, input_path):
    code = f"""
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = '{seq_name}'
        self.train_ann = ""
        self.val_ann = "{json_path}/{seq_name}.json"
        # self.input_size = (800, 1440)
        # self.test_size = (800, 1440)
        self.input_size = (384, 672)
        self.test_size = (384, 672)
        self.random_size = (18, 32)
        self.max_epoch = 80
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.1
        self.nmsthre = 0.95
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.data_dir = "{input_path}"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name='',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {{"num_workers": self.data_num_workers, "pin_memory": True}}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name='',
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {{
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }}
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
"""

    with open(f"{exp_path}/yolox_{seq_name}.py", "w") as f:
        f.write(code)

    print("Tạo thành công file exp tại, ", f"{exp_path}/yolox_{seq_name}.py")


# Hàm chính 
def video_preprocess():

    # Đọc file config
    config_env = configparser.ConfigParser()
    config_env.read('env.ini')


    # Config
    root_dir = config_env.get("Path", "root_path")
    input_path = config_env.get("Path", "input_path")
    mode = config_env.get("General", "mode")
    input_video = config_env.get("Input", "input_video")
    seq_name = os.path.splitext(os.path.basename(input_video))[0]
    exp_path = config_env.get("YOLOX", "exp_path")
    json_path = config_env.get("YOLOX", "json_path")


    # Chạy các hàm tách video, tạo file json, tạo file exp
    extract_frames_to_mot17_format(input_video, input_path, mode, seq_name)
    create_anno_json(json_path, input_path, seq_name, mode)
    make_exp_file(json_path, seq_name, exp_path, input_path)