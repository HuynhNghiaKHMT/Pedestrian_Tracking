import subprocess
import sys
import os
import configparser
import argparse
import torch
import numpy as np
import pickle
from tqdm import tqdm

def install_ultralytics():
    command = [
        sys.executable,
        "-m", "pip",
        "install",
        "ultralytics",
        "--no-deps"
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("Cài đặt ultralytics thành công")
    except subprocess.CalledProcessError() as e:
        print("Lỗi cài đặt: ", e)

def make_parser(model_name, seq_name, output_path, nms_thr):
    parser = argparse.ArgumentParser("YOLO")

    # Can be changed
    parser.add_argument("--f", "--dummy", type=str, default=None, help="Dummy arg")
    parser.add_argument("-c", "--ckpt", 
                        default=f"{model_name}.pt",
                        type=str, help="ckpt for eval")
    parser.add_argument("-n","--exp_name", type=str,
                        default=f"{output_path}/{seq_name}/1. det/seq_{str(nms_thr)}_{model_name}.pickle")
    
    # Fixed
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn",)
    parser.add_argument("--fp16", dest="fp16", default=True, action="store_true",)

    # Det args
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=nms_thr, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=10000, type=int, help="eval seed")

    return parser


def detect(args, input_path, mode):

    # detection
    ckpt_file = args.ckpt


    # val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)

    # Import model
    from ultralytics import YOLO 
    model = YOLO(ckpt_file)

    vid_path = f"{input_path}/image_seq/{mode}/" # ../Inputs/image_seq/test

    res = {}
    for vid in os.listdir(vid_path):
        res[vid] = {}
        img1_path = os.path.join(vid_path, vid, "img1") # ../Inputs/image_seq/test/video2/img1
        seqinfo_path = os.path.join(vid_path, vid, "seqinfo.ini")
        seqinfo = configparser.ConfigParser()
        seqinfo.read(seqinfo_path)
        
        width = int(seqinfo["Sequence"]["imWidth"])
        height = int(seqinfo["Sequence"]["imHeight"])
        img_size = (height, width)  # Input size (h, w)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for image in tqdm(os.listdir(img1_path), desc=f"Frames in {vid}"):
            frame_id = int(os.path.splitext(image)[0])
            frame_dets = []
            image_path = f"{img1_path}/{image}"
            results = model.predict(image_path, imgsz=img_size, conf=args.conf, iou=args.nms, verbose=False, device=device, classes=[0])

            if results[0].boxes is not None:
                boxes = results[0].boxes.cpu().numpy()
                for box in boxes:
                    cls = box.cls.item()
                    conf = box.conf.item()
                    xyxy = box.xyxy
                    x1, y1, x2, y2 = xyxy[0][0], xyxy[0][1], xyxy[0][2], xyxy[0][3]
                    # Nếu cần normalized center: uncomment dưới và thay append
                    # w = x2 - x1; h = y2 - y1; cx = x1 + w/2; cy = y1 + h/2
                    # img_w, img_h = results[0].orig_shape
                    # cx_norm, cy_norm = cx/img_w, cy/img_h; w_norm, h_norm = w/img_w, h/img_h
                    # frame_dets.append([cx_norm, cy_norm, w_norm, h_norm, conf, int(cls)])
                    frame_dets.append([x1, y1, x2, y2, conf, int(cls)])  # Pixel xyxy + conf + cls
                    
            res[vid][frame_id] = np.array(frame_dets)
    
    # Lưu pickle
    with open(args.exp_name, "wb") as f:
        pickle.dump(res, f)
    print(f"Detections saved to {args.exp_name}")

def run():
    install_ultralytics()

    # Đọc file config
    config_env = configparser.ConfigParser()
    config_env.read('../env.ini')

    model_name = config_env.get("Model", "model")
    input_path = config_env.get("Path", "input_path")
    mode = config_env.get("General", "mode")
    # Get video name
    output_path = config_env.get("Path", "output_path")
    input_video = config_env.get("Input", "input_video")
    seq_name = os.path.splitext(os.path.basename(input_video))[0]

    for nms_thr in [0.8, 0.95]:
        filtered_argv = [arg for arg in sys.argv if not arg.startswith('--HistoryManager')]
        args = make_parser(model_name, seq_name, output_path, nms_thr).parse_args(filtered_argv[1:])


        detect(args, input_path, mode)