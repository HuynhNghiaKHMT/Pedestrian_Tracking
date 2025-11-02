import os
import sys
sys.path.append(os.path.dirname(__file__))
import cv2
import pickle
import random
import argparse
import numpy as np
import configparser
from tqdm import tqdm
from fastreid.emb_computer import EmbeddingComputer

def make_parser(det_output, reid_output, nms_thr, data_path, mode, data2model, model_name, config_path, weight_path):
    
    configs = {
        "mot17": f"{config_path}/MOT17/sbs_S50.yml",
        "mot20": f"{config_path}/MOT20/sbs_S50.yml"
    }

    weights = {
        "mot17": f"{weight_path}/mot17_sbs_S50.pth",
        "mot20": f"{weight_path}/mot20_sbs_S50.pth"
    }
    # Initialization
    parser = argparse.ArgumentParser("Track")

    # Data args
    parser.add_argument("-f", "--dummy_arg", type=str, default="")
    parser.add_argument("--dataset", type=str, default=f'{data_path}/')
    parser.add_argument("--data_path", type=str, default=f"{data_path}/{mode}/")
    parser.add_argument("--pickle_path", type=str, default=f"{det_output}/seq_{str(nms_thr)}_{model_name}.pickle")
    parser.add_argument("--output_path", type=str, default=f"{reid_output}/seq_{str(nms_thr)}_{model_name}.pickle")
    parser.add_argument("--config_path", type=str, default=configs[data2model])
    parser.add_argument("--weight_path", type=str, default=weights[data2model])

    # Else
    parser.add_argument("--seed", type=float, default=10000)

    return parser

def reid():

    # Config
    config_env = configparser.ConfigParser()
    config_env.read('../env.ini')

    # Get video name
    output_path = config_env.get("Path", "output_path")
    input_video = config_env.get("Input", "input_video")
    seq_name = os.path.splitext(os.path.basename(input_video))[0]

    det_output_path = f"{output_path}/{seq_name}/1. det" # VD: ../Outputs/video2/1. det
    reid_output_path = f"{output_path}/{seq_name}/2. det_feat" # VD: ../Outputs/video2/2. det_feat
    mode = config_env.get("General", "mode")
    data2model = config_env.get("Model", "data2model")
    model_name = config_env.get("Model", "model")
    config_path = config_env.get("ReID", "config_path")
    weight_path = config_env.get("ReID", "weight_path")
    input_path = config_env.get("Path", "input_path")
    img_path = f"{input_path}/image_seq"
     

    for nms in [0.80, 0.95]:
        nms_thr = nms
        
        # Get arguments
        filtered_argv = [arg for arg in sys.argv if not arg.startswith('--HistoryManager')]
        args = make_parser(det_output_path, reid_output_path, nms_thr, img_path, mode, data2model, model_name, config_path, weight_path).parse_args(filtered_argv[1:])
    
        # Set random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
    
        # Get encoder
        embedder = EmbeddingComputer(config_path=args.config_path, weight_path=args.weight_path)
    
        # Read detection
        with open(args.pickle_path, 'rb') as f:
            detections = pickle.load(f)
        # Feature extraction
        for vid_name in detections.keys():
            for frame_id in tqdm(detections[vid_name].keys()):
                # If there is no detection
                if detections[vid_name][frame_id] is None:
                    continue
    
                # Read image
                img = cv2.imread(args.data_path + vid_name + '/img1/%06d.jpg' % frame_id)
    
                # Get detection
                detection = detections[vid_name][frame_id]
    
                # Get features
                if detection is not None:
                    embedding = embedder.compute_embedding(img, detection[:, :4])
                    detections[vid_name][frame_id] = np.concatenate([detection, embedding], axis=1)
    
                # Logging
                # print(vid_name, frame_id, flush=True)
    
        # Save
        with open(args.output_path, 'wb') as handle:
            pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
