import os
import sys
sys.path.append(os.path.dirname(__file__))
import argparse
import torch
import random
import numpy as np
import pickle
import configparser
from yolox.exp import get_exp
from yolox.core import launch
from yolox.utils import fuse_model
import torch.backends.cudnn as cudnn
from yolox.evaluators import DetEvaluator
from torch.nn.parallel import DistributedDataParallel as DDP

def yolox_parser(nms_thr, exp_path, seq_name, data, output_path, weight_dir, model_name):
    parser = argparse.ArgumentParser("YOLOX")

    # Can be changed
    parser.add_argument("-f","--dummy", default="", type=str, help="dummy arg")
    parser.add_argument("-e", "--exp_file", default=f"{exp_path}/yolox_{seq_name}.py",
                        type=str, help="pls input your experiment description file",)
    parser.add_argument("-c", "--ckpt", 
                        default=f"{weight_dir}/{data}.pth.tar",
                        type=str, help="ckpt for eval")
    parser.add_argument("-n","--exp_name", type=str,
                        default=f"{output_path}/seq_{str(nms_thr)}_{model_name}.pickle")

    # Fixed
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")
    parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn",)
    parser.add_argument("--fp16", dest="fp16", default=True, action="store_true",)

    # distributed
    parser.add_argument("-t", "--type", default=None, type=str)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training",)
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model",)
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER,)

    # det args
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=nms_thr, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--min_box_area", default=100, type=int, help="filter out tiny boxes")
    parser.add_argument("--seed", default=10000, type=int, help="eval seed")

    return parser


def main(exp, args, num_gpu):
    rank = args.local_rank
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')

    if args.seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    if torch.cuda.is_available():
        cudnn.benchmark = True

    
    # detection
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    model = model.to(device)
    model.eval()

    
    if not args.speed and not args.trt:
        ckpt_file = args.ckpt
        if torch.cuda.is_available():
            loc = "cuda:{}".format(rank)
        else:
            loc = 'cpu' 
        ckpt = torch.load(ckpt_file, map_location=loc, weights_only=False)
        model.load_state_dict(ckpt["model"])
    if is_distributed:
        model = DDP(model, device_ids=[rank])
    if args.fuse:
        model = fuse_model(model)

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    evaluator = DetEvaluator(args=args, dataloader=val_loader, img_size=exp.test_size, conf_thresh=exp.test_conf,
                             nms_thresh=exp.nmsthre, num_classes=exp.num_classes,)

    # start evaluate, x1y1x2y2
    det_results = evaluator.detect(model, args.fp16)

    with open(args.exp_name, 'wb') as f:
        pickle.dump(det_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
def detect():
    # Config
    config_env = configparser.ConfigParser()
    config_env.read('../env.ini')

    data2model = config_env.get("Model", "data2model")
    model_name = config_env.get("Model", "model")
    
    # Get video name
    output_path = config_env.get("Path", "output_path")
    input_video = config_env.get("Input", "input_video")
    seq_name = os.path.splitext(os.path.basename(input_video))[0]

    det_output_path = f"{output_path}/{seq_name}/1. det" # VD: ../Outputs/video2/1. det
    weight_path = config_env.get("YOLOX", "weight_path")
    exp_path = config_env.get("YOLOX", "exp_path")

    for nms in [0.80, 0.95]:
        nms_thr = nms

        # Tạo argument
        filtered_argv = [arg for arg in sys.argv if not arg.startswith('--HistoryManager')]
        args = yolox_parser(nms_thr, exp_path, seq_name, data2model, det_output_path, weight_path, model_name).parse_args(filtered_argv[1:])

        exp = get_exp(args.exp_file)
        exp.merge(args.opts)
    
        if not torch.cuda.is_available():
            num_gpu = 0
        else:
            num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
        assert num_gpu <= torch.cuda.device_count()
        launch(
            main,
            num_gpu,
            args.num_machines,
            args.machine_rank,
            backend=args.dist_backend,
            dist_url=args.dist_url,
            args=(exp, args, num_gpu),
        )