import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import pickle
import argparse
import configparser
from AFLink.AppFreeLink import *
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from trackers.tracker import Tracker
from utils.etc import *
from utils.gbi import gb_interpolation


def make_parser(reid_output, track_output, input_path, mode):
    parser = argparse.ArgumentParser("Tracker")

    # Basic
    parser.add_argument("-f","--dummy", default="", type=str, help="dummy arg")
    parser.add_argument("--cmc_path", type=str, default=None)
    parser.add_argument("--pickle_dir", type=str, default=f"{reid_output}/")
    parser.add_argument("--output_dir", type=str, default=f"{track_output}")
    parser.add_argument("--data_dir", type=str, default=f"{input_path}/")
    parser.add_argument("--dataset", type=str, default=f"image_seq")
    parser.add_argument("--mode", type=str, default=mode)
    parser.add_argument("--seed", type=float, default=10000)

    # For trackers
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--min_box_area", type=float, default=100)
    parser.add_argument("--max_time_lost", type=float, default=30)
    parser.add_argument("--penalty_p", type=float, default=0.20)
    parser.add_argument("--penalty_q", type=float, default=0.40)
    parser.add_argument("--reduce_step", type=float, default=0.05)
    parser.add_argument("--tai_thr", type=float, default=0.55)

    return parser


def track(args, detections, detections_95, data_path, result_folder, mode, model_type):
    # For each video
    total_time, total_count = 0, 0
    for vid_name in detections.keys():
        # Set proper parameters
        set_parameters(args, vid_name, mode, model_type)

        # Set max time lost
        seq_info = open(data_path + vid_name + '/seqinfo.ini', mode='r')  # Input/image_seq/test/video2/seqinfo.ini
        for s_i in seq_info.readlines():
            if 'frameRate' in s_i:
                args.max_time_lost = int(s_i.split('=')[-1]) * 2
            if 'imWidth' in s_i:
                args.img_w = int(s_i.split('=')[-1])
            if 'imHeight' in s_i:
                args.img_h = int(s_i.split('=')[-1])

        # Set tracker
        tracker = Tracker(args, vid_name)

        # For each frame
        results = []
        for frame_id in detections[vid_name].keys():
            # Run tracking
            start = time.time()
            if detections[vid_name][frame_id] is not None:
                track_results = tracker.update(detections[vid_name][frame_id], detections_95[vid_name][frame_id])
            else:
                track_results = tracker.update_without_detections()
            total_time += time.time() - start
            total_count += 1

            # Filter out the results
            x1y1whs, track_ids, scores = [], [], []
            for t in track_results:
                # Check aspect ratio
                if 'MOT' in data_path and t.x1y1wh[2] / t.x1y1wh[3] > 1.6:
                    continue

                # Check track id, minimum box area
                if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                    x1y1whs.append(t.x1y1wh)
                    track_ids.append(t.track_id)
                    scores.append(t.score)

            # Merge
            results.append([frame_id, track_ids, x1y1whs, scores])

        # Logging & Write results
        result_filename = os.path.join(result_folder, '{}.txt'.format(vid_name))
        write_results(result_filename, results)

    return total_time, total_count


def run(args, model_type, af_link):
    # Initialize AFLink
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PostLinker()
    model.load_state_dict(torch.load(f'{af_link}', map_location=device))
    aflink_dataset = LinkData('', '')

    # Logging & Set proper parameters
    print('Running %s %s...' % (args.dataset, args.mode))
    set_parameters(args, args.dataset, args.mode, model_type)
    # Make result folder
    trackers_to_eval = args.pickle_path.split('/')[-1].split('.pickle')[0]
    result_folder = os.path.join(args.output_dir, trackers_to_eval)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(result_folder + '_post/', exist_ok=True)

    # Read detection result
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)
    with open(args.pickle_path_95, 'rb') as f:
        detections_95 = pickle.load(f)

    # Track
    total_time, total_count = track(args, detections, detections_95, args.data_path, result_folder, args.mode, model_type)

    # Post-processing
    print('Running post-processing...')
    for result_file in os.listdir(result_folder):
        # Set Path
        path_in = result_folder + '/' + str(result_file)
        path_out = result_folder + '_post/' + str(result_file)

        # Link
        if 'Dance' in args.dataset:
            linker = AFLink(path_in=path_in, path_out=path_out, model=model, dataset=aflink_dataset,
                            thrT=(0, 20), thrS=100, thrP=0.05)
            linker.link()

        # Gaussian Interpolation
        if 'MOT' in args.dataset:
            gb_interpolation(path_in, path_out, interval=30, tau=12)

    # Logging
    print(total_count / total_time, flush=True)
    print('', flush=True)


def tracker():
    # Config
    config_env = configparser.ConfigParser()
    config_env.read('../env.ini')

    # Get video name
    output_path = config_env.get("Path", "output_path")
    input_video = config_env.get("Input", "input_video")
    seq_name = os.path.splitext(os.path.basename(input_video))[0]

    reid_output_path = f"{output_path}/{seq_name}/2. det_feat" # VD: ../Outputs/video2/2. det_feat
    track_output_path = f"{output_path}/{seq_name}/3. track" # VD: ../Outputs/video2/3. track

    mode = config_env.get("General", "mode")
    input_path = config_env.get("Path", "input_path")
    model_type = config_env.get("Model", "type")
    af_link = config_env.get("Track", "af_link")
    
    
    # Get arguments
    filtered_argv = [arg for arg in sys.argv if not arg.startswith('--HistoryManager')]
    args = make_parser(reid_output_path, track_output_path, input_path, mode).parse_args(filtered_argv[1:])

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Run
    run(args, model_type, af_link)