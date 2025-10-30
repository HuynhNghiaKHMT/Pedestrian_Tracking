import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import trackeval
import numpy as np

# Randomly select bbox color for each object id
color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(5000)]


def set_parameters(args, vid_name, mode, model_type):
    # Set properly for each dataset
    if 'MOT17' in vid_name:
        # Path
        if mode == 'val':
            args.pickle_path = args.pickle_dir + f'mot17_val_0.8{model_type}.pickle'
            args.pickle_path_95 = args.pickle_dir + f'mot17_val_0.95{model_type}.pickle'
            args.data_path = args.data_dir + 'MOT17/train/'
        else:
            args.pickle_path = args.pickle_dir + f'mot17_test_0.8{model_type}.pickle'
            args.pickle_path_95 = args.pickle_dir + f'mot17_test_0.95{model_type}.pickle'
            args.data_path = args.data_dir + 'MOT17/test/'

        if '01' in vid_name or '03' in vid_name or '12' in vid_name:
            args.det_thr, args.init_thr = 0.65, 0.75
        elif '07' in vid_name:
            args.det_thr, args.init_thr = 0.60, 0.60
        elif '14' in vid_name:
            args.det_thr, args.init_thr = 0.45, 0.55
        else:
            args.det_thr, args.init_thr = 0.60, 0.70
        args.match_thr = 0.70

    elif 'MOT20' in vid_name:
        if mode == 'val':
            args.pickle_path = args.pickle_dir + f'mot20_val_0.8{model_type}.pickle'
            args.pickle_path_95 = args.pickle_dir + f'mot20_val_0.95{model_type}.pickle'
            args.data_path = args.data_dir + 'MOT20/train/'
        else:
            args.pickle_path = args.pickle_dir + f'mot20_test_0.8{model_type}.pickle'
            args.pickle_path_95 = args.pickle_dir + f'mot20_test_0.95{model_type}.pickle'
            args.data_path = args.data_dir + 'MOT20/test/'

        if '08' in vid_name:
            args.det_thr, args.init_thr = 0.30, 0.40
        if '04' in vid_name or '06' in vid_name or '07' in vid_name:
            args.det_thr, args.init_thr = 0.40, 0.50
        else:
            args.det_thr, args.init_thr = 0.40, 0.40
        args.match_thr = 0.55
        
    else:
        if mode == 'val':
            args.pickle_path = args.pickle_dir + f'seq_0.8{model_type}.pickle'
            args.pickle_path_95 = args.pickle_dir + f'seq_0.95{model_type}.pickle'
            args.data_path = args.data_dir + 'image_seq/test/'
        else:
            args.pickle_path = args.pickle_dir + f'seq_0.8{model_type}.pickle'
            args.pickle_path_95 = args.pickle_dir + f'seq_0.95{model_type}.pickle'
            args.data_path = args.data_dir + 'image_seq/test/'

        # Baseline Setting
        args.det_thr = 0.60
        args.init_thr = 0.60
        args.match_thr = 0.80 if mode == 'val' else 0.60

def write_results(filename, results):
    # Set save format
    save_format = '{frame}, {id}, {x1}, {y1}, {w}, {h}, {s}, -1, -1, -1\n'

    # Open file
    f = open(filename, 'w')

    # Write
    for frame_id, track_ids, x1y1whs, scores in results:
        for track_id, x1y1wh, score in zip(track_ids, x1y1whs, scores):
            # Get box
            x1, y1, w, h = x1y1wh

            # Generate line to write
            line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                                      w=round(w, 1), h=round(h, 1), s=round(score, 2))

            # Write
            f.write(line)

    # Close
    f.close()


def evaluate(args, trackers_to_eval, dataset):
    # Set evaluation configurations
    eval_config = {'USE_PARALLEL': True,
                   'NUM_PARALLEL_CORES': 8,
                   'BREAK_ON_ERROR': True,
                   'RETURN_ON_ERROR': False,
                   'LOG_ON_ERROR': '../outputs/error_log.txt',

                   'PRINT_RESULTS': False,
                   'PRINT_ONLY_COMBINED': False,
                   'PRINT_CONFIG': False,
                   'TIME_PROGRESS': False,
                   'DISPLAY_LESS_PROGRESS': True,

                   'OUTPUT_SUMMARY': False,
                   'OUTPUT_EMPTY_CLASSES': False,
                   'OUTPUT_DETAILED': False,
                   'PLOT_CURVES': False}

    dataset_config = {'GT_FOLDER': args.data_path,
                      'TRACKERS_FOLDER': args.output_dir,
                      'OUTPUT_FOLDER': None,
                      'TRACKERS_TO_EVAL': [trackers_to_eval],
                      'CLASSES_TO_EVAL': ['pedestrian'],
                      'BENCHMARK': dataset if 'MOT' in dataset else 'MOT17',
                      'SPLIT_TO_EVAL': 'val',
                      'INPUT_AS_ZIP': False,
                      'PRINT_CONFIG': False,
                      'DO_PREPROC': True,
                      'TRACKER_SUB_FOLDER': '',
                      'OUTPUT_SUB_FOLDER': '',
                      'TRACKER_DISPLAY_NAMES': None,
                      'SEQMAP_FOLDER': None,
                      'SEQMAP_FILE': './trackeval/seqmap/%s/val.txt' % dataset.lower(),
                      'SEQ_INFO': None,
                      'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
                      'SKIP_SPLIT_FOL': True}

    # Set configuration
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = evaluator.evaluate(dataset_list, metrics_list)

    # Get
    hota = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']).item()
    idf1 = res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['Identity']['IDF1']
    mota = res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA']
    assa = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['AssA']).item()
    deta = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['DetA']).item()

    # Print
    print('%.3f %.3f %.3f %.3f %.3f' % (hota * 100, idf1 * 100, mota * 100, assa * 100, deta * 100), flush=True)