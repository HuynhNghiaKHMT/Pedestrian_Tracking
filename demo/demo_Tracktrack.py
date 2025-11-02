import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configparser import ConfigParser
from Tracktrack.Tracker.utils.etc import *
# Đường dẫn



def main():
    
    # Đọc file config
    config_env = ConfigParser()
    config_env.read('../env.ini')

    model = config_env.get("Model", "model")


    
    # Tách video thành folder ảnh, tạo các file json, exp
    from Utils.split_video import video_preprocess
    video_preprocess()


    # Tạo folder outputs
    from Utils.create_output import create_output_folder
    create_output_folder()


    print("Detect bằng model: ",model)
    if model == "yoloxx":
        # YOLOX detect
        from Tracktrack.YOLOX.run_detect import detect
        detect()
    else:
        from Tracktrack.YOLOv.run_detect import run
        run()

    # ReID
    print("ReID")
    from Tracktrack.FastReID.ext_feats import reid
    reid()


    # Tracker
    print("Track")
    from Tracktrack.Tracker.run import tracker
    tracker()


    # Tạo video đầu ra
    print("Create video output")
    from Utils.create_output import create_video
    create_video()

    
if __name__ == "__main__":
    main()