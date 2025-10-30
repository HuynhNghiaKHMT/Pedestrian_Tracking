import torch
import numpy as np
from tqdm import tqdm
from yolox.utils import (is_main_process, postprocess,)

# Trả về kết quả detection trên 1 bức ảnh

class DetEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(self, args, dataloader, img_size, conf_thresh, nms_thresh, num_classes):
        self.dataloader = dataloader
        self.img_size = img_size

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = num_classes

        self.args = args

    def detect(self, model, half=False):
        # To half
        if torch.cuda.is_available():
            tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        else:
            tensor_type = torch.HalfTensor if half else torch.FloatTensor
        model = model.eval()
        if half:
            model = model.half()

        # Initialize
        det_results = {}

        # Detect
        for images, _, infos, ids in tqdm(self.dataloader):
            # Get video name and frame index
            video_name = infos[4][0].split('/')[2]
            frame_id = int(infos[2].item())

            # Initialize
            if video_name not in det_results.keys():
                det_results[video_name] = {}

            # Detect
            # outputs: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            with torch.no_grad():
                images = images.type(tensor_type)
                outputs = model(images) # Lấy output từ images
                outputs = postprocess(outputs, self.num_classes, self.conf_thresh, self.nms_thresh)[0]

            if outputs is not None:
                # Get final confidence
                outputs[:, 4] *= outputs[:, 5]
                outputs[:, 5] = outputs[:, 6]
                outputs = outputs[:, :6]

                # Prepare un-normalize size
                img_h, img_w = infos[0], infos[1]
                img_h, img_w = float(img_h), float(img_w)
                scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))

                # Un-normalize size
                outputs = outputs.detach().cpu().numpy()
                outputs[:, :4] /= scale

                # Clip
                outputs = outputs[(np.minimum(outputs[:, 2], img_w - 1) - np.maximum(outputs[:, 0], 0)) > 0]
                outputs = outputs[(np.minimum(outputs[:, 3], img_h - 1) - np.maximum(outputs[:, 1], 0)) > 0]

                # Save
                det_results[video_name][frame_id] = outputs if len(outputs) > 0 else None

            # If there is no detection result
            else:
                det_results[video_name][frame_id] = None

        return det_results
