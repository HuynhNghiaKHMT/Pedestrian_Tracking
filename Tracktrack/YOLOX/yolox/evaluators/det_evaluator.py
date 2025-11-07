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
            
            # Detect THE ENTIRE BATCH (images)
            with torch.no_grad():
                images = images.type(tensor_type)
                # outputs sẽ là một list hoặc tensor có kích thước Batch Size x ...
                outputs = model(images)
                # postprocess TRẢ VỀ MỘT LIST CÁC KẾT QUẢ DETECTION cho từng ảnh trong batch
                postprocess_outputs = postprocess(outputs, self.num_classes, self.conf_thresh, self.nms_thresh)

            # LẶP QUA TỪNG KẾT QUẢ ĐƠN LẺ TRONG BATCH
            for i, output in enumerate(postprocess_outputs):
                
                # Lấy thông tin cho hình ảnh thứ i trong batch
                # Cần chắc chắn rằng infos được tổ chức theo Batch Size
                # Ví dụ: infos[2] là tensor chứa [frame_id_0, frame_id_1, ...]
                #        infos[4] là list chứa [path_0, path_1, ...]
                
                try:
                    # Trích xuất thông tin video và frame_id
                    video_name = infos[4][i].split('/')[2]
                    frame_id = int(infos[2][i].item()) # Dùng [i] để lấy phần tử thứ i trong batch

                except IndexError:
                    # Xử lý trường hợp không đủ thông tin cho batch (ít xảy ra nếu dataloader chuẩn)
                    print("Lỗi: Không đủ thông tin infos cho batch size lớn.")
                    continue

                # Khởi tạo (nếu cần)
                if video_name not in det_results.keys():
                    det_results[video_name] = {}
                
                # Xử lý kết quả Detection cho frame_id này
                if output is not None:
                    
                    # Get final confidence (output: x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                    output[:, 4] *= output[:, 5]
                    output[:, 5] = output[:, 6]
                    output = output[:, :6]

                    # Prepare un-normalize size (Lấy thông tin cho ảnh thứ i)
                    img_h, img_w = infos[0][i].item(), infos[1][i].item() # Cần lấy theo index [i]
                    scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
                    
                    # Un-normalize size
                    final_outputs = output.detach().cpu().numpy()
                    final_outputs[:, :4] /= scale

                    # Clip
                    final_outputs = final_outputs[(np.minimum(final_outputs[:, 2], img_w - 1) - np.maximum(final_outputs[:, 0], 0)) > 0]
                    final_outputs = final_outputs[(np.minimum(final_outputs[:, 3], img_h - 1) - np.maximum(final_outputs[:, 1], 0)) > 0]

                    # Save
                    det_results[video_name][frame_id] = final_outputs if len(final_outputs) > 0 else None

                # If there is no detection result
                else:
                    det_results[video_name][frame_id] = None

        return det_results
