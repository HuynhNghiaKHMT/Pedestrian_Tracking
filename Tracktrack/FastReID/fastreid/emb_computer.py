import os
import sys
import cv2
import torch
import numpy as np
sys.path.append(os.path.dirname(__file__))
from .fastreid_adaptor import FastReID


class EmbeddingComputer:
    def __init__(self, config_path, weight_path, max_batch=32): # 1024
        self.model = None
        self.config_path = config_path
        self.weight_path = weight_path
        self.crop_size = (128, 384)
        self.max_batch = max_batch

    def initialize_model(self):
        self.model = FastReID(self.config_path, self.weight_path)

    def compute_embedding(self, img, bbox):
        # Initialization
        if self.model is None:
            self.initialize_model()

        # Basic embeddings
        h, w = img.shape[:2]
        bbox_clip = np.round(bbox).astype(np.int32)
        bbox_clip[:, 0] = bbox_clip[:, 0].clip(0, w)
        bbox_clip[:, 1] = bbox_clip[:, 1].clip(0, h)
        bbox_clip[:, 2] = bbox_clip[:, 2].clip(0, w)
        bbox_clip[:, 3] = bbox_clip[:, 3].clip(0, h)
        
        # Get patches
        crops = []
        try:
            for box in bbox_clip:
                # Get patch, BGR -> RGB, Resize
                if box[1] == box[3]:
                    if box[3] == h:
                        box[1]-=1
                    else:
                        box[3]+=1
                if box[0] == box[2]:
                    if box[2] == w:
                        box[0]-=1
                    else:
                        box[2]+=1
                crop = img[box[1]:box[3], box[0]:box[2]]
                # print(box[0], box[1], box[2], box[3])
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

                # To Tensor, Append
                crop = torch.as_tensor(crop.transpose(2, 0, 1)).unsqueeze(0).cuda()
                crops.append(crop)
        except (cv2.error, AssertionError) as e:
            print(f"Warning: Bỏ qua embedding do crop lỗi: {e} (box: {crop}) (box: {box[0]}, {box[1]}, {box[2]}, {box[3]})")  # Tùy chọn log

        # To batch
        crops = torch.cat(crops, dim=0)

        # Get embeddings
        embeddings = []
        for idx in range(0, len(crops), self.max_batch):
            # Get batch
            batch_crops = crops[idx:idx + self.max_batch]
            batch_crops = batch_crops.cuda()

            # Inference
            with torch.no_grad():
                batch_embeddings = self.model(batch_crops)

            # Add (extend)
            embeddings.extend(batch_embeddings)

        # Stack, L2 Normalize, To numpy
        embeddings = torch.stack(embeddings)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = embeddings.cpu().numpy()
        return embeddings