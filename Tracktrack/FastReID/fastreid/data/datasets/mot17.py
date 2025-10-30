import re
import glob
import os.path as osp
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MOT17(ImageDataset):
    dataset_dir = "mot17"
    dataset_name = "mot17"

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'patch')
        train = self.process_dir(self.train_dir)

        super(MOT17, self).__init__(train, [], [], **kwargs)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)_([\d]+)[.]')

        data = []
        for img_path in img_paths:
            obj_id, cam_id, frame_id = map(int, pattern.search(img_path).groups())
            obj_id = self.dataset_name + "_" + str(obj_id)
            cam_id = self.dataset_name + "_" + str(cam_id)
            data.append((img_path, obj_id, cam_id))

        return data
