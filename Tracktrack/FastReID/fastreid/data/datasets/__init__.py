# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Person re-id datasets
from fastreid.data.datasets.mot17 import MOT17
from fastreid.data.datasets.mot17_half import MOT17_half
from fastreid.data.datasets.mot20 import MOT20
from fastreid.data.datasets.mot20_half import MOT20_half
from fastreid.data.datasets.dancetrack import DanceTrack

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
