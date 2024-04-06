# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import, division, print_function

import numpy as np
from datasets.pascal_voc import pascal_voc
from datasets.imagenet import imagenet
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.clipart import clipart
from datasets.kitti_car import kitti_car
# from datasets.jgw import jgw
# from datasets.jgw_coco import jgw_coco
from datasets.water import water
from datasets.sim10k import sim10k
__sets = {}


# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'trainval_aug', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ["2007"]:
    for split in ["trainval", "train", "test"]:
        name = "clipart_{}".format(split)
        __sets[name] = lambda split=split: clipart(split, year)

for year in ['2012']:
    for split in ["train", "train_aug"]:
        name = "kitti_car_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: kitti_car(split, year)
        
for year in ['2012']:
    for split in ["train"]:
        name = "sim10k_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: sim10k(split, year)

for year in ["2007"]:
    for split in ["train_s", "train_all", "train_t", "test_t"]:
        name = "cityscape_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: cityscape(split, year)
        
# for year in ["2007"]:
# #     for split in ["train_tb", "train_mb800", "test_mb200"]:
# #     for split in ["train_tb", "train_mb", "test_mb"]:
#     for split in ["train_tb2000v2", "train_mb2000v2", "test_tb"]:
#         name = "jgw_{}_{}".format(year, split)
#         __sets[name] = lambda split=split, year=year: jgw(split, year)

for year in ["2007"]:
    for split in ["train_s", "train_t", "test_s", "test_t", "train_aug"]:
        name = "cityscape_car_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: cityscape_car(split, year)
        
for year in ['2007']:
    for split in ['train', 'test']:
        name = 'water_{}'.format(split)
        __sets[name] = (lambda split=split : water(split,year))
    
# for split in ['train_tb8895','train_tb2000', 'test_tb','train_mb2000','test_mb400','train_zp','test_zp']:
# #     name = 'jgw_od_{}'.format(split)
#     name = 'jgw_{}'.format(split)
#     __sets[name] = (lambda split=split, year=year: jgw_coco(split))
    
def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError("Unknown dataset: {}".format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())