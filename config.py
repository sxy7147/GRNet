# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 17:18:04
# @Email:  cshzxie@gmail.com

from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C
# ith_box_combined/Chair-3/
#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.COMPLETION3D                        = edict()
__C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/raid/wuruihai/GRNet_FILES/xkh/Completion3D/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/raid/wuruihai/GRNet_FILES/xkh/Completion3D/%s/gt/%s/%s.h5'
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 8
__C.DATASETS.SHAPENET.N_POINTS                   = 16384
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/raid/wuruihai/GRNet_FILES/xkh/ShapeNetCompletion/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/raid/wuruihai/GRNet_FILES/xkh/ShapeNetCompletion/%s/complete/%s/%s.pcd'
__C.DATASETS.KITTI                               = edict()
__C.DATASETS.KITTI.CATEGORY_FILE_PATH            = './datasets/KITTI.json'
__C.DATASETS.KITTI.PARTIAL_POINTS_PATH           = '/home2/wuruihai/GRNet_FILES/KITTI/cars/%s.pcd'
__C.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH        = '/home2/wuruihai/GRNet_FILES/KITTI/bboxes/%s.txt'

#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars, KITTI
__C.DATASET.TRAIN_DATASET                        = 'ShapeNet'
__C.DATASET.TEST_DATASET                         = 'ShapeNet'
# __C.DATASET.TRAIN_DATASET                        = 'Completion3D'
# __C.DATASET.TEST_DATASET                         = 'Completion3D'
# __C.DATASET.TRAIN_DATASET                        = 'KITTI'
# __C.DATASET.TEST_DATASET                         = 'KITTI'

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.DEVICE                                 = '1'
__C.CONST.NUM_WORKERS                            = 8
__C.CONST.N_INPUT_POINTS                         = 2048

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './output_Completion3D'

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048
__C.NETWORK.GRIDDING_LOSS_SCALES                 = [128]
__C.NETWORK.GRIDDING_LOSS_ALPHAS                 = [0.1]

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 32
__C.TRAIN.N_EPOCHS                               = 300
__C.TRAIN.SAVE_FREQ                              = 5
__C.TRAIN.LEARNING_RATE                          = 1e-4
__C.TRAIN.LR_MILESTONES                          = [50]
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'

