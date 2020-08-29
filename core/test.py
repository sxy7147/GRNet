# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:29:37
# @Email:  cshzxie@gmail.com

import logging
import torch
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
# import scipy.misc
# import cv2 as cv
# import numpy as np


import utils.data_loaders
import utils.helpers

from extensions.chamfer_dist import ChamferDistance
from extensions.gridding_loss import GriddingLoss
from models.grnet import GRNet
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from PIL import Image


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, grnet=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        # 在data_loader.py中修改这里的dataset值
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
                                                       collate_fn=utils.data_loaders.collate_fn,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
    if grnet is None:
        grnet = GRNet(cfg)

        if torch.cuda.is_available():
            grnet = torch.nn.DataParallel(grnet).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        grnet.load_state_dict(checkpoint['grnet'])

    # Switch models to evaluation mode
    grnet.eval()

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
                                 alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)    # lgtm [py/unused-import]

    # Testing loop
    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['SparseLoss', 'DenseLoss'])
    test_metrics = AverageMeter(Metrics.names())  # 'F-score, CD
    category_metrics = dict()


    # Testing loop
    # 通过data得到sparse_pucloud,  data from test_data_loader
    for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            sparse_ptcloud, dense_ptcloud = grnet(data)
            sparse_loss = chamfer_dist(sparse_ptcloud, data['gtcloud'])
            dense_loss = chamfer_dist(dense_ptcloud, data['gtcloud'])
            test_losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            _metrics = Metrics.get(dense_ptcloud, data['gtcloud']) # return: values
            test_metrics.update(_metrics)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)


            # train时不用存数据
            '''
            save_path = '/home2/wuruihai/GRNet_FILES/Results/grnet_all_ep150/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 只存了 final results (dense_ptcloud)
            save_npz_path = save_path+'part_1/'
            if not os.path.exists(save_npz_path):
                os.makedirs(save_npz_path)
            dense_pts = np.array(dense_ptcloud.cpu())
            np.savez(save_npz_path + '%s.npz' % model_id, pts = dense_pts)
            '''

            '''
            plt.figure()

            sparse_ptcloud = sparse_ptcloud.squeeze().cpu().numpy()
            sparse_ptcloud_img = utils.helpers.get_ptcloud_img(sparse_ptcloud)
            matplotlib.image.imsave(save_path+'%s_sps.png' % model_id,
                                    sparse_ptcloud_img)

            dense_ptcloud = dense_ptcloud.squeeze().cpu().numpy()
            dense_ptcloud_img = utils.helpers.get_ptcloud_img(dense_ptcloud)
            matplotlib.image.imsave(save_path+'%s_dns.png' % model_id,
                                    dense_ptcloud_img)

            gt_ptcloud = data['gtcloud'].squeeze().cpu().numpy()
            gt_ptcloud_img = utils.helpers.get_ptcloud_img(gt_ptcloud)
            matplotlib.image.imsave(save_path+'%s_gt.png' % model_id,
                                    gt_ptcloud_img)
            '''

            '''
            if model_idx in range(510, 600):

                now_num=model_idx-499
                # if test_writer is not None and model_idx < 3:
                # sparse_ptcloud = sparse_ptcloud.squeeze().cpu().numpy()
                sparse_ptcloud = sparse_ptcloud.squeeze().numpy()
                sparse_ptcloud_img = utils.helpers.get_ptcloud_img(sparse_ptcloud)
                matplotlib.image.imsave('/home2/wuruihai/GRNet_FILES/results2/%s_%s_sps.png'%(model_idx,model_id), sparse_ptcloud_img)

                # dense_ptcloud = dense_ptcloud.squeeze().cpu().numpy()
                dense_ptcloud = dense_ptcloud.squeeze().numpy()
                dense_ptcloud_img = utils.helpers.get_ptcloud_img(dense_ptcloud)
                matplotlib.image.imsave('/home2/wuruihai/GRNet_FILES/results2/%s_%s_dns.png' % (model_idx, model_id),
                                        dense_ptcloud_img)


                # gt_ptcloud = data['gtcloud'].squeeze().cpu().numpy()
                gt_ptcloud = data['gtcloud'].squeeze().numpy()
                gt_ptcloud_img = utils.helpers.get_ptcloud_img(gt_ptcloud)
                matplotlib.image.imsave('/home2/wuruihai/GRNet_FILES/results2/%s_%s_gt.png'%(model_idx,model_id), gt_ptcloud_img)

                cv.imwrite("/home2/wuruihai/GRNet_FILES/out3.png", sparse_ptcloud_img)
                im = Image.fromarray(sparse_ptcloud_img).convert('RGB')
                im.save("/home2/wuruihai/GRNet_FILES/out.jpeg")
            
                test_writer.add_image('Model%02d/SparseReconstruction' % model_idx, sparse_ptcloud_img, epoch_idx)
                dense_ptcloud = dense_ptcloud.squeeze().cpu().numpy()
                dense_ptcloud_img = utils.helpers.get_ptcloud_img(dense_ptcloud)
                test_writer.add_image('Model%02d/DenseReconstruction' % model_idx, dense_ptcloud_img, epoch_idx)

                gt_ptcloud = data['gtcloud'].squeeze().cpu().numpy()
                gt_ptcloud_img = utils.helpers.get_ptcloud_img(gt_ptcloud)
                test_writer.add_image('Model%02d/GroundTruth' % model_idx, gt_ptcloud_img, epoch_idx)
            '''

            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                            ], ['%.4f' % m for m in _metrics]))
    plt.show()
    plt.savefig('/raid/wuruihai/GRNet_FILES/results.png')
    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(1), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return Metrics(cfg.TEST.METRIC_NAME, test_metrics.avg())
