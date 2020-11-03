import numpy as np
import os
import torch
import h5py
import open3d
# from chamfer_distance import ChamferDistance
from config import cfg
from extensions.chamfer_dist import ChamferDistance
from models.grnet import GRNet



gt_path = '/raid/wuruihai/GRNet_FILES/xkh/Completion3D/val/gt/03001627/'
pred_path = '/raid/wuruihai/GRNet_FILES/Results/Completion3D_grnet_alldata_ep300_npz_small_2048d/'  # 0.0030
# pred_path = '/raid/wuruihai/GRNet_FILES/Results/Completion3D_grnet_data_ep300_npz_2048d/'  # 0.0033

# rescale
def rescale_pc_parts(full_part_pc, num_points=2048):
    # full_part_pc = full_part_pc.view(-1, 3)
    now_points = full_part_pc.shape[0]
    while (now_points < num_points):
        full_part_pc = full_part_pc.repeat(2, 1)
        now_points = now_points * 2
    if now_points > num_points:
        idx_selected = np.arange(now_points)
        np.random.shuffle(idx_selected)
        full_part_pc = full_part_pc[idx_selected[:num_points]]
    return full_part_pc


chamferLoss = ChamferDistance()
chamfer_dists = []
avg_chamfer_dist = []

n_points = 2048
n_shape = 1

grnet = GRNet(cfg)
grnet.eval()

# ShapeNet: pred & gt 都是 16384, 不用rescale
# 但是16384的维度 可能会导致cd值大？

for view in range(1):
    # print("------------------- view: %d ---------------------" % view)
    for root, dirs, files in os.walk(pred_path):
        len_files = len(files)
        all_gt = np.zeros((len_files, n_points, 3))
        all_pred = np.zeros((len_files, n_points, 3))
        idx = -1
        tot = 0

        pred_batch = np.zeros((1, n_points, 3))
        gt_batch = np.zeros((1, n_points, 3))

        chamfer_dist = ChamferDistance(ignore_zeros=True)

        for file in files:
            idx += 1
            file_id = os.path.splitext(file)[0]

            pred = np.load(pred_path + file)['pts']
            all_pred[idx] = pred.reshape(n_points, 3)
            pred_batch[0] = np.array(pred, dtype=np.float32)

            # gt = open3d.io.read_point_cloud(gt_path + file_id + '.pcd')  # ShapeNet
            gt = h5py.File(gt_path + file_id + '.h5', 'r')['data'][:]  # Completion3D
            gt = np.array(gt).astype(np.float32)
            all_gt[idx] = gt.reshape(n_points, 3)
            gt_batch[0] = np.array(gt, dtype=np.float32)

            with torch.no_grad():
                cd = chamfer_dist(torch.tensor(pred_batch, dtype=torch.float32).cuda(), torch.tensor(gt_batch, dtype=torch.float32).cuda())

            # cd1, cd2 = chamferLoss(torch.tensor(all_gt[idx], dtype=torch.float32).view(1, n_points, 3), torch.tensor(all_pred[idx], dtype=torch.float32).view(1, n_points, 3))
            # cd = ((cd1.mean() + cd2.mean()) / 2).item()
            # print(cd)
            print(cd)
            tot += cd

        print('avg: ', tot / len_files)
        # cd = chamferLoss(torch.tensor(all_gt, dtype = torch.float32), torch.tensor(all_pred, dtype = torch.float32))
        # chamfer_dict = ChamferDistance()
        cd2 = chamfer_dist(torch.tensor(all_gt, dtype = torch.float32).cuda(), torch.tensor(all_pred, dtype = torch.float32).cuda())
        print('avg: ', cd2)
        # cd = ((cd1.sqrt().mean() + cd2.sqrt().mean()) / 2 ).item()
        # print(cd)


