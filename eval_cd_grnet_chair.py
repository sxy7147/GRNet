import numpy as np
import os
import torch
import open3d
from chamfer_distance import ChamferDistance



gt_path = '/raid/wuruihai/GRNet_FILES/xkh/ShapeNetCompletion/test/complete/03001627/'
grnet_path = '/raid/wuruihai/GRNet_FILES/Results/grnet_model_npz/'


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

n_points = 16384
n_shape = 1

# ShapeNet: pred & gt 都是 16384, 不用rescale
# 但是16384的维度 可能会导致cd值大？

for view in range(1):
    print("------------------- view: %d ---------------------" % view)
    for root, dirs, files in os.walk(grnet_path):
        len_files = len(files)
        all_gt = np.zeros((len_files, n_points, 3))
        all_pred = np.zeros((len_files, n_points, 3))
        idx = -1
        tot = 0

        for file in files:
            idx += 1
            file_id = os.path.splitext(file)[0]

            pred = np.load(grnet_path + file)['pts']
            pred = pred.reshape(16384, 3)
            # pred = rescale_pc_parts(pred/0.45, n_points)
            all_pred[idx] = pred.reshape(n_points, 3)

            gt = open3d.io.read_point_cloud(gt_path + file_id + '.pcd')
            gt = np.array(gt.points).astype(np.float32)
            gt = gt.reshape(16384, 3)
            # gt = rescale_pc_parts(gt/0.45, n_points)
            all_gt[idx] = gt.reshape(n_points, 3)

            cd1, cd2 = chamferLoss(torch.tensor(all_gt[idx], dtype=torch.float32).view(1, n_points, 3), torch.tensor(all_pred[idx], dtype=torch.float32).view(1, n_points, 3))
            cd = ((cd1.mean() + cd2.mean()) / 2).item()
            print(cd)
            tot += cd

        print(tot / len_files)
        cd1, cd2 = chamferLoss(torch.tensor(all_gt, dtype = torch.float32), torch.tensor(all_pred, dtype = torch.float32))
        cd = ((cd1.mean() + cd2.mean()) / 2 ).item()
        print(cd)


