import numpy as np
import os
import torch
from chamfer_distance import ChamferDistance

import ipdb

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

'''  grnet的output(dense_cloud)转为2048维，再和gt(val_full_2048)计算每个part的cd，最后summary  '''

# ShapeNet, 不需要再rescale
gt_path = '/raid/wuruihai/GRNet_FILES/zy/Completion3D/full/val/'   # 初始的gt
grnet_root = '/raid/wuruihai/GRNet_FILES/Results/Completion3D_zy_data_ep500_npz/'  # output的16384转到2048


# rescale
def rescale_pc_parts(full_part_pc, num_points=2048):
    full_part_pc = full_part_pc.reshape(-1, 3)
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

for view in range(8):
    # print("------------------- view: %d ---------------------" % view)
    grnet_path = grnet_root + 'part_%d/' % view
    for root, dirs, files in os.walk(grnet_path):
        # ipdb.set_trace()
        len_files = len(files)
        all_gt = np.zeros((len_files, n_points, 3))
        all_pred = np.zeros((len_files, n_points, 3))
        idx = -1
        # tot = 0
        for file in files:
            idx += 1

            output = np.load(grnet_path + file)['pts']
            output = rescale_pc_parts(output, n_points) / 0.45    # rescale & 放大
            gt = np.load(gt_path + file)['arr_0']

            all_gt[idx] = gt.reshape(2048, 3)
            all_pred[idx] = output.reshape(2048, 3)

            # cd1, cd2 = chamferLoss(torch.tensor(all_gt[idx], dtype=torch.float32).view(1, 2048, 3), torch.tensor(all_pred[idx], dtype=torch.float32).view(1, 2048, 3))
            # cd = ((cd1.mean() + cd2.mean()) / 2).item()
            # print(cd)
            # tot += cd
        cd1, cd2 = chamferLoss(torch.tensor(all_gt, dtype=torch.float32), torch.tensor(all_pred, dtype=torch.float32))
        cd = ((cd1.sqrt().mean() + cd2.sqrt().mean()) / 2).item()
        print("view_%d: " % view, cd)
        # print(tot / len_files)
        chamfer_dists.append(cd)

print("avg_CD: ", float(sum(chamfer_dists) / float(len(chamfer_dists))))


