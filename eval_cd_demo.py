import numpy as np
import os
import torch
from chamfer_distance import ChamferDistance

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# ShapeNet, 不需要再rescale
gt_path = '/raid/wuruihai/GRNet_FILES/cal_cd_test/2254_gt.npz'   # 初始的gt
grnet_path = '/raid/wuruihai/GRNet_FILES/cal_cd_test/2254_grnet.npz'  # output的16384转到2048
pcn_path = '/raid/wuruihai/GRNet_FILES/cal_cd_test/2254_PCN_F.npz'


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

all_gt = np.zeros((1, n_points, 3))
all_grnet = np.zeros((1, n_points, 3))
all_pcn = np.zeros((1, n_points, 3))

gt = np.load(gt_path)['full_pts']
grnet = np.load(grnet_path)['pts']/0.45
pcn = np.load(pcn_path)['pts']

all_gt[0] = gt
all_grnet[0] = grnet
all_pcn[0] = pcn

cd1, cd2 = chamferLoss(torch.tensor(all_gt, dtype=torch.float32), torch.tensor(all_grnet, dtype=torch.float32))
cd1, cd2 = chamferLoss(torch.tensor(all_gt, dtype=torch.float32), torch.tensor(all_pcn, dtype=torch.float32))
cd = ((cd1.sqrt().mean() + cd2.sqrt().mean()) / 2).item()
print('cd: ', cd)

# all_gt[idx] = gt.reshape(2048, 3)
# all_pred[idx] = output.reshape(2048, 3)

# cd1, cd2 = chamferLoss(torch.tensor(all_gt[idx], dtype=torch.float32).view(1, 2048, 3), torch.tensor(all_pred[idx], dtype=torch.float32).view(1, 2048, 3))
# cd = ((cd1.mean() + cd2.mean()) / 2).item()
# print(cd)
# tot += cd
# cd1, cd2 = chamferLoss(torch.tensor(all_gt, dtype=torch.float32), torch.tensor(all_pred, dtype=torch.float32))
# cd = ((cd1.sqrt().mean() + cd2.sqrt().mean()) / 2).item()
# print("view_%d: " % view, cd)
# print(tot / len_files)
# chamfer_dists.append(cd)

# print("avg_CD: ", float(sum(chamfer_dists) / float(len(chamfer_dists))))


