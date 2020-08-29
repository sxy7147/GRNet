import numpy as np
import os
import torch
from chamfer_distance import ChamferDistance



'''  grnet的output(dense_cloud)转为2048维，再和gt(val_full_2048)计算每个part的cd，最后summary  '''

gt_path = '/raid/wuruihai/GRNet_FILES/zy/ShapeNetCompletion/full/val_2048/'
grnet_root = '/raid/wuruihai/GRNet_FILES/Results/zy_chair_ep150_npz/'


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

n_shape = 2048

for view in range(8):
    print("------------------- view: %d ---------------------" % view)
    grnet_path = grnet_root + 'part_%d/' % view
    for root, dirs, files in os.walk(grnet_path):
        for file in files:
            output = np.load(grnet_path + file)['pts']
            output = rescale_pc_parts(output, n_shape)    # rescale
            gt = np.load(gt_path + file)['arr_0']
            cd1, cd2 = chamferLoss(torch.tensor(gt, dtype=torch.float32).view(n_shape, -1, 3), torch.tensor(output, dtype=torch.float32).view(n_shape, -1, 3))
            cd = ((cd1.mean() + cd2.mean()) / 2).item()
            print(cd)
            chamfer_dists.append(cd)

    '''
    for idx in range(len(chamfer_dists)):
        print(idx, chamfer_dists[idx])
    '''
    avg_chamfer_dist.append(float(sum(chamfer_dists) / float(len(chamfer_dists))))  # avg in every views

for view in range(8):
    print("view_%d: " % view, avg_chamfer_dist[view])

print("avg_CD: ", float(sum(avg_chamfer_dist) / float(len(avg_chamfer_dist))))


