import numpy as np
import os
import torch
from chamfer_distance import ChamferDistance

import ipdb

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

gt_path = '/raid/wuruihai/GRNet_FILES/zy/ShapeNetCompletion/full/val_2048/'
# grnet_root = '/raid/wuruihai/GRNet_FILES/Results/zy_chair_ep300_2048d/'
grnet_root = '/raid/wuruihai/GRNet_FILES/Results/zy_chair_ep150_2048d_300/'



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
    print("------------------- view: %d ---------------------" % view)
    grnet_path = grnet_root + 'part_%d/' % view
    for root, dirs, files in os.walk(grnet_path):
        # ipdb.set_trace()
        len_files = len(files)
        all_gt = np.zeros((len_files, n_points, 3))
        all_pred = np.zeros((len_files, n_points, 3))
        idx = -1
        tot = 0
        for file in files:
            idx += 1
            # print(idx, len_files)
            output = np.load(grnet_path + file)['pts']
            gt = np.load(gt_path + file)['arr_0']
            # print(output.shape)
            # print(gt.shape)
            all_gt[idx] = gt.reshape(2048, 3)
            all_pred[idx] = output.reshape(2048, 3)
            '''
            cd1, cd2 = chamferLoss(torch.tensor(all_gt[idx], dtype=torch.float32).view(1, 2048, 3), torch.tensor(all_pred[idx], dtype=torch.float32).view(1, 2048, 3))
            cd = ((cd1.mean() + cd2.mean()) / 2).item()
            print(cd)
            if cd > 0.01:
                # print(file, ' , ', cd)
                file_id = os.path.splitext(file)[0]
                print(file_id, end=', ')
            tot += cd
            '''
        cd1, cd2 = chamferLoss(torch.tensor(all_gt, dtype=torch.float32), torch.tensor(all_pred, dtype=torch.float32))
        cd = ((cd1.sqrt().mean() + cd2.sqrt().mean()) / 2).item()
        print('cd: ', cd)
        # print(tot / len_files)
        chamfer_dists.append(cd)

    '''
    for idx in range(len(chamfer_dists)):
        print(idx, chamfer_dists[idx])
    '''
    # avg_chamfer_dist.append(float(sum(chamfer_dists) / float(len(chamfer_dists))))  # avg in every views
    # print(avg_chamfer_dist[view])

for view in range(8):
    print("view_%d: " % view, chamfer_dists[view])

print("avg_CD: ", float(sum(chamfer_dists) / float(len(chamfer_dists))))


