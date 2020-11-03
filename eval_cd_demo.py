import numpy as np
import os
import torch
from chamfer_distance import ChamferDistance
import open3d as o3d

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# ShapeNet, 不需要再rescale
gt_path = '/raid/wuruihai/GRNet_FILES/cal_cd_test/2254_gt.npz'   # 初始的gt
grnet_path = '/raid/wuruihai/GRNet_FILES/cal_cd_test/2254_grnet.npz'  # output的16384转到2048
pcn_path = '/raid/wuruihai/GRNet_FILES/cal_cd_test/2254_PCN_F.npz'

root = '/raid/wuruihai/GRNet_FILES/xkh/ShapeNetCompletion/test/complete/03001627/'
root2 = '/raid/wuruihai/GRNet_FILES/Results/ShapeNet_grnet_model_npz/'

gt_path = '/raid/wuruihai/GRNet_FILES/xkh/ShapeNetCompletion/test/complete/03001627/124ef426dfa0aa38ff6069724068a578.pcd'   # 初始的gt
grnet_path = '/raid/wuruihai/GRNet_FILES/Results/ShapeNet_grnet_model_npz/124ef426dfa0aa38ff6069724068a578.npz'  # output的16384转到2048


def get_fscore(pred_pc, full_pc, th=0.01):

    sum_fscore = 0.0
    for i in range(full_pc.shape[0]):
        full_np = full_pc[i]
        pred_np = pred_pc[i]

        pred = o3d.geometry.PointCloud()
        pred.points = o3d.utility.Vector3dVector(pred_np)
        gt = o3d.geometry.PointCloud()
        gt.points = o3d.utility.Vector3dVector(full_np)

        dist1 = pred.compute_point_cloud_distance(gt)
        dist2 = gt.compute_point_cloud_distance(pred)

        recall = float(sum(d < th for d in dist2)) / float(len(dist2))
        precision = float(sum(d < th for d in dist1)) / float(len(dist1))

        sum_fscore += 2 * recall * precision / (recall + precision) if recall + precision else 0

    return sum_fscore / full_pc.shape[0]


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

# all_gt = np.zeros((1, n_points, 3))
# all_grnet = np.zeros((1, n_points, 3))
# all_pcn = np.zeros((1, n_points, 3))
#
# gt = np.load(gt_path)['full_pts']
# grnet = np.load(grnet_path)['pts']/0.45
# pcn = np.load(pcn_path)['pts']
#
# all_gt[0] = gt
# all_grnet[0] = grnet
# all_pcn[0] = pcn
#
# print('type: ', type(gt))
# pc1 = o3d.geometry.PointCloud()
# pc1.points = o3d.utility.Vector3dVector(grnet)
# pc2 = o3d.geometry.PointCloud()
# pc2.points = o3d.utility.Vector3dVector(gt)
#
# dist1 = pc1.compute_point_cloud_distance(pc2)
# dist2 = pc2.compute_point_cloud_distance(pc1)
#
# th = 0.01
# recall = float(sum(d < th for d in dist2)) / float(len(dist2))
# precision = float(sum(d < th for d in dist1)) / float(len(dist1))
# F_score = 2 * recall * precision / (recall + precision) if recall + precision else 0

tot_fscore = 0.0
n_file = 0
for root, dirs, files in os.walk(root):
    n_file = len(files)
    for file in files:
        file_name = os.path.splitext(file)[0]

        gt = o3d.io.read_point_cloud(root + file)
        gt = np.array(gt.points, dtype=np.float32)[None, :, :]
        pred = np.load(root2 + file_name + '.npz')['pts']
        pred = np.array(pred, dtype=np.float32)
        F_score = get_fscore(pred, gt)
        tot_fscore += F_score

print(tot_fscore / n_file)

# pred = np.load(grnet_path)['pts']
# pred = np.array(pred, dtype=np.float32)
# gt = o3d.io.read_point_cloud(gt_path)
# gt = np.array(gt.points, dtype=np.float32)[None, :, :]
# # gt = np.load(gt_path)['data']
# # gt = np.array(gt, dtype=np.float32)[None, :, :]
# print(gt.shape)
# F_score = get_fscore(pred, gt)
#
# print('grnet: ', F_score)

# cd1, cd2 = chamferLoss(torch.tensor(all_gt, dtype=torch.float32), torch.tensor(all_grnet, dtype=torch.float32))
# cd1, cd2 = chamferLoss(torch.tensor(all_gt, dtype=torch.float32), torch.tensor(all_pcn, dtype=torch.float32))
# cd = ((cd1.sqrt().mean() + cd2.sqrt().mean()) / 2).item()
# print('cd: ', cd)

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


