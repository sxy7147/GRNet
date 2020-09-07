import numpy as np
import os


def calculate_two_pts_distance(pts1, pts2):
    pts1 = np.array(pts1, dtype='float32')
    pts2 = np.array(pts2, dtype='float32')
    A = np.tile(np.expand_dims(np.sum(pts1**2, axis=1), axis=-1), [1, pts2.shape[0]])
    B = np.tile(np.expand_dims(np.sum(pts2**2, axis=1), axis=0), [pts1.shape[0], 1])
    C = np.dot(pts1, pts2.T)
    dist = A + B - 2 * C
    return dist

def propagate_a_label(shapenet_np, partnet_pts, partnet_label):
    assert len(partnet_label) == 2048
    # ipdb.set_trace()
    # print()
    dist = calculate_two_pts_distance(partnet_pts, shapenet_np)
    idx = np.argmin(dist, axis=0)
    shapenet_label = partnet_label[idx]
    return shapenet_label


root_dir = '/Users/gongbeida/Documents/GitHub/data/'
# grnet_dir = os.path.join(root_dir, 'GRNet_outputs/ShapeNet_zy_chair_ep500_npz_2048d/')    # ShapeNet
# target_dir = os.path.join(root_dir, 'GRNet_outputs/ShapeNet_zy_chair_ep500_npz_sem_2048d/') 
grnet_dir = os.path.join(root_dir, 'GRNet_outputs/Completion3D_zy_data_ep500_npz_2048d/')   # Completion3D
target_dir = os.path.join(root_dir, 'GRNet_outputs/Completion3D_zy_chair_ep500_npz_sem_2048d/')  # Completion3D

gt_dir = os.path.join(root_dir, 'Chair3_new_npz/val/')


part_name = 'part_0/'

grnet_dir = grnet_dir + part_name
target_dir = target_dir + part_name
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


for root, dirs, files in os.walk(grnet_dir):
    for file in files:
        print(file)
        pred_pts = np.load(grnet_dir + file)['pts']
        gt = np.load(gt_dir + file)
        gt_pts = gt['full_pts']
        gt_labels = gt['full_gt_label']  # 2048个点, 每个点所属的label的编号
        pred_labels = propagate_a_label(pred_pts, gt_pts, gt_labels)

        sem_list = np.array([i for i in range(39)])
        pred_mask = np.zeros((39, 2048), dtype=np.bool)
        tot_used = 0
        for idx_sem in range(39):
            idxes = np.where(pred_labels == idx_sem)[0]
            if idxes.shape[0] > 0:
                pred_mask[tot_used, idxes] = True
                sem_list[tot_used] = idx_sem
                tot_used += 1
        # print(tot_used)
        np.savez(target_dir + file, pts=pred_pts, pred_mask=pred_mask[:tot_used], pred_sem=sem_list[:tot_used])


        
