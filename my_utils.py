from patch_conv import patch_conv
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import open3d as o3d
import chamfer3D.dist_chamfer_3D, fscore
from plot_utils import *
#from chamfer_distance import ChamferDistance
#from time import time
import copy
import cv2
from sklearn.cluster import KMeans, estimate_bandwidth, DBSCAN
from sklearn.cluster import MeanShift
def creat_patch_conv(heat_map, batch_size, device, num_patch=1, test_time=1, kernal_size=200): # 75
    patch_list = []
    patch_select = []
    heat_map = heat_map.view(1, -1, heat_map.shape[0], heat_map.shape[1])
    patch_sum = patch_conv(kernal_size=kernal_size).to(device)
    # patch_sum.half()
    out = patch_sum(heat_map)
    print(out.shape)
    out = out.float()
    out_t = out.flatten()
    sorted, _ = torch.sort(out_t)
    out = out.cpu().detach()
    sorted = sorted.cpu().detach().numpy()
    sorted = sorted[::-1]
    patch_xy = np.zeros([1, 2])
    ind = np.argwhere(out == sorted[0])
    print("max_entropy", sorted[0])

    x1 = int(ind[3, 0])
    y1 = int(ind[2, 0])
    patch_list.append([x1, y1])
    #ind = np.argwhere(out == sorted[-1])
    #print("min_entropy", sorted[-1])
    #x2 = int(ind[3, 0])
    #y2 = int(ind[2, 0])
    #patch_list.append([int(x1+kernal_size/2), int(y1+kernal_size/2)])
    return patch_list

def entropy_map(pred):
    #entropy_map-->[h, w, cls]-->[640, 1024, 17]-->[640, 1024]
    cls = pred.shape[2]
    #pred = torch.sigmoid(pred)
    nc_log = torch.log(pred)
    nc_entropy = torch.mul(pred, nc_log)
    nc_entropy = torch.where(torch.isnan(nc_entropy), torch.full_like(nc_entropy, 0), nc_entropy)
    #entropy_heat_map = torch.add(torch.sum(nc_map, dim=1), torch.sum(conf_entropy, dim=1))
    entropy_heat_map = torch.sum(nc_entropy, dim=2) / cls
    #entropy_heat_map = torch.sum(conf_entropy, dim=1)
    return -entropy_heat_map

def project_xy_world(xy_list, viewMat, projMat):
    x1 = xy_list[0]
    y1 = xy_list[1]
    z = xy_list[2]
    width = 1024
    height = 640
    viewMat = np.array(viewMat).reshape((4, 4), order='F')
    projMat = np.array(projMat).reshape((4, 4), order='F')
    x = (2 * x1 - width) / width
    y = -(2 * y1 - height) / height
    z = 2 * z - 1
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
    tran_pix_world = np.linalg.inv(projMat @ viewMat)
    position = tran_pix_world @ pix_pos.T
    position = position.T
    position[:, :] /= position[:, 3:4]
    #print("position", position.shape)
    return position

def show_ec_map(entropy_map, path):
    plt.matshow(entropy_map.view(-1, entropy_map.size(-1)).detach().cpu().numpy().astype(np.int16), interpolation='nearest')
    #plt.matshow(entropy_map.astype(np.int16), interpolation='nearest')
    #plt.show()
    plt.savefig(path, transparent=True, dpi=800)
def show_ec_map_numpy(entropy_map, path):
    #plt.matshow(entropy_map.view(-1, entropy_map.size(-1)).detach().cpu().numpy().astype(np.int16), interpolation='nearest')
    plt.matshow(entropy_map.astype(np.int16), interpolation='nearest')
    plt.savefig(path, transparent=True, dpi=800)
def get_heightmap(surface_pts, workspace_limits, segment_heat_map, segment_results, heightmap_resolution=0.002):

    entropy_pts = segment_heat_map.reshape([-1, 1])
    mask_pts = segment_results.reshape([-1, 1])
    # Compute heightmap size

    heightmap_size = np.ceil(((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
                               (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)).astype(int)
    # Sort surface points by z value

    sort_z_ind = np.argsort(surface_pts[:, 2])
    entropy_pts = entropy_pts[sort_z_ind]
    mask_pts = mask_pts[sort_z_ind]
    surface_pts = surface_pts[sort_z_ind]
    # Filter out surface points outside heightmap boundaries

    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(surface_pts[:, 0] > workspace_limits[0][0], surface_pts[:, 0] < workspace_limits[0][1]),
        surface_pts[:, 1] > workspace_limits[1][0]), surface_pts[:, 1] < workspace_limits[1][1]),
                                         surface_pts[:, 2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    entropy_pts = entropy_pts[heightmap_valid_ind]
    mask_pts = mask_pts[heightmap_valid_ind]
    entropy_heightmap = np.zeros((heightmap_size[0], heightmap_size[1], 1))
    mask_heightmap = np.zeros((heightmap_size[0], heightmap_size[1], 1))
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
    entropy_heightmap[heightmap_pix_y, heightmap_pix_x] = entropy_pts[:, [0]]
    mask_heightmap[heightmap_pix_y, heightmap_pix_x] = mask_pts[:, [0]]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return depth_heightmap, entropy_heightmap.reshape([entropy_heightmap.shape[0], -1]), mask_heightmap.reshape([mask_heightmap.shape[0], -1])

def get_target_center(entropy_heightmap, workspace_limits_one, workspace_limits, heightmap_resolution=0.002):
    heightmap_pix_x1 = np.floor((workspace_limits_one[0][0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    heightmap_pix_y1 = np.floor((workspace_limits_one[1][0] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    heightmap_pix_x2 = np.floor((workspace_limits_one[0][1] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    heightmap_pix_y2 = np.floor((workspace_limits_one[1][1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    entropy_heightmap_one = entropy_heightmap[heightmap_pix_y1:heightmap_pix_y2, heightmap_pix_x1:heightmap_pix_x2]
    print(entropy_heightmap_one.shape)
    print(entropy_heightmap.shape)
    y, x = np.where(entropy_heightmap == entropy_heightmap_one.max())
    target_center = (int(np.mean(x)), int(np.mean(y)))
    return target_center

def get_patch(img, img_xy, patch_size=200):
    x1 = img_xy[0]
    y1 = img_xy[1]
    patch = img[y1:y1+patch_size, x1:x1+patch_size]
    return patch
def get_shift(start_list, shift):
    start_point_1 = start_list[-2]
    start_point_2 = start_list[-1]
    x1 = start_point_1[0, 0]
    y1 = start_point_1[0, 1]
    x2 = start_point_2[0, 0]
    y2 = start_point_2[0, 1]
    distance = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    if distance < 0.05:
        shift = shift + 0.05
    return shift
def get_point(depthImg, viewMat, projMat):
    viewMat = np.array(viewMat).reshape((4, 4), order='F')
    projMat = np.array(projMat).reshape((4, 4), order='F')
    width = depthImg.shape[1]
    height = depthImg.shape[0]
    x = (2 * np.arange(0, width) - width) / width
    x = np.repeat(x[None, :], height, axis=0)
    y = -(2 * np.arange(0, height) - height) / height
    y = np.repeat(y[:, None], width, axis=1)
    z = 2 * depthImg - 1
    pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
    tran_pix_world = np.linalg.inv(projMat @ viewMat)
    position = tran_pix_world @ pix_pos.T
    position = position.T
    position[:, :] /= position[:, 3:4]
    return position
def get_point_view(depthImg, viewMat, projMat):
    viewMat = np.array(viewMat).reshape((4, 4), order='F')
    projMat = np.array(projMat).reshape((4, 4), order='F')
    width = depthImg.shape[1]
    height = depthImg.shape[0]
    x = (2 * np.arange(0, width) - width) / width
    x = np.repeat(x[None, :], height, axis=0)
    y = -(2 * np.arange(0, height) - height) / height
    y = np.repeat(y[:, None], width, axis=1)
    z = 2 * depthImg - 1
    pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
    tran_pix_world = np.linalg.inv(projMat @ viewMat)
    position = tran_pix_world @ pix_pos.T
    position = position.T
    view_factor = position[:, 3:4].copy()
    position[:, :] /= position[:, 3:4]
    return position, view_factor

def get_worklimits(position, depthImg, point_list, robot_workspace_limits, patch_size=200):
    depth_mask = np.zeros_like(depthImg)
    robot_workspace_limits_one = robot_workspace_limits.copy()
    depth_mask[point_list[0][1]:point_list[0][1] + patch_size, point_list[0][0]:point_list[0][0] + patch_size] = 1
    data = np.concatenate((position[:, :3], depth_mask.reshape([-1, 1])), axis=1)
    data = data[data[:, 3] > 0]

    point_max = data[:, :3].max(axis=0)
    point_min = data[:, :3].min(axis=0)

    robot_workspace_limits_one[0][0] = point_min[0]
    robot_workspace_limits_one[0][1] = point_max[0]

    robot_workspace_limits_one[1][0] = point_min[1]
    robot_workspace_limits_one[1][1] = point_max[1]

    return robot_workspace_limits_one

def get_pts_from_mask(position, mask):
    data = np.concatenate((position[:, :3], mask.reshape([-1, 1])), axis=1)
    data = data[data[:, 3] == 1]
    return data
def get_pts_from_mask_filter(position, mask_id, voxel_id=0):
    data = position.copy()
    data = data[data[:, 4] == mask_id]
    data_xyz = data[:, :3]
    if data_xyz.shape[0] <= 25:
        filter_data_one = np.zeros([1])
        return filter_data_one
    else:
        # ms = KMeans(n_clusters=4, random_state=0).fit(data_xyz)
        bandwidth = estimate_bandwidth(data_xyz, quantile=0.2, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data_xyz)

        labels = ms.labels_
        # cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        number_list = []
        for i in range(n_clusters_):
            my_members = labels == i
            my_data = data[my_members]
            number_list.append(my_data.shape[0])
        max_value = max(number_list)
        max_idx = number_list.index(max_value)
        filter_data_one = data[labels == max_idx]
        return filter_data_one

def push_generator(point_list, valid_depth_heightmap, entropy_heightmap, task=0):

    # [100, 100]
    area_shape_default = [200, 200]
    H, W = valid_depth_heightmap.shape
    target_center = (int(point_list[0][0]+area_shape_default[0]/2), int(point_list[0][1]+area_shape_default[0]/2))
    #print((H, W))
    #print(valid_depth_heightmap.shape)

    if task == 0:
        #height_target = valid_depth_heightmap[target_center[1], target_center[0]]
        height_target = get_max_height(valid_depth_heightmap, target_center, area_shape_default)
        print(height_target)
        #area_shape_default = [100, 100]

    else:
        #height_target = get_average_height(valid_depth_heightmap, target_center, area_shape_default)
        #height_target = get_max_height(valid_depth_heightmap, target_center, area_shape_default)
        height_target = valid_depth_heightmap[target_center[1], target_center[0]]
        area_shape_default = [100, 100]
    target_center_x = target_center[0]
    target_center_y = target_center[1]

    proposal_area_x = np.arange(max(0, target_center_x - area_shape_default[0] / 2),
                                min(target_center_x + area_shape_default[0] / 2, W-1))
    proposal_area_y = np.arange(max(0, target_center_y - area_shape_default[1] / 2),
                                min(target_center_y + area_shape_default[1] / 2, H-1))
    # Get the indices of the area of interest
    area_indices = np.zeros((len(proposal_area_x), len(proposal_area_y), 2), dtype=np.int)
    for i, x in enumerate(proposal_area_x):
        for j, y in enumerate(proposal_area_y):
            area_indices[i, j] = [x, y]
    area_indices = area_indices.reshape((-1, 2))
    # Searching for suitable starting points as candidates
    # det_inds_mat[x, y]
    det_inds_mat = np.zeros((4, len(proposal_area_x) * len(proposal_area_y), 2))
    det_height_mat = np.zeros((4, len(proposal_area_x) * len(proposal_area_y)))
    push_ind_mat = area_indices.copy()
    vshift = 6
    hshift = 6
    translations = np.zeros((2, 2, 2))
    for i in range(translations.shape[0]):
        for j in range(translations.shape[1]):
            translations[i, j] = [(2 * j - 1) * hshift, (2 * i - 1) * vshift]
    for i in range(2):
        for j in range(2):
            det_inds_mat[2 * i + j, :] = np.floor((push_ind_mat + translations[i, j]))
    det_inds_mat = det_inds_mat.astype(np.int)
    for i in range(2):
        for j in range(2):
            idx = tuple([det_inds_mat[2 * i + j, :, 1], det_inds_mat[2 * i + j, :, 0]])
            idx[0][idx[0] >= H] = H-1
            idx[0][idx[0] < 0] = 0
            idx[1][idx[1] >= W] = W-1
            idx[1][idx[1] < 0] = 0
            det_height_mat[2 * i + j, :] = valid_depth_heightmap[idx]
    height_det = np.max(det_height_mat, axis=0)
    valid = (height_target - height_det >= 0.015)
    # height_score
    height_score_map= height_target - valid_depth_heightmap
    #height_score = height_score.reshape([area_shape_default[0], -1])

    valid_push_inds = np.where(valid == True)[0].flatten()
    num_valid_push = valid_push_inds.shape[0]
    if num_valid_push >= 1:
        quadrants = [[], [], [], []]
        valid_push_ids = area_indices[valid_push_inds]
        for idx in valid_push_ids:
            if idx[0] <= target_center[0] and idx[1] <= target_center[1]:
                quadrants[0].append(idx)
            elif idx[0] > target_center[0] and idx[1] <= target_center[1]:
                quadrants[1].append(idx)
            elif idx[0] <= target_center[0] and idx[1] > target_center[1]:
                quadrants[2].append(idx)
            elif idx[0] > target_center[0] and idx[1] > target_center[1]:
                quadrants[3].append(idx)

        candidates = []
        for sector in quadrants:
            #if len(sector) > 10:
            #    sampled_push = random.sample(sector, 10)
            #    candidates.append(sampled_push)
            #else:
            candidates.append(sector)
        all_candidates = []
        all_candidates_entropy = []
        all_candidates_height = []
        for sector in candidates:
            for start_point in sector:
                all_candidates.append(start_point)
                x0 = start_point[0]
                y0 = start_point[1]
                all_candidates_entropy.append(entropy_heightmap[y0][x0])
                # [9, 2]
                #rowIndex = np.where((area_indices == [y0, x0]).all(axis=1))
                all_candidates_height.append(height_score_map[y0][x0])

        all_candidates_entropy = np.asarray(all_candidates_entropy)
        all_candidates_height = np.asarray(all_candidates_height)
        all_score = np.multiply(all_candidates_height, all_candidates_entropy)
        id_max = np.argsort(all_score)[-1]
        push_score = all_score[id_max]
        print("push score", all_score[id_max])
        push_start_point = all_candidates[id_max]
    else:
        print("SCT faile")
        if target_center[0] < W/2:
            push_start_point = [np.max((target_center[0] - int(area_shape_default[0]/4), 0)), target_center[1]]
        else:
            push_start_point = [np.min((target_center[0] + int(area_shape_default[0]/4), W-1)), target_center[1]]
        push_start_point = np.array(push_start_point)
    return push_start_point, target_center, push_score

def push_generator_random(point_list, valid_depth_heightmap, entropy_heightmap, task=0):
    area_shape_default = [200, 200]
    H, W = valid_depth_heightmap.shape
    push_start_point = []

    start_x = np.random.randint(0, W - 200)
    start_y = np.random.randint(0, H - 200)
    target_center = (int(start_x + area_shape_default[0] / 2), int(start_y + area_shape_default[0] / 2))
    height_target = get_max_height(valid_depth_heightmap, target_center, area_shape_default)
    push_start_point.append(start_x)
    push_start_point.append(start_y)
    return push_start_point, target_center, height_target


def get_push_end(start_point, end_point, shift=0.15):
    # start_point: [x, y, z]
    # end_point: [[x, y, z]]
    x1 = start_point[0]
    y1 = start_point[1]
    z1 = start_point[2]
    x2 = end_point[0]
    y2 = end_point[1]
    z2 = end_point[2]
    r = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) + 0.0001
    end_y = (shift*(y2-y1))/r + y1
    end_x = (shift*(x2-x1))/r + x1
    end_point = np.array([[end_x, end_y, z1]])
    return end_point

def get_push_slow_list(start_point, end_point):
    x1 = start_point[0][0]
    y1 = start_point[0][1]
    x2 = end_point[0][0]
    y2 = end_point[0][1]
    x_list = np.linspace(x1, x2, num=5)
    y_list = np.linspace(y1, y2, num=5)
    return x_list[1:], y_list[1:]

def get_average_height(valid_depth_heightmap, target_center, area_shape_default):
    H, W = valid_depth_heightmap.shape
    target_center_x = target_center[0]
    target_center_y = target_center[1]
    x1 = max(0, target_center_x - area_shape_default[0] / 2)
    y1 = max(0, target_center_y - area_shape_default[1] / 2)
    x2 = min(target_center_x + area_shape_default[0] / 2, W - 1)
    y2 = min(target_center_y + area_shape_default[1] / 2, H - 1)
    height = np.mean(valid_depth_heightmap[int(y1):int(y2), int(x1):int(x2)])
    return height

def get_max_height(valid_depth_heightmap, target_center, area_shape_default):
    H, W = valid_depth_heightmap.shape
    target_center_x = target_center[0]
    target_center_y = target_center[1]
    x1 = max(0, target_center_x - area_shape_default[0] / 2)
    y1 = max(0, target_center_y - area_shape_default[1] / 2)
    x2 = min(target_center_x + area_shape_default[0] / 2, W - 1)
    y2 = min(target_center_y + area_shape_default[1] / 2, H - 1)
    height = np.max(valid_depth_heightmap[int(y1):int(y2), int(x1):int(x2)])
    return height

def get_point_score(position, workspace_limits, num_class=27, test_time=0):
    score_list = []
    score_list_print = []
    position_print = position.copy()
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(position[:, 0] > workspace_limits[0][0], position[:, 0] < workspace_limits[0][1]),
        position[:, 1] > workspace_limits[1][0]), position[:, 1] < workspace_limits[1][1]),
        position[:, 2] < workspace_limits[2][1])
    position = position[heightmap_valid_ind]
    for i in range(1, num_class+1):
        data = position[position[:, 3] == i]
        data_print = position_print[position_print[:, 3]==i]

        if data.shape[0] <= 400:
            score_list.append(100)
            continue
        label_path = os.path.join(base_path, object_name[:-5], 'google_16k', label_name)
        target = o3d.io.read_point_cloud(label_path, format="ply")
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(data[:, :3])

        voxel_size = 0.03  # means 5cm for this dataset
        distance_threshold = voxel_size * 0.5
        trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，
                                 [0, 1, 0, 0],  #
                                 [0, 0, 1, 0],  #
                                 [0, 0, 0, 1]])
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        source.transform(result_ransac.transformation)
        # ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        source.transform(reg_p2p.transformation)

        source_value = np.asarray(source.points)
        target_value = np.asarray(target.points)

        source_value = torch.from_numpy(source_value).to(torch.float32).cuda()
        target_value = torch.from_numpy(target_value).to(torch.float32).cuda()
        source_value = source_value.unsqueeze(0)
        target_value = target_value.unsqueeze(0)
        chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        dist1, dist2, idx1, idx2 = chamLoss(target_value, source_value)
        f_score, precision, recall = fscore.fscore(dist1, dist2)
        score_list.append(f_score.cpu().numpy()[0])
        score_list_print.append(f_score.cpu().numpy()[0])

    return score_list, score_list_print
def get_point_trans(position, workspace_limits, base_path, obj_mesh_list, obj_label_list, num_class=27, test_time=0):
    trans_list = []
    #position_print = position.copy()
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(position[:, 0] > workspace_limits[0][0], position[:, 0] < workspace_limits[0][1]),
        position[:, 1] > workspace_limits[1][0]), position[:, 1] < workspace_limits[1][1]),
        position[:, 2] < workspace_limits[2][1])
    position = position[heightmap_valid_ind]
    for i in range(1, num_class + 1):
        data = position[position[:, 3] == i]
        object_name = obj_mesh_list[i-1]
        label_name = obj_label_list[i-1]
        label_format = label_name[-3:]

        if data.shape[0] == 0:
            trans_list.append(np.zeros(1))
            continue
        label_path = os.path.join(base_path, object_name[:-5], 'google_16k', label_name)

        target = o3d.io.read_point_cloud(label_path, format=label_format)
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(data[:, :3])

        voxel_size = 0.03  # means 5cm for this dataset
        distance_threshold = voxel_size * 0.5
        trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix
                                 [0, 1, 0, 0],  #
                                 [0, 0, 1, 0],  #
                                 [0, 0, 0, 1]])
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        source.transform(result_ransac.transformation)
        # ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        source.transform(reg_p2p.transformation)
        trans_list.append(reg_p2p.transformation[:3, :3].copy())
    return trans_list
def get_trans_points_score(position, workspace_limits, base_path, obj_mesh_list, obj_label_list, num_class=27, test_time=0):

    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(position[:, 0] > workspace_limits[0][0], position[:, 0] < workspace_limits[0][1]),
        position[:, 1] > workspace_limits[1][0]), position[:, 1] < workspace_limits[1][1]),
        position[:, 2] < workspace_limits[2][1])
    position = position[heightmap_valid_ind]
    score_list_all = []
    object_cls_list = []
    for i in range(1, num_class + 1):
        data = position[position[:, 3] == i]
        if data.shape[0] == 0:
            continue
        score_list = []
        object_cls_list.append(i)
        for j in range(len(obj_mesh_list)):
            object_name = obj_mesh_list[j]
            label_name = obj_label_list[j]
            label_format = label_name[-3:]
            label_path = os.path.join(base_path, object_name[:-5], 'google_16k', label_name)
            target = o3d.io.read_point_cloud(label_path, format=label_format)
            root_base_path = os.getcwd()
            save_path = os.path.join(root_base_path, time_label + '.xyz')
            np.savetxt(save_path,
                   data[:, :3], fmt="%.6f", delimiter=" ")
            source = o3d.io.read_point_cloud(save_path, format="xyz")
            voxel_size = 0.03  # means 5cm for this dataset
            distance_threshold = voxel_size * 0.5
            trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix
                                     [0, 1, 0, 0],  #
                                     [0, 0, 1, 0],  #
                                     [0, 0, 0, 1]])
            source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
            target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
            result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            source.transform(result_ransac.transformation)
            # ICP
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, distance_threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            source.transform(reg_p2p.transformation)
            source_value = np.asarray(source.points)
            target_value = np.asarray(target.points)
            # tensor
            source_value = torch.from_numpy(source_value).to(torch.float32).cuda()
            target_value = torch.from_numpy(target_value).to(torch.float32).cuda()
            source_value = source_value.unsqueeze(0)
            target_value = target_value.unsqueeze(0)
            chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
            dist1, dist2, idx1, idx2 = chamLoss(target_value, source_value)
            f_score, precision, recall = fscore.fscore(dist1, dist2)
            score_list.append(f_score.cpu().numpy()[0]+0.001)
        score_list_all.append(np.asarray(score_list).reshape([1, -1]))
    score_all = np.concatenate(score_list_all, axis=0)
    object_cls_list.insert(0, 0)
    return score_all, object_cls_list

def get_ints_points_score(position_list, workspace_limits, base_path, obj_mesh_list, obj_label_list, num_class=27, test_time=0):
    score_list_all = []
    object_cls_list = []
    target_list = []

    for j in range(len(obj_mesh_list)):
        object_name = obj_mesh_list[j]
        label_name = obj_label_list[j]
        #label_name = "nontextured.ply"
        label_format = label_name[-3:]
        label_path = os.path.join(base_path, object_name[:-5], 'google_16k', label_name)
        target = o3d.io.read_point_cloud(label_path, format=label_format)
        target_list.append(target)
    for i in range(len(position_list)):
        score_list = []
        data = position_list[i]
        if data.shape[0] == 0:
            score_list.append(np.zeros([1, 27]))
            score = np.asarray(score_list)
            object_cls_list.append(np.argmax(score) + 1)
            score_list_all.append(np.asarray(score_list).reshape([1, -1]))
            continue
        for j in range(len(obj_mesh_list)):
            target = target_list[j]

            #target = target.voxel_down_sample(voxel_size=0.03)
            root_base_path = os.getcwd()
            save_path = os.path.join(root_base_path, time_label + '.xyz')
            np.savetxt(save_path,
                data[:, :3], fmt="%.6f", delimiter=" ")
            source = o3d.io.read_point_cloud(save_path, format="xyz")
            voxel_size = 0.03  # means 5cm for this dataset
            distance_threshold = voxel_size * 0.5
            trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，
                                     [0, 1, 0, 0],  #
                                     [0, 0, 1, 0],  #
                                     [0, 0, 0, 1]])
            source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
            target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
            result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            source.transform(result_ransac.transformation)
            # ICP

            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, distance_threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            source.transform(reg_p2p.transformation)

            source_value = np.asarray(source.points)
            target_value = np.asarray(target.points)
            # tensor
            source_value = torch.from_numpy(source_value).to(torch.float32).cuda()
            target_value = torch.from_numpy(target_value).to(torch.float32).cuda()
            source_value = source_value.unsqueeze(0)
            target_value = target_value.unsqueeze(0)
            chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
            dist1, dist2, idx1, idx2 = chamLoss(target_value, source_value)
            f_score, precision, recall = fscore.fscore(dist1, dist2)

            score_list.append(f_score.cpu().numpy()[0]+0.001)
        #
        score = np.asarray(score_list)
        object_cls_list.append(np.argmax(score)+1)
        score_list_all.append(np.asarray(score_list).reshape([1, -1]))
    score_all = np.concatenate(score_list_all, axis=0)
    return score_all, object_cls_list

def get_entropy(score):
    # score[n×27] --> [n×1]
    log_score = np.log(score)
    log_score[np.isnan(log_score)] = 0
    entropy = np.sum(np.multiply(score, log_score), axis=1)/27.0

    entropy = entropy.reshape([-1, 1])
    #entropy = np.insert(entropy, 0, np.zeros([1, 1]), axis=0)
    return -entropy
def get_trans_distance(trans_list, base_path, obj_name_list, obj_trans_list, object_mesh_id, num_class=27):
    score_list = []
    score_list_print = []
    for i in range(0, num_class):
        if trans_list[i].any() == 0:
            score_list.append(-1)
            continue
        else:
            object_name = obj_name_list[i]
            label_path = os.path.join(base_path, object_name[:-5], 'google_16k', 'pos_label.npy')
            label_trans = np.load(label_path)
            obj_id = object_mesh_id.index(i+1) - 1
            obj_trans = obj_trans_list[obj_id]
            C1 = np.dot(label_trans, np.linalg.inv(obj_trans))
            C2 = trans_list[i]

            dist = np.linalg.norm(C2 - C1)
            score_list_print.append(dist)
            score_list.append(dist)
    return score_list, score_list_print
def preprocess_point_cloud(pcd, voxel_size):

    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5

    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(source_down, target_down,
                                                                                    source_fpfh, target_fpfh,
                                                                                    o3d.pipelines.registration.FastGlobalRegistrationOption(
                                                                                        maximum_correspondence_distance=distance_threshold))
    return result


def prepare_dataset(voxel_size, source, target):

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def get_retrieval(gt_list, pred_list):

    gt_cls = np.unique(gt_list)
    gt_number = []
    for i in gt_cls:
        gt_number.append(np.sum(gt_list == i))
    gt_number = np.asarray(gt_number)
    pred_cls = np.unique(pred_list)
    pred_number = []
    for i in pred_cls:
        pred_number.append(np.sum(pred_list == i))
    pred_number = np.asarray(pred_number)
    print("label", ((gt_cls), (gt_number)))
    print("pred", ((pred_cls), (pred_number)))

    pred_r = 0
    for i in range(pred_cls.shape[0]):
        if pred_cls[i] in gt_cls:
            indx = np.where(gt_cls == pred_cls[i])
            pred_r = pred_r + min(gt_number[indx], pred_number[i])
    pred_all = np.sum(pred_number)
    gt_all = np.sum(gt_number)
    P = (pred_r) / pred_all
    R = (pred_r) / gt_all
    F1_score = (2 * P * R) / (P + R)
    return  P, R, F1_score

def get_gt_label(segImg, object_mesh_id, obj_id_list):
    segImg = np.asarray(segImg)
    label = np.asarray(object_mesh_id)
    #obj_list = np.asarray(obj_id_list)
    bbox_list = []
    mask_list = []
    num_crowd = 0
    w = segImg.shape[1]
    h = segImg.shape[0]
    obj_list = np.unique(segImg)
    for obj_id in obj_list[1:]:
        y, x = np.where(segImg == obj_id)
        xmin = x.min()
        ymin = y.min()
        xmax = x.max()
        ymax = y.max()
        bbox_list.append(np.asarray([xmin, ymin, xmax, ymax, label[obj_id]-1]).reshape([1, -1]))
        patch = segImg.copy()
        patch[patch == obj_id] = 1
        patch[patch != 1] = 0
        mask_list.append(patch.reshape([1, patch.shape[0], -1]))

    gt_mask = np.concatenate(mask_list, axis=0)
    gt = np.concatenate(bbox_list, axis=0)
    gt = gt.astype(np.float64)
    gt[:, [0, 2]] = gt[:, [0, 2]] / w
    gt[:, [1, 3]] = gt[:, [1, 3]] / h
    return gt, gt_mask, h, w, num_crowd

def get_cd_distance(source, target, device):
    source_value = source[:, :3].copy()
    target_value = target[:, :3].copy()
    source_value = torch.from_numpy(source_value).to(torch.float32).cuda()
    target_value = torch.from_numpy(target_value).to(torch.float32).cuda()
    source_value = source_value.unsqueeze(0)
    target_value = target_value.unsqueeze(0)
    chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = chamLoss(target_value, source_value)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    return loss.cpu().numpy()

def get_view_list(mask_list, mask_list_other, pts, class_list):
    mask_2_4 = []
    obj_pts_2 = []
    pts_2_list = []
    obj_class = []
    for i in mask_list_other:
        mask_2_single = mask_list[i, :, :]
        mask_2_4.append(mask_2_single)
        #pts_one_2 = get_pts_from_mask_filter(pts, i+1)
        pts_one_2 = get_pts_from_mask(pts, mask_2_single)
        if pts_one_2.shape[0] <= 1:
            continue
        obj_pts_2.append(pts_one_2)
        pts_2_list.append(pts_one_2)
        obj_class.append(class_list[i])
    return obj_pts_2, mask_2_4, pts_2_list, obj_class

def get_instance(mask_list_old, pts_list, class_list, device):
    mask_list = copy.deepcopy(mask_list_old)
    num_of_view = len(mask_list)
    for i in range(len(mask_list)):
        mask = mask_list[i]
        for j in range(mask.shape[0]):
            kernel = np.ones((3, 3), np.uint8) # 投稿(3, 3)
            mask_one = mask[j, :, :]
            erosion_1 = cv2.erode(mask_one, kernel, iterations=1)
            mask[j, :, :] = erosion_1
    mask_top = mask_list[0]  # [n, h, w]
    pts_top = pts_list[0]  # [n, 4]-->[n, x, y, z, 1]
    class_top = class_list[0]

    obj_pts = []
    masks_top_inds = []
    obj_pts_final = []
    obj_class_final = []

    other_mask_list = []
    pts_top_list = []
    obj_class = []
    other_class_list = []

    for j in range(mask_top.shape[0]):
        #pts_one_top = get_pts_from_mask_filter(pts_top, mask_top[j, :, :])
        pts_one_top = get_pts_from_mask(pts_top, mask_top[j, :, :])
        pts_top_list.append(pts_one_top)

        obj_pts.append(pts_one_top)

        obj_class.append(class_top[j])
    for view_id in range(1, num_of_view): # 5
        mask_single = mask_list[view_id]
        pts_single = pts_list[view_id]
        class_single = class_list[view_id]
        other_mask_id = []
        other_class_id = []
        for i in range(mask_single.shape[0]):
            #pts_one = get_pts_from_mask_filter(pts_single, mask_single[i, :, :])
            pts_one = get_pts_from_mask(pts_single, mask_single[i, :, :])

            # score = np.ones([mask_top.shape[1]])
            if pts_one.shape[0] <= 0:
                continue
            score = []
            for j in range(mask_top.shape[0]):
                pts_one_top = pts_top_list[j]
                if pts_one_top.shape[0] <= 0:
                    score.append(np.ones(1)[0])
                else:
                    cd_dis = get_cd_distance(pts_one, pts_one_top, device)
                    score.append(cd_dis)

            score = np.asarray(score)
            inds = np.argmin(score)
            score_min = score[inds]
            #print(score_min)
            if score_min > 0.005:
                other_mask_id.append(i)
                other_class_id.append(class_single[i])
            else:
                obj_pts[inds] = np.concatenate((obj_pts[inds], pts_one), axis=0)


        other_mask_list.append(other_mask_id)
        other_class_list.append(other_class_id)

    if num_of_view <= 1:
        pts_list_other = []
        mask_other = []
        obj_pts_other = []
        obj_class_other = []
    elif num_of_view <= 2:
        pts_list_other = []
        obj_pts_other = other_mask_list[0]
        obj_class_other = other_class_list[0]
    elif num_of_view <= 4:
        obj_pts_2, mask_2, pts_2_list, obj_2_class = get_view_list(mask_list[2], other_mask_list[1], pts_list[2], class_list[2])
        mask_other = mask_2
        obj_pts_other = obj_pts_2
        pts_list_other = pts_2_list
        obj_class_other = obj_2_class
    else:
        obj_pts_2, mask_2, pts_2_list, obj_2_class = get_view_list(
            mask_list[2], other_mask_list[1], pts_list[2], class_list[2])
        obj_pts_4, mask_4, pts_4_list, obj_4_class = get_view_list(
            mask_list[4], other_mask_list[3], pts_list[4], class_list[4])
        # obj_pts_1, mask_1, pts_1_list = get_view_list(mask_list[1], other_mask_list[0], pts_list[1])
        # obj_pts_3, mask_3, pts_3_list = get_view_list(mask_list[3], other_mask_list[2], pts_list[3])
        mask_other = mask_2 + mask_4
        obj_pts_other = obj_pts_2 + obj_pts_4
        pts_list_other = pts_2_list + pts_4_list
        obj_class_other = obj_2_class + obj_4_class

    if len(pts_list_other) > 0:
        other_mask_list_2 = []
        for view_id in [1, 3]:
            if view_id >= num_of_view:
                break
            masks_inds = other_mask_list[view_id - 1]
            mask_single = mask_list[view_id]
            pts_single = pts_list[view_id]
            other_mask_id = []
            for i in masks_inds:
                #pts_one = get_pts_from_mask_filter(pts_single, mask_single[i, :, :])
                pts_one = get_pts_from_mask(pts_single, mask_single[i, :, :])
                # score = np.ones(len(mask_other))
                if pts_one.shape[0] <= 0:
                    continue
                score = []
                for j in range(len(mask_other)):
                    pts_one_other = pts_list_other[j]
                    if pts_one_other.shape[0] <= 0:
                        score.append(np.ones(1)[0])
                    else:
                        cd_dis = get_cd_distance(pts_one, pts_one_other, device)
                        score.append(cd_dis)
                score = np.asarray(score)
                inds = np.argmin(score)
                score_min = score[inds]
                if score_min > 0.005:
                    other_mask_id.append(i)
                else:
                    obj_pts_other[inds] = np.concatenate((obj_pts_other[inds], pts_one), axis=0)

            other_mask_list_2.append(other_mask_id)
        obj_pts_final = []
        obj_class_final = []
        for i in other_mask_list_2[0]:
            #obj_pts_final.append(get_pts_from_mask_filter(pts_list[1], mask_list[1][i, :, :]))
            obj_pts_final.append(get_pts_from_mask(pts_list[1], mask_list[1][i, :, :]))
            obj_class_final.append(class_list[1][i])
        if num_of_view > 3:
            for i in other_mask_list_2[1]:
                #obj_pts_final.append(get_pts_from_mask_filter(pts_list[3], mask_list[3][i, :, :]))
                obj_pts_final.append(get_pts_from_mask(pts_list[3], mask_list[3][i, :, :]))
                obj_class_final.append(class_list[3][i])
    obj_pts_list_all = obj_pts + obj_pts_other + obj_pts_final
    obj_class_all = obj_class + obj_class_other + obj_class_final

    return obj_pts_list_all, obj_class_all

def data_normal_2d(orign_data, dim="col"):
    if dim == "col":
        dim = 1
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    else:
        dim = 0
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data, d_min).true_divide(dst)
    return norm_data

def get_entropy_mask(entropy_heightmap_list, mask_heightmap_list, time = 0):

    num_of_mask = len(mask_heightmap_list)

    entropy_heightmap = np.zeros_like(entropy_heightmap_list[0])
    for i in range(len(entropy_heightmap_list)):
        entropy_heightmap = entropy_heightmap + entropy_heightmap_list[i]

    mask_heightmap_all = np.concatenate(mask_heightmap_list, axis=0)
    mask_heightmap_all = mask_heightmap_all.reshape([num_of_mask, -1]) # 5
    mask_heightmap = np.zeros([mask_heightmap_all.shape[1]])
    for i in range(mask_heightmap_all.shape[1]):
        mask_heightmap[i] = len(np.unique( mask_heightmap_all[:, i]))
    mask_heightmap = mask_heightmap + 0.0001
    mask_heightmap = mask_heightmap.reshape([-1, entropy_heightmap.shape[1]])

    entropy_heightmap = np.multiply(entropy_heightmap, mask_heightmap)
    return entropy_heightmap

def get_tp_mask(masks, class_list, tp_index):
    tp_mask = masks[tp_index]
    tp_class = class_list[tp_index]
    return tp_mask, tp_class
